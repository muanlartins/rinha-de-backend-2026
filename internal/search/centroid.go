package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// scoreAllCentroidsGeneric is the pure-Go reference implementation of
// the centroid-distance fold. It is the parity oracle for the asm path
// in centroid_amd64.s and the fallback on non-amd64 builds.
//
// centroids is SoA: centroids[d*K + c] is dim d of cluster c.
//
// This is also good autovectorized Go on GOAMD64=v3: the inner loop
// unrolls and the compiler emits VFMADD231PS for the per-dim fold,
// achieving ~2.5x over a naive triple-nested loop. The hand-written asm
// shaves a further ~30-40 % on top by keeping the 14 query broadcasts
// register-resident (avoids re-broadcast or stack spill per batch).
func scoreAllCentroidsGeneric(q *[dataset.Dims]float32, centroids []float32, K int, out []float32) {
	if len(out) < K {
		panic("out buffer too small")
	}
	for c := 0; c < K; c++ {
		out[c] = 0
	}
	for d := 0; d < dataset.Dims; d++ {
		qd := q[d]
		base := d * K
		for c := 0; c < K; c++ {
			diff := qd - centroids[base+c]
			out[c] += diff * diff
		}
	}
}

// PickNextNUnscanned selects the n smallest dists whose indices are NOT
// marked in the scanned bitmap, writing their indices into picked in
// ascending-by-distance order. Used by Phase 27's top-N escalation to
// pick the next N nearest clusters that weren't already scanned by the
// fast tier.
//
// scanned is a bitmap with one bit per cluster (1 = already scanned).
//
// Phase 34 changed the internal data structure from a sorted insertion
// array (O(n) insertion) to a max-heap (O(log n) insertion). The
// motivation: top-N=32 means each insertion under the old scheme could
// shift up to 31 elements; the heap caps it at ⌈log₂(32)⌉=5 swaps.
// The final picked[] is sorted via heapsort-extract at the end.
//
// The bitmap walk runs 64 clusters per scanned-word: in production the
// fast tier marks 1 cluster total, so 63 of the 64 words are all-zero
// and the per-cluster bit test is trivially false. Inlined for the
// Go compiler to recognize.
func PickNextNUnscanned(dists []float32, scanned []uint64, n int, picked []uint16) {
	if n > len(picked) {
		n = len(picked)
	}
	if n == 0 {
		return
	}
	K := len(dists)

	// Heap state. heapDist[i] is the f32 distance currently at heap slot
	// i; heapIdx[i] is the cluster id. The root (slot 0) is the largest
	// distance in the current top-n. Empty heap is signalled by heapSize.
	var heapDist [MaxNProbe]float32
	var heapIdx [MaxNProbe]uint16
	heapSize := 0
	worst := float32(1e38) // = root once heap is full

	// Process 64 clusters per scanned-word.
	for w := 0; w*64 < K; w++ {
		mask := scanned[w]
		base := w * 64
		end := base + 64
		if end > K {
			end = K
		}
		for c := base; c < end; c++ {
			if mask&(1<<uint(c-base)) != 0 {
				continue
			}
			d := dists[c]
			if heapSize == n && d >= worst {
				continue
			}
			if heapSize < n {
				// Insert: place at end, sift up.
				i := heapSize
				heapDist[i] = d
				heapIdx[i] = uint16(c)
				for i > 0 {
					parent := (i - 1) >> 1
					if heapDist[parent] >= heapDist[i] {
						break
					}
					heapDist[parent], heapDist[i] = heapDist[i], heapDist[parent]
					heapIdx[parent], heapIdx[i] = heapIdx[i], heapIdx[parent]
					i = parent
				}
				heapSize++
				if heapSize == n {
					worst = heapDist[0]
				}
			} else {
				// Replace root, sift down.
				heapDist[0] = d
				heapIdx[0] = uint16(c)
				i := 0
				for {
					l := 2*i + 1
					r := 2*i + 2
					largest := i
					if l < n && heapDist[l] > heapDist[largest] {
						largest = l
					}
					if r < n && heapDist[r] > heapDist[largest] {
						largest = r
					}
					if largest == i {
						break
					}
					heapDist[largest], heapDist[i] = heapDist[i], heapDist[largest]
					heapIdx[largest], heapIdx[i] = heapIdx[i], heapIdx[largest]
					i = largest
				}
				worst = heapDist[0]
			}
		}
	}

	// Extract-min iteratively to produce ascending order.
	// We extract from the max-heap by repeatedly popping the root (max),
	// writing it to picked[size-1], picked[size-2], ..., picked[0]. After
	// the loop, picked[0..n-1] is ascending.
	for i := 0; i < n; i++ {
		picked[i] = ^uint16(0)
	}
	for size := heapSize; size > 0; size-- {
		picked[size-1] = heapIdx[0]
		// Move last to root, sift down.
		heapDist[0] = heapDist[size-1]
		heapIdx[0] = heapIdx[size-1]
		i := 0
		end := size - 1
		for {
			l := 2*i + 1
			r := 2*i + 2
			largest := i
			if l < end && heapDist[l] > heapDist[largest] {
				largest = l
			}
			if r < end && heapDist[r] > heapDist[largest] {
				largest = r
			}
			if largest == i {
				break
			}
			heapDist[largest], heapDist[i] = heapDist[i], heapDist[largest]
			heapIdx[largest], heapIdx[i] = heapIdx[i], heapIdx[largest]
			i = largest
		}
	}
}

// pickArgminScalar finds the index of the smallest float32 in dists.
// Lowest-index wins on ties. Used as the argmin oracle by the parity
// test and as the non-amd64 fallback for pickArgminFast.
func pickArgminScalar(dists []float32) uint16 {
	bestI := uint16(0)
	bestD := dists[0]
	for i := 1; i < len(dists); i++ {
		if dists[i] < bestD {
			bestD = dists[i]
			bestI = uint16(i)
		}
	}
	return bestI
}

// PickTopNCentroids selects the n smallest dists and writes their indices
// into picked. dists has length K; n must be <= len(picked). The picked
// slice is filled in ascending-by-distance order.
//
// Production calls this with n == FastNProbe == 1 on every query; that
// path goes through pickArgminFast (asm on amd64). For n > 1, we use
// the linear-scan + sorted-insert path below — O(K*n), trivial at
// K=4096, n≤8.
func PickTopNCentroids(dists []float32, n int, picked []uint16) {
	if n == 1 && len(picked) >= 1 {
		picked[0] = pickArgminFast(dists)
		return
	}
	if n > len(picked) {
		n = len(picked)
	}
	// Sentinel: float32 max + 1 cluster id past K.
	for i := 0; i < n; i++ {
		picked[i] = ^uint16(0)
	}
	worst := float32(1e38)
	worstIdx := 0

	for c := 0; c < len(dists); c++ {
		d := dists[c]
		if d >= worst {
			continue
		}
		// Find insertion position in the sorted-ascending picked array.
		var pos int
		for pos = 0; pos < n; pos++ {
			id := picked[pos]
			if id == ^uint16(0) || dists[id] > d {
				break
			}
		}
		if pos == n {
			continue
		}
		// Shift right from pos..n-2, then write.
		for i := n - 1; i > pos; i-- {
			picked[i] = picked[i-1]
		}
		picked[pos] = uint16(c)

		// Refresh worst.
		worstID := picked[n-1]
		if worstID == ^uint16(0) {
			worst = 1e38
		} else {
			worst = dists[worstID]
			worstIdx = int(worstID)
			_ = worstIdx
		}
	}
}
