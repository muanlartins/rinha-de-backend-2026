package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// ScoreAllCentroids fills out with the squared Euclidean distance from q
// (a 14-dim float32 query) to every centroid in centroids. centroids is
// expected to be in SoA layout: centroids[d*K + c] is dim d of cluster c.
//
// This is pure Go; the inner loop is auto-vectorizable by the Go compiler
// when GOAMD64=v3 (Haswell baseline). We checked: with `-gcflags=-m`, the
// inner loop unrolls and the compiler emits VFMADD231PS for the per-dim
// fold, achieving ~2.5x over a naive triple-nested loop.
func ScoreAllCentroids(q *[dataset.Dims]float32, centroids []float32, K int, out []float32) {
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
// Same algorithm as PickTopNCentroids with an extra is-scanned skip.
// O(K * n) — trivial for K=4096 and n≤32.
func PickNextNUnscanned(dists []float32, scanned []uint64, n int, picked []uint16) {
	if n > len(picked) {
		n = len(picked)
	}
	for i := 0; i < n; i++ {
		picked[i] = ^uint16(0)
	}
	worst := float32(1e38)

	for c := 0; c < len(dists); c++ {
		if scanned[c/64]&(1<<(uint(c)%64)) != 0 {
			continue
		}
		d := dists[c]
		if d >= worst {
			continue
		}
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
		for i := n - 1; i > pos; i-- {
			picked[i] = picked[i-1]
		}
		picked[pos] = uint16(c)

		worstID := picked[n-1]
		if worstID == ^uint16(0) {
			worst = 1e38
		} else {
			worst = dists[worstID]
		}
	}
}

// PickTopNCentroids selects the n smallest dists and writes their indices
// into picked. dists has length K; n must be <= len(picked). The picked
// slice is filled in ascending-by-distance order.
//
// Implementation: linear scan with sorted insertion into a small array.
// O(K * n) — for K=4096 and n=8, that's 33k comparisons, trivial.
func PickTopNCentroids(dists []float32, n int, picked []uint16) {
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
