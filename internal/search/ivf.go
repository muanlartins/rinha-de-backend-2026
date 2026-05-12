package search

import (
	"slices"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// IVFNProbe: number of clusters scanned per query, sorted by centroid
// distance ascending. With K=128 clusters per partition and ~735 vectors
// per cluster, nprobe=8 scans ~5,900 vectors per query — comparable to a
// well-pruned grid sweep, but with no per-cell LB recompute.
var IVFNProbe = 8

// FraudCountIVF: approximate KNN-5 via IVF. Compute distance to all K
// centroids, sort, scan the IVFNProbe nearest clusters with the SIMD
// kernel, return the fraud count among the 5 nearest.
//
// Accuracy: ~99.9% of queries match exact brute-force in measurement
// (a tiny number of borderline cases miss when the true 5th-nearest sits in
// a cluster ranked beyond nprobe by centroid distance).
func FraudCountIVF(query *[stride]int16, ds *dataset.Dataset) int {
	key := dataset.ComputeKey(query)
	pg := ds.IVF[key]
	if pg == nil || len(pg.Clusters) == 0 {
		return 0
	}

	numClusters := len(pg.Clusters)
	s := getScratch(numClusters)
	defer putScratch(s)

	centroidDists := s.cellLBs[:numClusters]
	qPtr := &query[0]
	for ci := 0; ci < numClusters; ci++ {
		c := &pg.Clusters[ci]
		if useAVX2 {
			centroidDists[ci] = sqdistAVX2(qPtr, &c.Centroid[0])
		} else {
			centroidDists[ci] = sqdistFallback(query, &c.Centroid)
		}
	}

	sorted := s.sortedCells[:numClusters]
	for i := range sorted {
		sorted[i] = i
	}
	slices.SortFunc(sorted, func(a, b int) int {
		switch {
		case centroidDists[a] < centroidDists[b]:
			return -1
		case centroidDists[a] > centroidDists[b]:
			return 1
		default:
			return 0
		}
	})

	nprobe := IVFNProbe
	if nprobe > numClusters {
		nprobe = numClusters
	}

	tk := newTopK()
	if useAVX2 {
		scanProbedAVX2(query, ds.Vectors, sorted[:nprobe], pg, &tk)
	} else {
		isSentinel5 := (key & 0x08) != 0
		isSentinel6 := (key & 0x10) != 0
		scanProbedScalar(query, ds.Vectors, sorted[:nprobe], pg, isSentinel5, isSentinel6, &tk)
	}

	frauds := 0
	labels := ds.Labels
	if tk.i0 >= 0 && labels[tk.i0] == 1 {
		frauds++
	}
	if tk.i1 >= 0 && labels[tk.i1] == 1 {
		frauds++
	}
	if tk.i2 >= 0 && labels[tk.i2] == 1 {
		frauds++
	}
	if tk.i3 >= 0 && labels[tk.i3] == 1 {
		frauds++
	}
	if tk.i4 >= 0 && labels[tk.i4] == 1 {
		frauds++
	}
	return frauds
}

func scanProbedAVX2(query *[stride]int16, vectors []int16, probed []int, pg *dataset.IVFPartition, tk *topK) {
	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	i0, i1, i2, i3, i4 := tk.i0, tk.i1, tk.i2, tk.i3, tk.i4
	qPtr := &query[0]

	for _, ci := range probed {
		c := &pg.Clusters[ci]
		start := int(c.Start)
		end := start + int(c.Count)
		for i := start; i < end; i++ {
			dist := sqdistAVX2(qPtr, &vectors[i*stride])
			if dist >= d4 {
				continue
			}
			switch {
			case dist < d0:
				d4, i4 = d3, i3
				d3, i3 = d2, i2
				d2, i2 = d1, i1
				d1, i1 = d0, i0
				d0, i0 = dist, i
			case dist < d1:
				d4, i4 = d3, i3
				d3, i3 = d2, i2
				d2, i2 = d1, i1
				d1, i1 = dist, i
			case dist < d2:
				d4, i4 = d3, i3
				d3, i3 = d2, i2
				d2, i2 = dist, i
			case dist < d3:
				d4, i4 = d3, i3
				d3, i3 = dist, i
			default:
				d4, i4 = dist, i
			}
		}
	}

	tk.d0, tk.d1, tk.d2, tk.d3, tk.d4 = d0, d1, d2, d3, d4
	tk.i0, tk.i1, tk.i2, tk.i3, tk.i4 = i0, i1, i2, i3, i4
}

func scanProbedScalar(query *[stride]int16, vectors []int16, probed []int, pg *dataset.IVFPartition, isSentinel5, isSentinel6 bool, tk *topK) {
	q0 := int32(query[0])
	q1 := int32(query[1])
	q2 := int32(query[2])
	q3 := int32(query[3])
	q4 := int32(query[4])
	q5 := int32(query[5])
	q6 := int32(query[6])
	q7 := int32(query[7])
	q8 := int32(query[8])
	q12 := int32(query[12])
	q13 := int32(query[13])

	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	i0, i1, i2, i3, i4 := tk.i0, tk.i1, tk.i2, tk.i3, tk.i4

	for _, ci := range probed {
		c := &pg.Clusters[ci]
		start := int(c.Start)
		end := start + int(c.Count)
		for i := start; i < end; i++ {
			base := i * stride
			t := q0 - int32(vectors[base])
			dist := int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			t = q1 - int32(vectors[base+1])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			t = q2 - int32(vectors[base+2])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			t = q3 - int32(vectors[base+3])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			t = q4 - int32(vectors[base+4])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			if !isSentinel5 {
				t = q5 - int32(vectors[base+5])
				dist += int64(t) * int64(t)
				if dist >= d4 {
					continue
				}
			}
			if !isSentinel6 {
				t = q6 - int32(vectors[base+6])
				dist += int64(t) * int64(t)
				if dist >= d4 {
					continue
				}
			}
			t = q7 - int32(vectors[base+7])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			t = q8 - int32(vectors[base+8])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			t = q12 - int32(vectors[base+12])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			t = q13 - int32(vectors[base+13])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
			switch {
			case dist < d0:
				d4, i4 = d3, i3
				d3, i3 = d2, i2
				d2, i2 = d1, i1
				d1, i1 = d0, i0
				d0, i0 = dist, i
			case dist < d1:
				d4, i4 = d3, i3
				d3, i3 = d2, i2
				d2, i2 = d1, i1
				d1, i1 = dist, i
			case dist < d2:
				d4, i4 = d3, i3
				d3, i3 = d2, i2
				d2, i2 = dist, i
			case dist < d3:
				d4, i4 = d3, i3
				d3, i3 = dist, i
			default:
				d4, i4 = dist, i
			}
		}
	}

	tk.d0, tk.d1, tk.d2, tk.d3, tk.d4 = d0, d1, d2, d3, d4
	tk.i0, tk.i1, tk.i2, tk.i3, tk.i4 = i0, i1, i2, i3, i4
}

// sqdistFallback is the non-AVX2 squared distance over a stride-int16 pair.
// Used only on non-amd64 builds for centroid distance.
func sqdistFallback(query, c *[stride]int16) int64 {
	var d int64
	for i := 0; i < stride; i++ {
		t := int64(query[i]) - int64(c[i])
		d += t * t
	}
	return d
}
