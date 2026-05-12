package dataset

import (
	"math/rand/v2"
	"slices"
)

const (
	// IVFClustersPerPartition: number of k-means clusters built within each
	// of the 32 partitions. With 3M vectors across 32 partitions, each
	// partition has ~94k vectors → ~735 per cluster at K=128.
	IVFClustersPerPartition = 128
	// IVFKmeansIters: Lloyd iterations for k-means. Few enough to keep
	// image build time under a couple minutes; many enough for usable
	// clusters (random init + 5 passes converges to ~95% of full converge).
	IVFKmeansIters = 5
)

type Cluster struct {
	Centroid [Stride]int16 // dim 14, 15 always 0
	Start    uint32        // absolute index in ds.Vectors (after the in-place reorder)
	Count    uint32
	// Per-cluster AABB across all dims, for query-side LB pruning.
	BboxMin [Stride]int16
	BboxMax [Stride]int16
}

type IVFPartition struct {
	Clusters []Cluster
}

// BuildIVF replaces the cell grid with k-means clusters within each
// partition. Must run after Partition(). The vectors slab is reordered
// in place so each cluster's vectors sit contiguously.
func (ds *Dataset) BuildIVF() {
	ds.IVF = make([]*IVFPartition, NumPartitions)
	for k := uint8(0); k < NumPartitions; k++ {
		if ds.PartitionCounts[k] == 0 {
			continue
		}
		ds.IVF[k] = ds.buildPartitionIVF(k)
	}
}

func (ds *Dataset) buildPartitionIVF(key uint8) *IVFPartition {
	pStart := ds.PartitionStarts[key]
	pCount := ds.PartitionCounts[key]

	K := IVFClustersPerPartition
	if int(pCount) < K {
		K = int(pCount)
	}

	rng := rand.New(rand.NewPCG(uint64(key)*0x9e3779b97f4a7c15, 0xa5a5a5a5a5a5a5a5))

	centroids := make([][Stride]int16, K)
	picked := make(map[uint32]struct{}, K)
	for c := 0; c < K; c++ {
		for {
			r := uint32(rng.IntN(int(pCount)))
			if _, dup := picked[r]; dup {
				continue
			}
			picked[r] = struct{}{}
			base := (pStart + r) * Stride
			copy(centroids[c][:], ds.Vectors[base:base+Stride])
			break
		}
	}

	assign := make([]uint16, pCount)
	for iter := 0; iter < IVFKmeansIters; iter++ {
		for i := uint32(0); i < pCount; i++ {
			base := (pStart + i) * Stride
			var bestD int64 = (1 << 62)
			var bestC uint16
			for c := 0; c < K; c++ {
				d := sqdistI16(ds.Vectors[base:base+Stride], centroids[c][:])
				if d < bestD {
					bestD = d
					bestC = uint16(c)
				}
			}
			assign[i] = bestC
		}

		// Recompute centroids as mean of assigned vectors.
		var sums [][Stride]int64
		sums = make([][Stride]int64, K)
		counts := make([]uint32, K)
		for i := uint32(0); i < pCount; i++ {
			c := assign[i]
			counts[c]++
			base := (pStart + i) * Stride
			for d := 0; d < Stride; d++ {
				sums[c][d] += int64(ds.Vectors[base+uint32(d)])
			}
		}
		for c := 0; c < K; c++ {
			if counts[c] == 0 {
				// Reseed an empty cluster from a random vector.
				r := uint32(rng.IntN(int(pCount)))
				base := (pStart + r) * Stride
				copy(centroids[c][:], ds.Vectors[base:base+Stride])
				continue
			}
			for d := 0; d < Stride; d++ {
				centroids[c][d] = int16(sums[c][d] / int64(counts[c]))
			}
		}
	}

	// Compute cluster sizes (final assignment), then reorder vectors so
	// each cluster's members sit contiguously inside the partition.
	counts := make([]uint32, K)
	for _, c := range assign {
		counts[c]++
	}
	startsLocal := make([]uint32, K)
	for c := 1; c < K; c++ {
		startsLocal[c] = startsLocal[c-1] + counts[c-1]
	}
	// Inverse / source map for the cycle walker (same convention as the grid).
	src := make([]uint32, pCount)
	cursors := slices.Clone(startsLocal)
	for i := uint32(0); i < pCount; i++ {
		c := assign[i]
		src[cursors[c]] = i
		cursors[c]++
	}
	permuteInPartition(ds, pStart, pCount, src, nil)

	// Build per-cluster AABBs over all dims.
	clusters := make([]Cluster, K)
	for c := 0; c < K; c++ {
		clusters[c].Centroid = centroids[c]
		clusters[c].Start = pStart + startsLocal[c]
		clusters[c].Count = counts[c]
		for d := 0; d < Stride; d++ {
			clusters[c].BboxMin[d] = 32767
			clusters[c].BboxMax[d] = -32768
		}
		start := clusters[c].Start
		cnt := clusters[c].Count
		for k := uint32(0); k < cnt; k++ {
			vBase := (start + k) * Stride
			for d := 0; d < Stride; d++ {
				v := ds.Vectors[vBase+uint32(d)]
				if v < clusters[c].BboxMin[d] {
					clusters[c].BboxMin[d] = v
				}
				if v > clusters[c].BboxMax[d] {
					clusters[c].BboxMax[d] = v
				}
			}
		}
	}

	return &IVFPartition{Clusters: clusters}
}

// sqdistI16 — scalar squared Euclidean between two Stride-int16 vectors.
// Used at build time only; runtime uses the SIMD path.
func sqdistI16(a, b []int16) int64 {
	var sum int64
	for d := 0; d < Stride; d++ {
		t := int64(a[d]) - int64(b[d])
		sum += t * t
	}
	return sum
}
