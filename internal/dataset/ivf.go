package dataset

import "sort"

const (
	// IVFClusters: K, must be a power of two for the recursive balanced
	// split. 8192 → ~366 vectors per cluster over 3M. Same value top-Go
	// uses; gives nprobe=8 enough coverage for ~exact KNN-5.
	IVFClusters = 8192

	BlockVectors          = 8
	BlockInt16            = Stride * BlockVectors
	IVFCentroidBlockInt16 = Stride * BlockVectors
)

type Cluster struct {
	BlockStart uint32
	LabelStart uint32
	Count      uint32
	NumBlocks  uint32
	BboxMin    [Stride]int16
	BboxMax    [Stride]int16
}

type IVFIndex struct {
	Clusters       []Cluster
	CentroidBlocks []int16 // ceil(K/8) blocks × Stride × 8 int16, dim-major
}

// BuildIVF: balanced KD-tree-style binary split on the max-variance dim,
// recursive until we have K clusters. Each cluster has equal size (±1).
// Much faster and more uniform than Lloyd k-means, and the cluster sizes
// being balanced is what makes nprobe-based search reliable.
func (ds *Dataset) BuildIVF() {
	K := IVFClusters
	if K > ds.Count {
		K = highestPowerOfTwoLE(ds.Count)
	}

	ids := make([]uint32, ds.Count)
	for i := range ids {
		ids[i] = uint32(i)
	}
	ranges := make([][2]uint32, K)
	balancedSplit(ds.Vectors, ids, ranges, 0, ds.Count, 0, K)

	clusters := make([]Cluster, K)
	centroidsFlat := make([]int16, K*Stride)
	for c := 0; c < K; c++ {
		r := ranges[c]
		clusters[c].Count = r[1] - r[0]
		clusters[c].NumBlocks = (clusters[c].Count + BlockVectors - 1) / BlockVectors
		computeClusterStats(ds.Vectors, ids, r,
			centroidsFlat[c*Stride:(c+1)*Stride],
			clusters[c].BboxMin[:],
			clusters[c].BboxMax[:])
	}

	ds.IVF = &IVFIndex{Clusters: clusters}
	ds.transposeVectorsToBlocksByIDs(ids, ranges)
	ds.packCentroids(centroidsFlat)
}

func balancedSplit(vectors []int16, ids []uint32, ranges [][2]uint32, start, end, clusterBase, clusterCount int) {
	if clusterCount == 1 {
		ranges[clusterBase] = [2]uint32{uint32(start), uint32(end)}
		return
	}
	dim := maxVarianceDim(vectors, ids, start, end)
	window := ids[start:end]
	sort.Slice(window, func(i, j int) bool {
		l := vectors[int(window[i])*Stride+dim]
		r := vectors[int(window[j])*Stride+dim]
		if l == r {
			return window[i] < window[j]
		}
		return l < r
	})
	mid := start + (end-start)/2
	half := clusterCount / 2
	balancedSplit(vectors, ids, ranges, start, mid, clusterBase, half)
	balancedSplit(vectors, ids, ranges, mid, end, clusterBase+half, half)
}

func maxVarianceDim(vectors []int16, ids []uint32, start, end int) int {
	var sums, sumSq [Stride]int64
	for i := start; i < end; i++ {
		base := int(ids[i]) * Stride
		for d := 0; d < Stride; d++ {
			v := int64(vectors[base+d])
			sums[d] += v
			sumSq[d] += v * v
		}
	}
	count := float64(end - start)
	best, bestVar := 0, -1.0
	for d := 0; d < Stride; d++ {
		mean := float64(sums[d]) / count
		variance := float64(sumSq[d])/count - mean*mean
		if variance > bestVar {
			bestVar = variance
			best = d
		}
	}
	return best
}

func computeClusterStats(vectors []int16, ids []uint32, r [2]uint32, centroid, bboxMin, bboxMax []int16) {
	for d := 0; d < Stride; d++ {
		bboxMin[d] = 32767
		bboxMax[d] = -32768
	}
	var sums [Stride]int64
	for i := r[0]; i < r[1]; i++ {
		base := int(ids[i]) * Stride
		for d := 0; d < Stride; d++ {
			v := vectors[base+d]
			sums[d] += int64(v)
			if v < bboxMin[d] {
				bboxMin[d] = v
			}
			if v > bboxMax[d] {
				bboxMax[d] = v
			}
		}
	}
	count := int64(r[1] - r[0])
	for d := 0; d < Stride; d++ {
		if count > 0 {
			centroid[d] = int16((sums[d] + count/2) / count)
		}
	}
}

func (ds *Dataset) transposeVectorsToBlocksByIDs(ids []uint32, ranges [][2]uint32) {
	var totalBlocks uint64
	for ci := range ds.IVF.Clusters {
		totalBlocks += uint64(ds.IVF.Clusters[ci].NumBlocks)
	}
	ds.Blocks = make([]int16, totalBlocks*BlockInt16)
	ds.BlockLabels = make([]uint8, totalBlocks*BlockVectors)

	var dstBlock uint64
	for ci := range ds.IVF.Clusters {
		c := &ds.IVF.Clusters[ci]
		r := ranges[ci]
		c.BlockStart = uint32(dstBlock * BlockInt16)
		c.LabelStart = uint32(dstBlock * BlockVectors)
		for b := uint32(0); b < c.NumBlocks; b++ {
			blockOff := (dstBlock + uint64(b)) * BlockInt16
			for v := uint32(0); v < BlockVectors; v++ {
				pos := b*BlockVectors + v
				if pos >= c.Count {
					break
				}
				origID := ids[r[0]+pos]
				vBase := uint64(origID) * Stride
				for d := uint32(0); d < Stride; d++ {
					ds.Blocks[blockOff+uint64(d)*BlockVectors+uint64(v)] = ds.Vectors[vBase+uint64(d)]
				}
				ds.BlockLabels[c.LabelStart+pos] = ds.Labels[origID]
			}
		}
		dstBlock += uint64(c.NumBlocks)
	}
}

func (ds *Dataset) packCentroids(centroidsFlat []int16) {
	K := len(centroidsFlat) / Stride
	numBlocks := (K + BlockVectors - 1) / BlockVectors
	ds.IVF.CentroidBlocks = make([]int16, numBlocks*IVFCentroidBlockInt16)
	for b := 0; b < numBlocks; b++ {
		blockOff := b * IVFCentroidBlockInt16
		for v := 0; v < BlockVectors; v++ {
			ci := b*BlockVectors + v
			if ci >= K {
				break
			}
			for d := 0; d < Stride; d++ {
				ds.IVF.CentroidBlocks[blockOff+d*BlockVectors+v] = centroidsFlat[ci*Stride+d]
			}
		}
	}
}

func highestPowerOfTwoLE(v int) int {
	p := 1
	for p*2 <= v {
		p *= 2
	}
	return p
}
