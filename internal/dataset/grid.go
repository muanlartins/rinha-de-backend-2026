package dataset

import "slices"

type Partition struct {
	NumCells   int
	CellStarts []uint32
	CellCounts []uint32

	// Flattened per-cell AABBs over all 14 dims: BboxMin[cell*Dims + d].
	BboxMin []int16
	BboxMax []int16

	GridDims   []uint8
	BinsPerDim []uint16
	Boundaries []int16
}

// selectGridDims: for sentinel partitions (dim 6 pinned to SentinelInt),
// substitute dim 7 since dim 6 carries no information.
func selectGridDims(key uint8) ([]uint8, []uint16) {
	if key&0x10 != 0 {
		return []uint8{0, 12, 7}, []uint16{16, 8, 8}
	}
	return []uint8{0, 12, 6}, []uint16{16, 8, 8}
}

// BuildGrid must run after Partition().
func (ds *Dataset) BuildGrid() {
	ds.Partitions = make([]*Partition, NumPartitions)
	for k := uint8(0); k < NumPartitions; k++ {
		if ds.PartitionCounts[k] == 0 {
			continue
		}
		ds.Partitions[k] = ds.buildPartitionGrid(k)
	}
}

func (ds *Dataset) buildPartitionGrid(key uint8) *Partition {
	pStart := ds.PartitionStarts[key]
	pCount := ds.PartitionCounts[key]

	gridDims, binsPerDim := selectGridDims(key)

	var totalBounds int
	for _, b := range binsPerDim {
		totalBounds += int(b) - 1
	}
	boundaries := make([]int16, totalBounds)

	tmp := make([]int16, pCount)
	bOff := 0
	for gi, dim := range gridDims {
		bins := int(binsPerDim[gi])
		numBounds := bins - 1
		for j := uint32(0); j < pCount; j++ {
			tmp[j] = ds.Vectors[(pStart+j)*Dims+uint32(dim)]
		}
		slices.Sort(tmp)
		for b := 0; b < numBounds; b++ {
			pctIdx := int(float64(b+1) / float64(bins) * float64(pCount-1))
			boundaries[bOff+b] = tmp[pctIdx]
		}
		bOff += numBounds
	}

	cellKeys := make([]uint32, pCount)
	for j := uint32(0); j < pCount; j++ {
		base := (pStart + j) * Dims
		var ck uint32
		bOff := 0
		for gi, dim := range gridDims {
			bins := int(binsPerDim[gi])
			numBounds := bins - 1
			v := ds.Vectors[base+uint32(dim)]
			bin := 0
			for b := 0; b < numBounds; b++ {
				if v > boundaries[bOff+b] {
					bin = b + 1
				} else {
					break
				}
			}
			ck = ck*uint32(bins) + uint32(bin)
			bOff += numBounds
		}
		cellKeys[j] = ck
	}

	counts := map[uint32]uint32{}
	for _, ck := range cellKeys {
		counts[ck]++
	}
	uniqueKeys := make([]uint32, 0, len(counts))
	for ck := range counts {
		uniqueKeys = append(uniqueKeys, ck)
	}
	slices.Sort(uniqueKeys)

	cellID := make(map[uint32]int, len(uniqueKeys))
	cellStartsLocal := make([]uint32, len(uniqueKeys))
	cellCountsLocal := make([]uint32, len(uniqueKeys))
	for ci, ck := range uniqueKeys {
		cellID[ck] = ci
		cellCountsLocal[ci] = counts[ck]
	}
	for ci := 1; ci < len(uniqueKeys); ci++ {
		cellStartsLocal[ci] = cellStartsLocal[ci-1] + cellCountsLocal[ci-1]
	}

	// Inverse / source map for the cycle walker. See docs/CODE_NOTES.md.
	src := make([]uint32, pCount)
	cursors := slices.Clone(cellStartsLocal)
	for j := uint32(0); j < pCount; j++ {
		c := cellID[cellKeys[j]]
		src[cursors[c]] = j
		cursors[c]++
	}

	permuteInPartition(ds, pStart, pCount, src, cellKeys)

	numCells := len(uniqueKeys)
	bboxMin := make([]int16, numCells*Dims)
	bboxMax := make([]int16, numCells*Dims)
	for ci := 0; ci < numCells; ci++ {
		mi := ci * Dims
		for d := 0; d < Dims; d++ {
			bboxMin[mi+d] = 32767
			bboxMax[mi+d] = -32768
		}
		start := cellStartsLocal[ci]
		count := cellCountsLocal[ci]
		for k := uint32(0); k < count; k++ {
			vBase := (pStart + start + k) * Dims
			for d := 0; d < Dims; d++ {
				v := ds.Vectors[vBase+uint32(d)]
				if v < bboxMin[mi+d] {
					bboxMin[mi+d] = v
				}
				if v > bboxMax[mi+d] {
					bboxMax[mi+d] = v
				}
			}
		}
	}

	absStarts := make([]uint32, numCells)
	for ci := 0; ci < numCells; ci++ {
		absStarts[ci] = pStart + cellStartsLocal[ci]
	}

	return &Partition{
		NumCells:   numCells,
		CellStarts: absStarts,
		CellCounts: cellCountsLocal,
		BboxMin:    bboxMin,
		BboxMax:    bboxMax,
		GridDims:   gridDims,
		BinsPerDim: binsPerDim,
		Boundaries: boundaries,
	}
}

func permuteInPartition(ds *Dataset, pStart, pCount uint32, src []uint32, cellKeys []uint32) {
	visited := make([]bool, pCount)
	var buf [Dims]int16

	for i := uint32(0); i < pCount; i++ {
		if visited[i] || src[i] == i {
			visited[i] = true
			continue
		}

		copy(buf[:], ds.Vectors[(pStart+i)*Dims:(pStart+i+1)*Dims])
		labelBuf := ds.Labels[pStart+i]
		keyBuf := cellKeys[i]

		j := i
		for {
			visited[j] = true
			next := src[j]
			if next == i {
				copy(ds.Vectors[(pStart+j)*Dims:(pStart+j+1)*Dims], buf[:])
				ds.Labels[pStart+j] = labelBuf
				cellKeys[j] = keyBuf
				break
			}
			copy(ds.Vectors[(pStart+j)*Dims:(pStart+j+1)*Dims], ds.Vectors[(pStart+next)*Dims:(pStart+next+1)*Dims])
			ds.Labels[pStart+j] = ds.Labels[pStart+next]
			cellKeys[j] = cellKeys[next]
			j = next
		}
	}
}
