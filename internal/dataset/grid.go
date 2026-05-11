package dataset

import (
	"slices"
)

// Partition is a per-partition grid index built on top of the int16 vector
// slab. After BuildGrid(), references inside a partition are reordered so
// that all vectors belonging to the same cell sit consecutive.
type Partition struct {
	NumCells   int
	CellStarts []uint32 // index into ds.Vectors (absolute, not partition-relative)
	CellCounts []uint32

	// Axis-aligned bounding boxes, flattened: BboxMin[cell*Dims + d].
	// Tracked for all 14 dims (including the partition-constant ones) so the
	// search code's LB sweep can use the same indexing convention.
	BboxMin []int16
	BboxMax []int16

	// Which dims are used to compute the cell key. Length 1..3 in practice.
	GridDims   []uint8
	BinsPerDim []uint16
	// Boundaries are packed: for dim i with bins b, there are b-1 boundaries
	// laid out consecutively. Sum-of-(b-1) = len(Boundaries).
	Boundaries []int16
}

// Grid dimension selection per partition:
//
//   - dim 0 (amount):     16 bins, percentile-based
//   - dim 12 (mcc_risk):  8 bins
//   - dim 6 (km_from_last_tx)  for non-sentinel partitions (bit 4 unset): 8 bins
//   - dim 7 (km_from_home) for sentinel partitions (bit 4 set): 8 bins
//
// This matches QRust's tuning. The choice of grid dims matters less than the
// fact that we *partition* on them.
func selectGridDims(key uint8) ([]uint8, []uint16) {
	if key&0x10 != 0 {
		// dim 6 is pinned to SentinelInt for this partition; substitute dim 7.
		return []uint8{0, 12, 7}, []uint16{16, 8, 8}
	}
	return []uint8{0, 12, 6}, []uint16{16, 8, 8}
}

// BuildGrid builds a per-partition grid for every non-empty partition. Must
// be called after Partition().
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

	// === 1. compute percentile boundaries per grid dim
	var totalBounds int
	for _, b := range binsPerDim {
		totalBounds += int(b) - 1
	}
	boundaries := make([]int16, totalBounds)

	tmp := make([]int16, pCount) // scratch for sorting one dim's values
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

	// === 2. assign each vector to a cell
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

	// === 3. reorder partition's vectors so same-cell are contiguous
	//
	// We compute destination indices, then permute in-place via cycle
	// decomposition (the same trick used in Partition()).
	counts := map[uint32]uint32{}
	for _, ck := range cellKeys {
		counts[ck]++
	}
	// Sort cell keys to give cells deterministic ids.
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

	// Build the source map (inverse of the forward destination map): src[p]
	// is the partition-relative index of the vector that should end up at
	// partition-relative position p. The permutation routine walks cycles of
	// this src map.
	src := make([]uint32, pCount)
	cursors := slices.Clone(cellStartsLocal)
	for j := uint32(0); j < pCount; j++ {
		c := cellID[cellKeys[j]]
		src[cursors[c]] = j
		cursors[c]++
	}

	permuteInPartition(ds, pStart, pCount, src, cellKeys)

	// === 4. compute per-cell AABBs over all 14 dims
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

	// Cell starts in absolute (whole-dataset) coordinates make the search
	// loop's indexing trivial.
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

// permuteInPartition rotates vectors+labels in [pStart, pStart+pCount) so
// that the vector currently at partition-relative position src[p] ends up at
// position p. cellKeys is shuffled along so the per-cell metadata can be
// computed from the now-sorted partition.
func permuteInPartition(ds *Dataset, pStart, pCount uint32, src []uint32, cellKeys []uint32) {
	visited := make([]bool, pCount)
	var buf [Dims]int16

	for i := uint32(0); i < pCount; i++ {
		if visited[i] || src[i] == i {
			visited[i] = true
			continue
		}

		// Capture the value currently at position i; we'll fill it last after
		// walking the rest of the cycle.
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
