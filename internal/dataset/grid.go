package dataset

import (
	"sort"
)

// BuildGrid is the build-time pipeline that runs after LoadFromGzipJSON. It
// reorders ds.Vectors / ds.Labels in two passes — first by partition key,
// then within each partition by cell key — so that on read both the
// partition slab and the cell slab are contiguous and the search can walk
// them with a single increasing index. AABBs and per-partition bin
// boundaries are computed once here and serialized into the index file by
// SaveIndex.
func (ds *Dataset) BuildGrid() {
	ds.partitionInPlace()
	for p := uint8(0); p < NumPartitions; p++ {
		ds.buildPartitionGrid(p)
	}
}

// partitionInPlace permutes Vectors and Labels so that records with the
// same partition key are contiguous. Uses the in-place swap algorithm
// (see swapPermuteRows comment) — O(N) total swaps with one row-sized
// scratch buffer.
func (ds *Dataset) partitionInPlace() {
	keys := make([]uint8, ds.Count)
	for i := 0; i < ds.Count; i++ {
		keys[i] = PartitionKey(ds.Vectors[i*Stride : i*Stride+Stride])
	}

	var counts [NumPartitions]uint32
	for i := 0; i < ds.Count; i++ {
		counts[keys[i]]++
	}
	var starts [NumPartitions]uint32
	for p := 1; p < NumPartitions; p++ {
		starts[p] = starts[p-1] + counts[p-1]
	}

	// target[i] = absolute index where row i should land.
	cursor := starts
	target := make([]uint32, ds.Count)
	for i := 0; i < ds.Count; i++ {
		p := keys[i]
		target[i] = cursor[p]
		cursor[p]++
	}

	swapPermuteRows(ds.Vectors, ds.Labels, target, 0, uint32(ds.Count))

	ds.PartitionStarts = starts
	ds.PartitionCounts = counts
}

// buildPartitionGrid computes percentile bin boundaries on the partition's
// vectors, assigns cell keys, reorders within partition so cells are
// contiguous, then records cellStarts / Counts / Bboxes.
func (ds *Dataset) buildPartitionGrid(p uint8) {
	start := ds.PartitionStarts[p]
	count := ds.PartitionCounts[p]
	if count == 0 {
		return
	}

	// === Step 1: percentile boundaries for each of the 3 grid dims.
	bounds := computeBoundaries(ds.Vectors, start, count)

	// === Step 2: assign cell key per vector in [0, BinsTotal).
	cellKeys := make([]uint16, count)
	for i := uint32(0); i < count; i++ {
		base := (start + i) * Stride
		v0 := ds.Vectors[base+GridDim0]
		v1 := ds.Vectors[base+GridDim1]
		v2 := ds.Vectors[base+GridDim2]
		b0 := bin(v0, bounds[0:BinsD0-1])
		b1 := bin(v1, bounds[BinsD0-1:BinsD0-1+BinsD1-1])
		b2 := bin(v2, bounds[BinsD0-1+BinsD1-1:])
		cellKeys[i] = uint16(b0)*BinsD1*BinsD2 + uint16(b1)*BinsD2 + uint16(b2)
	}

	// === Step 3: bucket-count + prefix sum → target indices.
	var bucketCounts [BinsTotal]uint32
	for _, k := range cellKeys {
		bucketCounts[k]++
	}
	var bucketStarts [BinsTotal]uint32
	for c := 1; c < BinsTotal; c++ {
		bucketStarts[c] = bucketStarts[c-1] + bucketCounts[c-1]
	}
	cursor := bucketStarts
	target := make([]uint32, count)
	for i := uint32(0); i < count; i++ {
		k := cellKeys[i]
		target[i] = start + cursor[k]
		cursor[k]++
	}

	swapPermuteRows(ds.Vectors, ds.Labels, target, start, count)

	// === Step 4: record cell metadata for non-empty cells.
	cells := make([]Cell, 0, 256)
	for c := 0; c < BinsTotal; c++ {
		cnt := bucketCounts[c]
		if cnt == 0 {
			continue
		}
		offsetInPartition := bucketStarts[c]
		var cell Cell
		cell.Start = offsetInPartition
		cell.Count = cnt
		for d := 0; d < Dims; d++ {
			cell.BboxMn[d] = 32767
			cell.BboxMx[d] = -32768
		}
		for i := uint32(0); i < cnt; i++ {
			base := (start + offsetInPartition + i) * Stride
			for d := 0; d < Dims; d++ {
				v := ds.Vectors[base+uint32(d)]
				if v < cell.BboxMn[d] {
					cell.BboxMn[d] = v
				}
				if v > cell.BboxMx[d] {
					cell.BboxMx[d] = v
				}
			}
		}
		cells = append(cells, cell)
	}

	ds.Grids[p] = PartitionGrid{
		Cells:    cells,
		Bounds:   bounds,
		NumCells: uint32(len(cells)),
	}
}

// computeBoundaries returns the percentile cuts for the 3 grid dims.
func computeBoundaries(vectors []int16, start, count uint32) [BoundsCount]int16 {
	var out [BoundsCount]int16
	if count == 0 {
		return out
	}

	dims := [...]int{GridDim0, GridDim1, GridDim2}
	bins := [...]int{BinsD0, BinsD1, BinsD2}

	scratch := make([]int16, count)
	wOff := 0
	for di, d := range dims {
		for i := uint32(0); i < count; i++ {
			scratch[i] = vectors[(start+i)*Stride+uint32(d)]
		}
		sort.Slice(scratch, func(i, j int) bool { return scratch[i] < scratch[j] })
		numBounds := bins[di] - 1
		for b := 0; b < numBounds; b++ {
			// pctIdx = floor(((b+1)/bins) * (count-1))  — matches QRust.
			idx := int(float64(b+1) / float64(bins[di]) * float64(int(count)-1))
			if idx < 0 {
				idx = 0
			}
			if idx >= int(count) {
				idx = int(count) - 1
			}
			out[wOff+b] = scratch[idx]
		}
		wOff += numBounds
	}
	return out
}

// bin returns the bin index ∈ [0, len(bounds)] for value v. Bounds are
// ascending; the bin is the count of boundaries that v exceeds.
func bin(v int16, bounds []int16) int {
	b := 0
	for _, bd := range bounds {
		if v > bd {
			b++
		} else {
			break
		}
	}
	return b
}

// swapPermuteRows permutes a slab of consecutive rows in `vectors` /
// `labels` so each row i (in absolute coords) ends up at target[i - base].
// Target indices are absolute (same coordinate space as `vectors`). The
// permutation is constrained: target values must lie in
// [base, base + count). This is the standard in-place swap permutation
// (https://en.wikipedia.org/wiki/In-place_matrix_transposition#Following_the_cycles):
// at each position i, swap row i with row target[i-base], and swap their
// target entries too. Each swap puts at least one row into its final
// resting place, so total swaps ≤ count.
func swapPermuteRows(vectors []int16, labels []uint8, target []uint32, base, count uint32) {
	var buf [Stride]int16
	for i := uint32(0); i < count; i++ {
		absI := base + i
		for target[i] != absI {
			t := target[i]
			tRel := t - base
			// swap row absI and row t
			copy(buf[:], vectors[absI*Stride:absI*Stride+Stride])
			copy(vectors[absI*Stride:absI*Stride+Stride], vectors[t*Stride:t*Stride+Stride])
			copy(vectors[t*Stride:t*Stride+Stride], buf[:])
			labels[absI], labels[t] = labels[t], labels[absI]
			// swap target entries to track where the rows went
			target[i], target[tRel] = target[tRel], target[i]
		}
	}
}
