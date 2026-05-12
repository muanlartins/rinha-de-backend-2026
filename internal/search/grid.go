// Package search implements exact KNN-5 over the partitioned grid index.
// The algorithm (see docs/lectures/05-grid-revival.md):
//   1. Compute the 5-bit partition key from the query.
//   2. Compute lower-bound (LB) distance to each of that partition's cells
//      using per-cell AABBs.
//   3. Sort cells by ascending LB.
//   4. Walk cells in LB order; break when LB ≥ topD4. Inside each cell,
//      run the per-vector early-exit kernel skipping the partition's
//      constant dims.
//   5. Tally fraud labels in the final top-5.
//
// Correctness: identical to brute-force-over-the-partition, which is
// identical to brute-force-over-the-dataset because non-matching
// partitions contribute ≥1 to squared distance (well above any plausible
// 5th-NN squared distance once the grid pruning closes in).
package search

import (
	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

const (
	k      = 5
	stride = dataset.Stride
)

// scratch holds per-handler reusable buffers, sized for the largest
// possible cell count (BinsTotal = 1024).
type scratch struct {
	lbs    [dataset.BinsTotal]int64
	sorted [dataset.BinsTotal]uint16
}

// FraudCount computes the count of fraud labels in the 5 nearest neighbors
// of `query` inside `ds`. Exact (modulo int16 quantization).
//
// query[0..13] must be already quantized; query is a stack array of size
// dataset.Stride.
func FraudCount(query *[stride]int16, ds *dataset.Dataset) int {
	key := dataset.PartitionKey(query[:])
	g := &ds.Grids[key]
	if g.NumCells == 0 {
		return 0
	}

	// Detect which dims are constant in this partition:
	// dims 9, 10, 11 are always constant (by partition definition);
	// dim 5 is constant (= sentinel) when key bit 8 is set;
	// dim 6 is constant when key bit 16 is set.
	sentinel5 := key&0x08 != 0
	sentinel6 := key&0x10 != 0

	q0, q1, q2, q3 := query[0], query[1], query[2], query[3]
	q4, q5, q6, q7 := query[4], query[5], query[6], query[7]
	q8, q12, q13 := query[8], query[12], query[13]

	var s scratch
	cells := g.Cells

	// === Stage 1: LB per cell. Two specialized versions (sentinel vs not)
	// — extracted to keep the hot loop branch-free on the per-dim test.
	if sentinel5 && sentinel6 {
		// 9 variable dims: 0..4, 7, 8, 12, 13. (5, 6 sentinel; 9, 10, 11 const.)
		for ci := uint32(0); ci < g.NumCells; ci++ {
			c := &cells[ci]
			var lb int64
			lb += axisLB(q0, c.BboxMn[0], c.BboxMx[0])
			lb += axisLB(q1, c.BboxMn[1], c.BboxMx[1])
			lb += axisLB(q2, c.BboxMn[2], c.BboxMx[2])
			lb += axisLB(q3, c.BboxMn[3], c.BboxMx[3])
			lb += axisLB(q4, c.BboxMn[4], c.BboxMx[4])
			lb += axisLB(q7, c.BboxMn[7], c.BboxMx[7])
			lb += axisLB(q8, c.BboxMn[8], c.BboxMx[8])
			lb += axisLB(q12, c.BboxMn[12], c.BboxMx[12])
			lb += axisLB(q13, c.BboxMn[13], c.BboxMx[13])
			s.lbs[ci] = lb
		}
	} else if sentinel5 {
		// 10 variable dims: 0..4, 6, 7, 8, 12, 13. (5 sentinel; 9, 10, 11 const.)
		for ci := uint32(0); ci < g.NumCells; ci++ {
			c := &cells[ci]
			var lb int64
			lb += axisLB(q0, c.BboxMn[0], c.BboxMx[0])
			lb += axisLB(q1, c.BboxMn[1], c.BboxMx[1])
			lb += axisLB(q2, c.BboxMn[2], c.BboxMx[2])
			lb += axisLB(q3, c.BboxMn[3], c.BboxMx[3])
			lb += axisLB(q4, c.BboxMn[4], c.BboxMx[4])
			lb += axisLB(q6, c.BboxMn[6], c.BboxMx[6])
			lb += axisLB(q7, c.BboxMn[7], c.BboxMx[7])
			lb += axisLB(q8, c.BboxMn[8], c.BboxMx[8])
			lb += axisLB(q12, c.BboxMn[12], c.BboxMx[12])
			lb += axisLB(q13, c.BboxMn[13], c.BboxMx[13])
			s.lbs[ci] = lb
		}
	} else if sentinel6 {
		// 10 variable dims: 0..5, 7, 8, 12, 13. (6 sentinel; 9, 10, 11 const.)
		for ci := uint32(0); ci < g.NumCells; ci++ {
			c := &cells[ci]
			var lb int64
			lb += axisLB(q0, c.BboxMn[0], c.BboxMx[0])
			lb += axisLB(q1, c.BboxMn[1], c.BboxMx[1])
			lb += axisLB(q2, c.BboxMn[2], c.BboxMx[2])
			lb += axisLB(q3, c.BboxMn[3], c.BboxMx[3])
			lb += axisLB(q4, c.BboxMn[4], c.BboxMx[4])
			lb += axisLB(q5, c.BboxMn[5], c.BboxMx[5])
			lb += axisLB(q7, c.BboxMn[7], c.BboxMx[7])
			lb += axisLB(q8, c.BboxMn[8], c.BboxMx[8])
			lb += axisLB(q12, c.BboxMn[12], c.BboxMx[12])
			lb += axisLB(q13, c.BboxMn[13], c.BboxMx[13])
			s.lbs[ci] = lb
		}
	} else {
		// 11 variable dims: 0..8, 12, 13. (9, 10, 11 const.)
		for ci := uint32(0); ci < g.NumCells; ci++ {
			c := &cells[ci]
			var lb int64
			lb += axisLB(q0, c.BboxMn[0], c.BboxMx[0])
			lb += axisLB(q1, c.BboxMn[1], c.BboxMx[1])
			lb += axisLB(q2, c.BboxMn[2], c.BboxMx[2])
			lb += axisLB(q3, c.BboxMn[3], c.BboxMx[3])
			lb += axisLB(q4, c.BboxMn[4], c.BboxMx[4])
			lb += axisLB(q5, c.BboxMn[5], c.BboxMx[5])
			lb += axisLB(q6, c.BboxMn[6], c.BboxMx[6])
			lb += axisLB(q7, c.BboxMn[7], c.BboxMx[7])
			lb += axisLB(q8, c.BboxMn[8], c.BboxMx[8])
			lb += axisLB(q12, c.BboxMn[12], c.BboxMx[12])
			lb += axisLB(q13, c.BboxMn[13], c.BboxMx[13])
			s.lbs[ci] = lb
		}
	}

	// === Stage 2: sort cell indices by LB ascending.
	n := int(g.NumCells)
	for i := 0; i < n; i++ {
		s.sorted[i] = uint16(i)
	}
	insertionSortByLB(s.sorted[:n], s.lbs[:])

	// === Stage 3: walk in LB order, scan each cell with the early-exit
	// kernel, break when LB ≥ topD4.
	pStart := ds.PartitionStarts[key]
	const inf int64 = 1 << 62
	d0, d1, d2, d3, d4 := inf, inf, inf, inf, inf
	var l0, l1, l2, l3, l4 uint8

	vectors := ds.Vectors
	labels := ds.Labels

	if sentinel5 && sentinel6 {
		for si := 0; si < n; si++ {
			ci := s.sorted[si]
			if s.lbs[ci] >= d4 {
				break
			}
			cell := &cells[ci]
			startVec := pStart + cell.Start
			endVec := startVec + cell.Count
			for i := startVec; i < endVec; i++ {
				base := i * stride
				v := vectors[base : base+stride : base+stride]
				// 9 variable dims, ordered for fastest early-exit (high-variance first):
				diff := int64(q0) - int64(v[0])
				dist := diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q12) - int64(v[12])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q7) - int64(v[7])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q13) - int64(v[13])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q2) - int64(v[2])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q1) - int64(v[1])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q3) - int64(v[3])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q4) - int64(v[4])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q8) - int64(v[8])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				lab := labels[i]
				d0, d1, d2, d3, d4, l0, l1, l2, l3, l4 = topKInsert(
					dist, lab, d0, d1, d2, d3, d4, l0, l1, l2, l3, l4)
			}
		}
	} else if sentinel5 {
		for si := 0; si < n; si++ {
			ci := s.sorted[si]
			if s.lbs[ci] >= d4 {
				break
			}
			cell := &cells[ci]
			startVec := pStart + cell.Start
			endVec := startVec + cell.Count
			for i := startVec; i < endVec; i++ {
				base := i * stride
				v := vectors[base : base+stride : base+stride]
				// 10 variable dims (5 sentinel; 9, 10, 11 const).
				diff := int64(q0) - int64(v[0])
				dist := diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q12) - int64(v[12])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q6) - int64(v[6])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q7) - int64(v[7])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q13) - int64(v[13])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q2) - int64(v[2])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q1) - int64(v[1])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q3) - int64(v[3])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q4) - int64(v[4])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q8) - int64(v[8])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				lab := labels[i]
				d0, d1, d2, d3, d4, l0, l1, l2, l3, l4 = topKInsert(
					dist, lab, d0, d1, d2, d3, d4, l0, l1, l2, l3, l4)
			}
		}
	} else if sentinel6 {
		for si := 0; si < n; si++ {
			ci := s.sorted[si]
			if s.lbs[ci] >= d4 {
				break
			}
			cell := &cells[ci]
			startVec := pStart + cell.Start
			endVec := startVec + cell.Count
			for i := startVec; i < endVec; i++ {
				base := i * stride
				v := vectors[base : base+stride : base+stride]
				// 10 variable dims (6 sentinel; 9, 10, 11 const).
				diff := int64(q0) - int64(v[0])
				dist := diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q12) - int64(v[12])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q5) - int64(v[5])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q7) - int64(v[7])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q13) - int64(v[13])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q2) - int64(v[2])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q1) - int64(v[1])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q3) - int64(v[3])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q4) - int64(v[4])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q8) - int64(v[8])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				lab := labels[i]
				d0, d1, d2, d3, d4, l0, l1, l2, l3, l4 = topKInsert(
					dist, lab, d0, d1, d2, d3, d4, l0, l1, l2, l3, l4)
			}
		}
	} else {
		for si := 0; si < n; si++ {
			ci := s.sorted[si]
			if s.lbs[ci] >= d4 {
				break
			}
			cell := &cells[ci]
			startVec := pStart + cell.Start
			endVec := startVec + cell.Count
			for i := startVec; i < endVec; i++ {
				base := i * stride
				v := vectors[base : base+stride : base+stride]
				// 11 variable dims (9, 10, 11 const).
				diff := int64(q0) - int64(v[0])
				dist := diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q12) - int64(v[12])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q6) - int64(v[6])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q5) - int64(v[5])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q7) - int64(v[7])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q13) - int64(v[13])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q2) - int64(v[2])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q1) - int64(v[1])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q3) - int64(v[3])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q4) - int64(v[4])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				diff = int64(q8) - int64(v[8])
				dist += diff * diff
				if dist >= d4 {
					continue
				}
				lab := labels[i]
				d0, d1, d2, d3, d4, l0, l1, l2, l3, l4 = topKInsert(
					dist, lab, d0, d1, d2, d3, d4, l0, l1, l2, l3, l4)
			}
		}
	}

	frauds := 0
	if l0 == 1 {
		frauds++
	}
	if l1 == 1 {
		frauds++
	}
	if l2 == 1 {
		frauds++
	}
	if l3 == 1 {
		frauds++
	}
	if l4 == 1 {
		frauds++
	}
	return frauds
}

// axisLB returns the minimum squared contribution to the distance along
// one axis from q to the closed interval [mn, mx]:
//   - 0 if mn ≤ q ≤ mx (q is "inside" the box on this axis)
//   - (q - mn)^2 if q < mn
//   - (q - mx)^2 if q > mx
//
// Branchless via int64 widening + max(0, ...) on the two deltas.
func axisLB(q, mn, mx int16) int64 {
	qi := int64(q)
	if qi < int64(mn) {
		d := int64(mn) - qi
		return d * d
	}
	if qi > int64(mx) {
		d := qi - int64(mx)
		return d * d
	}
	return 0
}

// insertionSortByLB sorts the indices in `idx` so that `lbs[idx[i]]` is
// ascending. Insertion sort is fastest in this size range (≤1024) and has
// no allocation.
func insertionSortByLB(idx []uint16, lbs []int64) {
	for i := 1; i < len(idx); i++ {
		j := i
		cur := idx[i]
		curVal := lbs[cur]
		for j > 0 && lbs[idx[j-1]] > curVal {
			idx[j] = idx[j-1]
			j--
		}
		idx[j] = cur
	}
}

// topKInsert is the cascading 5-slot shift. Inputs already satisfy
// d0 ≤ d1 ≤ d2 ≤ d3 ≤ d4 (with paired labels). Returns the same shape
// after inserting (dist, lab) iff dist < d4. The caller already filters
// `dist >= d4` before reaching the top-5 maintenance, so this is the path
// that actually runs only on the "new top-5" hot trail.
func topKInsert(
	dist int64, lab uint8,
	d0, d1, d2, d3, d4 int64,
	l0, l1, l2, l3, l4 uint8,
) (int64, int64, int64, int64, int64, uint8, uint8, uint8, uint8, uint8) {
	switch {
	case dist < d0:
		return dist, d0, d1, d2, d3, lab, l0, l1, l2, l3
	case dist < d1:
		return d0, dist, d1, d2, d3, l0, lab, l1, l2, l3
	case dist < d2:
		return d0, d1, dist, d2, d3, l0, l1, lab, l2, l3
	case dist < d3:
		return d0, d1, d2, dist, d3, l0, l1, l2, lab, l3
	default:
		return d0, d1, d2, d3, dist, l0, l1, l2, l3, lab
	}
}
