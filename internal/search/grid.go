package search

import (
	"slices"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// topK holds the five nearest neighbors observed so far, sorted by distance.
type topK struct {
	d0, d1, d2, d3, d4 int64
	i0, i1, i2, i3, i4 int
}

func newTopK() topK {
	const inf = int64(1 << 62)
	return topK{
		d0: inf, d1: inf, d2: inf, d3: inf, d4: inf,
		i0: -1, i1: -1, i2: -1, i3: -1, i4: -1,
	}
}

// FraudCountGrid is the production search path. See docs/CODE_NOTES.md
// "Grid V2" for the LB-pruning and AABB rationale.
func FraudCountGrid(query *[stride]int16, ds *dataset.Dataset) int {
	key := dataset.ComputeKey(query)
	pg := ds.Partitions[key]
	if pg == nil || pg.NumCells == 0 {
		return 0
	}

	numCells := pg.NumCells
	s := getScratch(numCells)
	defer putScratch(s)
	cellLBs, sortedCells := computeAndSortLBs(query, key, pg, s)

	tk := newTopK()
	if useAVX2 {
		scanCellsAVX2(query, ds.Vectors, sortedCells, cellLBs, pg, &tk)
	} else {
		isSentinel5 := (key & 0x08) != 0
		isSentinel6 := (key & 0x10) != 0
		scanCellsScalar(query, ds.Vectors, sortedCells, cellLBs, pg, isSentinel5, isSentinel6, &tk)
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

func computeAndSortLBs(query *[stride]int16, key uint8, pg *dataset.Partition, s *scratch) (cellLBs []int64, sortedCells []int) {
	isSentinel5 := (key & 0x08) != 0
	isSentinel6 := (key & 0x10) != 0

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

	bboxMin := pg.BboxMin
	bboxMax := pg.BboxMax

	cellLBs = s.cellLBs
	numCells := len(cellLBs)
	for ci := 0; ci < numCells; ci++ {
		mi := ci * stride
		var lb int64
		lb = addAxisLB(lb, q0, int32(bboxMin[mi]), int32(bboxMax[mi]))
		lb = addAxisLB(lb, q1, int32(bboxMin[mi+1]), int32(bboxMax[mi+1]))
		lb = addAxisLB(lb, q2, int32(bboxMin[mi+2]), int32(bboxMax[mi+2]))
		lb = addAxisLB(lb, q3, int32(bboxMin[mi+3]), int32(bboxMax[mi+3]))
		lb = addAxisLB(lb, q4, int32(bboxMin[mi+4]), int32(bboxMax[mi+4]))
		if !isSentinel5 {
			lb = addAxisLB(lb, q5, int32(bboxMin[mi+5]), int32(bboxMax[mi+5]))
		}
		if !isSentinel6 {
			lb = addAxisLB(lb, q6, int32(bboxMin[mi+6]), int32(bboxMax[mi+6]))
		}
		lb = addAxisLB(lb, q7, int32(bboxMin[mi+7]), int32(bboxMax[mi+7]))
		lb = addAxisLB(lb, q8, int32(bboxMin[mi+8]), int32(bboxMax[mi+8]))
		lb = addAxisLB(lb, q12, int32(bboxMin[mi+12]), int32(bboxMax[mi+12]))
		lb = addAxisLB(lb, q13, int32(bboxMin[mi+13]), int32(bboxMax[mi+13]))
		cellLBs[ci] = lb
	}

	sortedCells = s.sortedCells
	for i := range sortedCells {
		sortedCells[i] = i
	}
	slices.SortFunc(sortedCells, func(a, b int) int {
		switch {
		case cellLBs[a] < cellLBs[b]:
			return -1
		case cellLBs[a] > cellLBs[b]:
			return 1
		default:
			return 0
		}
	})

	return cellLBs, sortedCells
}

// scanCellsScalar walks the sorted cells with the early-exit dim-by-dim kernel.
func scanCellsScalar(query *[stride]int16, vectors []int16, sortedCells []int, cellLBs []int64, pg *dataset.Partition, isSentinel5, isSentinel6 bool, tk *topK) {
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

	for _, ci := range sortedCells {
		if cellLBs[ci] >= d4 {
			break
		}
		start := int(pg.CellStarts[ci])
		end := start + int(pg.CellCounts[ci])
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

// scanCellsAVX2 computes the full 16-lane squared distance per vector via
// AVX2 (no per-dim early-exit) and inserts into the top-K. Padded dims (14,
// 15) and partition-constant dims contribute 0 because both query and ref
// hold equal values there.
func scanCellsAVX2(query *[stride]int16, vectors []int16, sortedCells []int, cellLBs []int64, pg *dataset.Partition, tk *topK) {
	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	i0, i1, i2, i3, i4 := tk.i0, tk.i1, tk.i2, tk.i3, tk.i4

	qPtr := &query[0]

	for _, ci := range sortedCells {
		if cellLBs[ci] >= d4 {
			break
		}
		start := int(pg.CellStarts[ci])
		end := start + int(pg.CellCounts[ci])
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

func addAxisLB(lb int64, q, min, max int32) int64 {
	if q < min {
		d := int64(min - q)
		return lb + d*d
	}
	if q > max {
		d := int64(q - max)
		return lb + d*d
	}
	return lb
}
