package search

import (
	"slices"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// FraudCountGrid is the production search path. See docs/CODE_NOTES.md
// "Grid V2" for the LB-pruning and AABB rationale.
func FraudCountGrid(query *[dims]int16, ds *dataset.Dataset) int {
	key := dataset.ComputeKey(query)
	pg := ds.Partitions[key]
	if pg == nil || pg.NumCells == 0 {
		return 0
	}

	numCells := pg.NumCells
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

	s := getScratch(numCells)
	defer putScratch(s)
	cellLBs := s.cellLBs
	for ci := 0; ci < numCells; ci++ {
		mi := ci * dims
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

	sortedCells := s.sortedCells
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

	const inf = int64(1 << 62)
	var (
		d0, d1, d2, d3, d4 = inf, inf, inf, inf, inf
		i0, i1, i2, i3, i4 = -1, -1, -1, -1, -1
	)

	vectors := ds.Vectors
	labels := ds.Labels

	for _, ci := range sortedCells {
		if cellLBs[ci] >= d4 {
			break
		}
		start := int(pg.CellStarts[ci])
		end := start + int(pg.CellCounts[ci])
		for i := start; i < end; i++ {
			base := i * dims

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

	frauds := 0
	if i0 >= 0 && labels[i0] == 1 {
		frauds++
	}
	if i1 >= 0 && labels[i1] == 1 {
		frauds++
	}
	if i2 >= 0 && labels[i2] == 1 {
		frauds++
	}
	if i3 >= 0 && labels[i3] == 1 {
		frauds++
	}
	if i4 >= 0 && labels[i4] == 1 {
		frauds++
	}
	return frauds
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
