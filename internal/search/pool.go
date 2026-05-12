package search

import "sync"

// maxCellsHint sizes the pooled scratch buffers for the worst-case partition.
// Grid V2 with bin layout (16, 8, 8) caps at 1024 cells.
const maxCellsHint = 1024

type scratch struct {
	cellLBs     []int64
	sortedCells []int
}

var scratchPool = sync.Pool{
	New: func() any {
		return &scratch{
			cellLBs:     make([]int64, maxCellsHint),
			sortedCells: make([]int, maxCellsHint),
		}
	},
}

func getScratch(numCells int) *scratch {
	s := scratchPool.Get().(*scratch)
	if cap(s.cellLBs) < numCells {
		s.cellLBs = make([]int64, numCells)
	} else {
		s.cellLBs = s.cellLBs[:numCells]
	}
	if cap(s.sortedCells) < numCells {
		s.sortedCells = make([]int, numCells)
	} else {
		s.sortedCells = s.sortedCells[:numCells]
	}
	return s
}

func putScratch(s *scratch) {
	scratchPool.Put(s)
}
