package search

import "sync"

// maxCellsHint is the worst-case number of cells across all partitions in the
// reference dataset. Used to size the scratch buffers so we never have to
// grow them. Grid v2 with bin layout (16, 8, 8) caps at 1024 cells; QRust's
// build typically lands ~370 cells per non-empty partition.
const maxCellsHint = 1024

// scratch holds per-request reusable buffers for the grid search.
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
