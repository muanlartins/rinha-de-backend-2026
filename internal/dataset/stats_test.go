package dataset

import (
	"os"
	"sort"
	"testing"
)

// TestPartitionStats: print partition + cell counts so we can spot anything
// pathological (a partition with <100 vectors would mean the algorithm can't
// always find 5 valid neighbors inside the partition, which would force a
// fallback we haven't implemented).
func TestPartitionStats(t *testing.T) {
	path := "../../references/rinha-official/resources/references.json.gz"
	if _, err := os.Stat(path); err != nil {
		t.Skip("references unavailable")
	}
	ds, err := LoadFromGzipJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	ds.BuildGrid()

	type pStat struct {
		p           int
		count       uint32
		numCells    uint32
		minCellCnt  uint32
		maxCellCnt  uint32
		avgCellCnt  uint32
	}
	stats := make([]pStat, 0, NumPartitions)
	totalCells := 0
	for p := 0; p < NumPartitions; p++ {
		cnt := ds.PartitionCounts[p]
		g := &ds.Grids[p]
		minC, maxC := uint32(1<<30), uint32(0)
		var sumC uint64
		for _, c := range g.Cells {
			if c.Count < minC {
				minC = c.Count
			}
			if c.Count > maxC {
				maxC = c.Count
			}
			sumC += uint64(c.Count)
		}
		avgC := uint32(0)
		if g.NumCells > 0 {
			avgC = uint32(sumC / uint64(g.NumCells))
		}
		stats = append(stats, pStat{p, cnt, g.NumCells, minC, maxC, avgC})
		totalCells += int(g.NumCells)
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].count < stats[j].count })

	t.Logf("3M vectors over %d non-empty partitions, %d cells total", NumPartitions, totalCells)
	for _, s := range stats {
		t.Logf("  p=%2d  vectors=%-8d  cells=%-5d  cell[min/avg/max]=%-5d/%-5d/%-5d",
			s.p, s.count, s.numCells, s.minCellCnt, s.avgCellCnt, s.maxCellCnt)
	}

	// Sanity: every non-empty partition has ≥ 5 vectors (else fewer than 5
	// possible neighbors).
	for _, s := range stats {
		if s.count > 0 && s.count < 5 {
			t.Errorf("partition %d has only %d vectors (need ≥5 for KNN-5)", s.p, s.count)
		}
	}
}
