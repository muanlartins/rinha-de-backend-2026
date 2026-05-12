package search

import (
	"slices"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// IVFNProbe: number of clusters scanned per query, sorted by centroid LB
// ascending. The bbox LB is a true lower bound, so the algorithm is EXACT
// as long as we don't break out before LB exceeds the current top-5 — even
// at small nprobe. The parameter mostly trades probe cost vs. early-quit
// quality.
var IVFNProbe = 128

// FraudCountIVF: exact KNN-5 via IVF + per-block 8-wide SIMD scan.
// Algorithm:
//  1. Compute bbox LB to all clusters in the query's partition.
//  2. Sort clusters by LB ascending.
//  3. Walk clusters in that order; for each, scan its blocks with
//     blockScan8AVX2 and update the top-5. Break when bbox LB ≥ topD5.
//
// Correctness: identical results to brute force as long as the bbox LB is
// computed correctly (it is — see clusterLB).
func FraudCountIVF(query *[stride]int16, ds *dataset.Dataset) int {
	key := dataset.ComputeKey(query)
	pg := ds.IVF[key]
	if pg == nil || len(pg.Clusters) == 0 {
		return 0
	}

	numClusters := len(pg.Clusters)
	s := getScratch(numClusters)
	defer putScratch(s)

	cellLBs := s.cellLBs[:numClusters]
	for ci := 0; ci < numClusters; ci++ {
		cellLBs[ci] = clusterLB(query, &pg.Clusters[ci])
	}

	sorted := s.sortedCells[:numClusters]
	for i := range sorted {
		sorted[i] = i
	}
	slices.SortFunc(sorted, func(a, b int) int {
		switch {
		case cellLBs[a] < cellLBs[b]:
			return -1
		case cellLBs[a] > cellLBs[b]:
			return 1
		default:
			return 0
		}
	})

	tk := newTopK()
	if useAVX2 {
		scanIVFBlocks(query, ds.Blocks, ds.BlockLabels, sorted, cellLBs, pg, &tk)
	} else {
		scanIVFScalar(query, ds.Blocks, ds.BlockLabels, sorted, cellLBs, pg, &tk)
	}

	frauds := 0
	if tk.lab0 == 1 {
		frauds++
	}
	if tk.lab1 == 1 {
		frauds++
	}
	if tk.lab2 == 1 {
		frauds++
	}
	if tk.lab3 == 1 {
		frauds++
	}
	if tk.lab4 == 1 {
		frauds++
	}
	return frauds
}

func clusterLB(query *[stride]int16, c *dataset.Cluster) int64 {
	var lb int64
	for d := 0; d < stride; d++ {
		q := int32(query[d])
		mn := int32(c.BboxMin[d])
		mx := int32(c.BboxMax[d])
		switch {
		case q < mn:
			t := int64(mn - q)
			lb += t * t
		case q > mx:
			t := int64(q - mx)
			lb += t * t
		}
	}
	return lb
}

func scanIVFBlocks(query *[stride]int16, blocks []int16, blockLabels []uint8, sorted []int, cellLBs []int64, pg *dataset.IVFPartition, tk *topK) {
	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	l0, l1, l2, l3, l4 := tk.lab0, tk.lab1, tk.lab2, tk.lab3, tk.lab4

	qPtr := &query[0]
	var distBuf [8]int64

	for _, ci := range sorted {
		if cellLBs[ci] >= d4 {
			break
		}
		c := &pg.Clusters[ci]
		blockOff := c.BlockStart
		labelOff := c.LabelStart
		count := c.Count
		nb := c.NumBlocks
		for b := uint32(0); b < nb; b++ {
			blockScan8AVX2(qPtr, &blocks[blockOff+b*dataset.BlockInt16], &distBuf)
			base := labelOff + b*dataset.BlockVectors
			lim := uint32(dataset.BlockVectors)
			realInBlock := count - b*dataset.BlockVectors
			if realInBlock < lim {
				lim = realInBlock
			}
			for v := uint32(0); v < lim; v++ {
				dist := distBuf[v]
				if dist >= d4 {
					continue
				}
				lab := blockLabels[base+v]
				switch {
				case dist < d0:
					d4, l4 = d3, l3
					d3, l3 = d2, l2
					d2, l2 = d1, l1
					d1, l1 = d0, l0
					d0, l0 = dist, lab
				case dist < d1:
					d4, l4 = d3, l3
					d3, l3 = d2, l2
					d2, l2 = d1, l1
					d1, l1 = dist, lab
				case dist < d2:
					d4, l4 = d3, l3
					d3, l3 = d2, l2
					d2, l2 = dist, lab
				case dist < d3:
					d4, l4 = d3, l3
					d3, l3 = dist, lab
				default:
					d4, l4 = dist, lab
				}
			}
		}
	}

	tk.d0, tk.d1, tk.d2, tk.d3, tk.d4 = d0, d1, d2, d3, d4
	tk.lab0, tk.lab1, tk.lab2, tk.lab3, tk.lab4 = l0, l1, l2, l3, l4
}

func scanIVFScalar(query *[stride]int16, blocks []int16, blockLabels []uint8, sorted []int, cellLBs []int64, pg *dataset.IVFPartition, tk *topK) {
	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	l0, l1, l2, l3, l4 := tk.lab0, tk.lab1, tk.lab2, tk.lab3, tk.lab4

	for _, ci := range sorted {
		if cellLBs[ci] >= d4 {
			break
		}
		c := &pg.Clusters[ci]
		blockOff := c.BlockStart
		labelOff := c.LabelStart
		count := c.Count
		nb := c.NumBlocks
		for b := uint32(0); b < nb; b++ {
			base := blockOff + b*dataset.BlockInt16
			labBase := labelOff + b*dataset.BlockVectors
			lim := uint32(dataset.BlockVectors)
			realInBlock := count - b*dataset.BlockVectors
			if realInBlock < lim {
				lim = realInBlock
			}
			for v := uint32(0); v < lim; v++ {
				var sum int64
				for d := 0; d < stride; d++ {
					t := int64(query[d]) - int64(blocks[base+uint32(d)*dataset.BlockVectors+v])
					sum += t * t
				}
				if sum >= d4 {
					continue
				}
				lab := blockLabels[labBase+v]
				switch {
				case sum < d0:
					d4, l4 = d3, l3
					d3, l3 = d2, l2
					d2, l2 = d1, l1
					d1, l1 = d0, l0
					d0, l0 = sum, lab
				case sum < d1:
					d4, l4 = d3, l3
					d3, l3 = d2, l2
					d2, l2 = d1, l1
					d1, l1 = sum, lab
				case sum < d2:
					d4, l4 = d3, l3
					d3, l3 = d2, l2
					d2, l2 = sum, lab
				case sum < d3:
					d4, l4 = d3, l3
					d3, l3 = sum, lab
				default:
					d4, l4 = sum, lab
				}
			}
		}
	}

	tk.d0, tk.d1, tk.d2, tk.d3, tk.d4 = d0, d1, d2, d3, d4
	tk.lab0, tk.lab1, tk.lab2, tk.lab3, tk.lab4 = l0, l1, l2, l3, l4
}
