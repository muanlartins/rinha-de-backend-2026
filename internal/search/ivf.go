package search

import (
	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// IVFNProbe: fast-path nprobe. With K=8192 and ~366 vectors/cluster,
// nprobe=32 scans ~11700 vectors per query.
var IVFNProbe = 32

// IVFAmbiguousNProbe: when fast probe gives a borderline count, expand.
// Trade: tail latency on the ~10% borderline queries vs accuracy.
var IVFAmbiguousNProbe = 128

// FraudCountIVF: fast path probes IVFNProbe centroids. If the result is
// ambiguous (fraud_count ∈ {2,3,4}), expand to IVFAmbiguousNProbe. Final
// fraud count is the tally of labels in the top-5.
func FraudCountIVF(query *[stride]int16, ds *dataset.Dataset) int {
	if ds.IVF == nil || len(ds.IVF.Clusters) == 0 {
		return 0
	}

	numClusters := len(ds.IVF.Clusters)
	probedIDs, probedDists := topCentroids(query, ds.IVF.CentroidBlocks, numClusters, IVFNProbe)

	tk := newTopK()
	if useAVX2 {
		scanIVFBlocks(query, ds.Blocks, ds.BlockLabels, probedIDs, probedDists, ds.IVF, &tk)
	} else {
		scanIVFScalar(query, ds.Blocks, ds.BlockLabels, probedIDs, probedDists, ds.IVF, &tk)
	}

	frauds := countFraudsTk(&tk)
	if frauds >= 2 && frauds <= 4 && IVFAmbiguousNProbe > IVFNProbe {
		// Borderline — probe more clusters and re-scan with the existing
		// top-5 carried over.
		moreIDs, moreDists := topCentroids(query, ds.IVF.CentroidBlocks, numClusters, IVFAmbiguousNProbe)
		// Skip the IDs we've already scanned (first IVFNProbe of `moreIDs`
		// are the same set if topCentroids is deterministic).
		extraIDs := moreIDs[IVFNProbe:]
		extraDists := moreDists[IVFNProbe:]
		if useAVX2 {
			scanIVFBlocks(query, ds.Blocks, ds.BlockLabels, extraIDs, extraDists, ds.IVF, &tk)
		} else {
			scanIVFScalar(query, ds.Blocks, ds.BlockLabels, extraIDs, extraDists, ds.IVF, &tk)
		}
		frauds = countFraudsTk(&tk)
	}
	return frauds
}

func countFraudsTk(tk *topK) int {
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

// topCentroids returns the nprobe cluster IDs with the smallest centroid
// distance to query, plus the distances themselves. Uses the batched 8-wide
// kernel against the block-major centroid layout.
func topCentroids(query *[stride]int16, centroidBlocks []int16, numClusters, nprobe int) (ids []uint32, dists []int64) {
	if nprobe > numClusters {
		nprobe = numClusters
	}
	bestIDs := make([]uint32, nprobe)
	bestD := make([]int64, nprobe)
	for i := range bestD {
		bestD[i] = (1 << 62)
	}
	count := 0

	if useAVX2 {
		var dbuf [8]int64
		qPtr := &query[0]
		nBlocks := (numClusters + 7) / 8
		for b := 0; b < nBlocks; b++ {
			blockOff := b * dataset.BlockInt16
			blockScan8AVX2(qPtr, &centroidBlocks[blockOff], &dbuf)
			for lane := 0; lane < 8; lane++ {
				ci := b*8 + lane
				if ci >= numClusters {
					break
				}
				d := dbuf[lane]
				if count < nprobe {
					insertCentroid(uint32(ci), d, bestIDs, bestD, count)
					count++
				} else if d < bestD[nprobe-1] {
					insertCentroid(uint32(ci), d, bestIDs, bestD, nprobe-1)
				}
			}
		}
	} else {
		for c := 0; c < numClusters; c++ {
			d := centroidDistScalar(query, centroidBlocks, c)
			if count < nprobe {
				insertCentroid(uint32(c), d, bestIDs, bestD, count)
				count++
			} else if d < bestD[nprobe-1] {
				insertCentroid(uint32(c), d, bestIDs, bestD, nprobe-1)
			}
		}
	}
	return bestIDs, bestD
}

func insertCentroid(c uint32, d int64, bestIDs []uint32, bestD []int64, last int) {
	i := last
	for i > 0 && d < bestD[i-1] {
		bestD[i] = bestD[i-1]
		bestIDs[i] = bestIDs[i-1]
		i--
	}
	bestD[i] = d
	bestIDs[i] = c
}

func centroidDistScalar(query *[stride]int16, centroidBlocks []int16, c int) int64 {
	blockOff := (c / 8) * dataset.BlockInt16
	lane := c % 8
	var sum int64
	for d := 0; d < stride; d++ {
		t := int64(query[d]) - int64(centroidBlocks[blockOff+d*dataset.BlockVectors+lane])
		sum += t * t
	}
	return sum
}

func scanIVFBlocks(query *[stride]int16, blocks []int16, blockLabels []uint8, probedIDs []uint32, _ []int64, ivf *dataset.IVFIndex, tk *topK) {
	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	l0, l1, l2, l3, l4 := tk.lab0, tk.lab1, tk.lab2, tk.lab3, tk.lab4
	qPtr := &query[0]
	var distBuf [8]int64

	for _, ci := range probedIDs {
		c := &ivf.Clusters[ci]
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

func scanIVFScalar(query *[stride]int16, blocks []int16, blockLabels []uint8, probedIDs []uint32, _ []int64, ivf *dataset.IVFIndex, tk *topK) {
	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	l0, l1, l2, l3, l4 := tk.lab0, tk.lab1, tk.lab2, tk.lab3, tk.lab4

	for _, ci := range probedIDs {
		c := &ivf.Clusters[ci]
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
