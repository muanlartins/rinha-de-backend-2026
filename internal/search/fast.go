package search

import (
	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
)

// FraudCountFastOnly runs only the fast-tier scan (no escalation) and
// returns the resulting top-5 fraud count and worst-of-top-5 i64 squared
// distance.
//
// This exists for the calibrate tool — production code calls
// FraudCountIVF, which adds class-conditional escalation on top.
//
// nprobe overrides the package-level FastNProbe so calibration can sweep
// candidate values without changing the runtime constant.
func FraudCountFastOnly(
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *ivf.IVFIndex,
	scratch *IVFScratch,
	nprobe int,
) (count uint8, worstI64 int64) {
	if nprobe > MaxNProbe {
		nprobe = MaxNProbe
	}
	ScoreAllCentroids(qf, idx.Centroids, int(idx.K), scratch.CentroidDists[:])
	PickTopNCentroids(scratch.CentroidDists[:], nprobe, scratch.Picked[:nprobe])

	scratch.Top.Reset()
	for i := range scratch.Scanned {
		scratch.Scanned[i] = 0
	}
	for i := 0; i < nprobe; i++ {
		c := scratch.Picked[i]
		if c == ^uint16(0) {
			break
		}
		scanCluster(c, qf, qi, idx, scratch)
		scratch.Scanned[c/64] |= 1 << (c % 64)
	}
	return scratch.Top.FraudCount(), scratch.Top.WorstI64()
}

// FraudCountFull runs the fast tier + unconditional sweep over all
// remaining clusters. This is the most accurate result the IVF index
// can produce (modulo f32 precision in the kernel). Used by calibrate
// as the "oracle" against which fast-tier and top-N results are
// compared.
func FraudCountFull(
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *ivf.IVFIndex,
	scratch *IVFScratch,
) uint8 {
	_, _ = FraudCountFastOnly(qf, qi, idx, scratch, FastNProbe)
	for c := uint16(0); c < uint16(ivf.K); c++ {
		if scratch.Scanned[c/64]&(1<<(c%64)) != 0 {
			continue
		}
		scanCluster(c, qf, qi, idx, scratch)
	}
	return scratch.Top.FraudCount()
}

// FraudCountTopN runs the fast tier (FastNProbe clusters), then if the
// escalation gate (count ∈ {2,3,4} or worst > ExtremeWorstThreshold[count])
// triggers, scans the next escalateN nearest unscanned centroids — this
// is the *production* phase-27 path. Used by calibrate to verify FP/FN
// at the chosen N before shipping.
func FraudCountTopN(
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *ivf.IVFIndex,
	scratch *IVFScratch,
	escalateN int,
) uint8 {
	_, _ = FraudCountFastOnly(qf, qi, idx, scratch, FastNProbe)
	count := scratch.Top.FraudCount()
	needSweep := count == 2 || count == 3 || count == 4
	if !needSweep {
		thr := ExtremeWorstThreshold[count]
		if thr > 0 && scratch.Top.WorstI64() > thr {
			needSweep = true
		}
	}
	if !needSweep {
		return count
	}
	if escalateN > MaxNProbe {
		escalateN = MaxNProbe
	}
	var escPicked [MaxNProbe]uint16
	PickNextNUnscanned(scratch.CentroidDists[:], scratch.Scanned[:], escalateN, escPicked[:escalateN])
	for i := 0; i < escalateN; i++ {
		c := escPicked[i]
		if c == ^uint16(0) {
			break
		}
		scanCluster(c, qf, qi, idx, scratch)
		scratch.Scanned[c/64] |= 1 << (c % 64)
	}
	return scratch.Top.FraudCount()
}

// FraudCountTopNByClass is like FraudCountTopN but takes a per-class
// escalation N. Used by calibrate to find the minimum N per class.
func FraudCountTopNByClass(
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *ivf.IVFIndex,
	scratch *IVFScratch,
	escalateN *[6]int,
) uint8 {
	_, _ = FraudCountFastOnly(qf, qi, idx, scratch, FastNProbe)
	count := scratch.Top.FraudCount()
	needSweep := count == 2 || count == 3 || count == 4
	if !needSweep {
		thr := ExtremeWorstThreshold[count]
		if thr > 0 && scratch.Top.WorstI64() > thr {
			needSweep = true
		}
	}
	if !needSweep {
		return count
	}
	n := escalateN[count]
	if n > MaxNProbe {
		n = MaxNProbe
	}
	if n <= 0 {
		return count
	}
	var escPicked [MaxNProbe]uint16
	PickNextNUnscanned(scratch.CentroidDists[:], scratch.Scanned[:], n, escPicked[:n])
	for i := 0; i < n; i++ {
		c := escPicked[i]
		if c == ^uint16(0) {
			break
		}
		scanCluster(c, qf, qi, idx, scratch)
		scratch.Scanned[c/64] |= 1 << (c % 64)
	}
	return scratch.Top.FraudCount()
}
