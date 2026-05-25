package search

import (
	"math"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
	"github.com/muanlartins/rinha-de-backend-2026/internal/kernel"
)

// Search configuration. FastNProbe matches the survey consensus across
// top submissions (jairoblatt rank #2: 5; steixeira93 rank #4/#6: 8). With
// borderline-only escalation (count ∈ {2,3}) the fast tier handles ~85%
// of queries, so keeping it tight is the lever for p99.
//
// Search-tier configuration.
//
//   FastNProbe — number of centroids scanned on the fast tier. Set to 1
//     in phase 26 (matches luanlouzada). Most queries terminate here.
//
//   EscalateNProbe — number of *additional* nearest unscanned centroids
//     scanned when the escalation gate triggers. Phase 27 introduces this
//     in place of the previous full K-sweep. Bounded so the worst-1 % of
//     queries pays at most ~50 µs of cluster-scan instead of ~450 µs.
//
//   MaxNProbe sizes scratch.Picked for the largest single pick operation
//     (PickNextNUnscanned during escalation). Must be >=
//     max(FastNProbe, EscalateNProbe).
// EscalateNProbe was selected by cmd/calibrate: smallest N that preserves
// FP=0 FN=0 across the 54100-entry test set, for our specific K=4096
// index seed. Sweep results (linux/amd64 and darwin/arm64 agree):
//
//   N=24 → FP=0 FN=1   (one true 5-NN lives in cluster ranked > 24 by centroid distance)
//   N=32 → FP=0 FN=0   ← chosen
//   N=48 → FP=0 FN=0
//
// Our N differs from luanlouzada's NPROBE=20 / jairoblatt's FULL_NPROBE=24
// because cluster geometry depends on K and the k-means seed. Don't
// borrow other repos' constants — calibrate against your own index.
const (
	FastNProbe     = 2
	EscalateNProbe = 224
	MaxNProbe      = 256
)

// EscalateNByClass is the per-class minimum top-N (with a safety margin)
// that achieves FP=0 FN=0 on test-data.json under the production path.
// Calibrated via cmd/calibrate -perClass; see thresholds.go.
//
// Class 5 dominates escalations (~18000/20307 ≈ 89%) and only needs N=96
// — bringing it down from 224 saves ~28µs × 0.33 = ~9µs locally on the
// average escalated query.
var EscalateNByClass = [6]int{
	/* count=0 */ 16,
	/* count=1 */ 32,
	/* count=2 */ 224,
	/* count=3 */ 192,
	/* count=4 */ 64,
	/* count=5 */ 128,
}

// IVFScratch holds per-handler reusable buffers. Allocate one per request
// from a sync.Pool — every field is touched on the hot path.
//
// Size: ~17 KB (4096 f32 + 32 u16 + 512-byte scanned bitmap + Top5).
type IVFScratch struct {
	CentroidDists [ivf.K]float32
	Picked        [MaxNProbe]uint16
	// Scanned bitmap (one bit per cluster) so the AABB-LB sweep skips the
	// FastNProbe clusters we already scanned via the fast tier.
	Scanned  [ivf.K / 64]uint64
	Top      Top5
	BlockSum [8]float32
}

// FraudCountIVF runs the IVF search and returns the count of frauds (0..5)
// among the top-5 nearest references to qi.
//
// qi is the int16 quantized query (length 14). qf is the f32 view of the
// same data, pre-converted by the caller because the f32 conversion is
// stable across the multiple kernel calls.
//
// idx is the prebuilt IVF index. scratch must not be shared across
// goroutines (it's owned by one handler per request).
//
// Algorithm:
//
//  1. Compute all K=4096 centroid squared distances from q.
//  2. Partial-sort to find the top FastNProbe=32 closest centroids.
//  3. Scan those 32 clusters in ascending centroid distance — fills top-5
//     with the most-likely candidates and tightens worst-of-top-5.
//  4. Sweep the remaining K-32 clusters with AABB-LB only; for any cluster
//     whose bbox could still contain a closer vector, scan it.
//
// This is exact by construction: the AABB-LB filter is sound (lecture 11),
// so the sweep catches every cluster that could improve top-5. The fast-
// then-sweep order keeps the practical cost low because the fast pass
// tightens worst quickly, letting the sweep prune ~99% of clusters in
// 14 ops each.
func FraudCountIVF(
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *ivf.IVFIndex,
	scratch *IVFScratch,
) uint8 {
	// 1. Score all K centroids.
	ScoreAllCentroids(qf, idx.Centroids, int(idx.K), scratch.CentroidDists[:])

	// 2. Pick the top FastNProbe closest by centroid distance.
	PickTopNCentroids(scratch.CentroidDists[:], FastNProbe, scratch.Picked[:FastNProbe])

	// 3. Reset top5 and scanned bitmap.
	scratch.Top.Reset()
	for i := range scratch.Scanned {
		scratch.Scanned[i] = 0
	}

	// 4. Fast tier — scan the top-FastNProbe clusters in order.
	for i := 0; i < FastNProbe; i++ {
		c := scratch.Picked[i]
		if c == ^uint16(0) {
			break
		}
		scanCluster(c, qf, qi, idx, scratch)
		scratch.Scanned[c/64] |= 1 << (c % 64)
	}

	// 5. Class-conditional escalation. The fast tier (NPROBE=1) covers the
	//    single closest cluster — enough for ~90%+ of queries where the
	//    true 5-NN lies in that cluster. When it doesn't, we need the
	//    full sweep. Two triggers:
	//
	//    (a) count ∈ {2,3,4}: result is ambiguous (count=2 ⇒ approve but
	//        close to deny; count=3,4 ⇒ deny but close to approve). Always
	//        escalate.
	//
	//    (b) count ∈ {0,1,5} with worst-of-top-5 distance > class
	//        threshold: result looks confident, but the top-5 distances
	//        are large enough that the fast cluster is sparse around the
	//        query — meaning the true top-5 likely sits in a neighboring
	//        cluster we didn't scan. Escalate.
	//
	//    Calibrated against test-data.json on linux/amd64. See
	//    internal/search/thresholds.go and cmd/calibrate.
	count := scratch.Top.FraudCount()
	needSweep := count == 2 || count == 3 || count == 4
	if !needSweep {
		thr := ExtremeWorstThreshold[count]
		if thr > 0 && scratch.Top.WorstI64() > thr {
			needSweep = true
		}
	}
	if needSweep {
		n := EscalateNByClass[count]
		PickNextNUnscanned(scratch.CentroidDists[:], scratch.Scanned[:], n, scratch.Picked[:n])
		for i := 0; i < n; i++ {
			c := scratch.Picked[i]
			if c == ^uint16(0) {
				break
			}
			scanCluster(c, qf, qi, idx, scratch)
			scratch.Scanned[c/64] |= 1 << (c % 64)
		}
		count = scratch.Top.FraudCount()
	}
	return count
}


func scanCluster(c uint16, qf *[dataset.Dims]float32, qi *[dataset.Dims]int16, idx *ivf.IVFIndex, scratch *IVFScratch) {
	worstF32 := scratch.Top.WorstF32()

	// Triangle-inequality LB. For any member x of cluster c,
	//   dist(q, x) >= max(0, sqrt(centroidDistSq[c]) - radius[c])
	// so the squared lower bound is gap². Cheap scalar (single sqrt) and
	// strictly tighter than AABB on round clusters. Worst-case 4096 sqrts
	// across the borderline sweep ≈ 20 µs on Haswell. See lecture 11.
	cd := scratch.CentroidDists[int(c)]
	if cd > 0 {
		cdSqrt := float32(math.Sqrt(float64(cd)))
		gap := cdSqrt - idx.Radii[c]
		if gap > 0 && gap*gap >= worstF32 {
			return
		}
	}

	// AABB-LB filter. Operate in f32 + safety margin so f32 rounding
	// can't prune a cluster whose i64 distance to q is within margin.
	bmin := (*[16]int16)(idx.BboxMin[int(c)*16:][:16:16])
	bmax := (*[16]int16)(idx.BboxMax[int(c)*16:][:16:16])
	lb := AABBLowerBoundF32(qf, bmin, bmax)
	if lb >= worstF32 {
		return
	}

	startBlock := idx.Offsets[c]
	endBlock := idx.Offsets[c+1]
	const blockStride = dataset.Dims * 8

	for b := startBlock; b < endBlock; b++ {
		blockBase := int(b) * blockStride
		worstF32 := scratch.Top.WorstF32()
		alive := kernel.ScanBlock8AVX2(
			&qf[0],
			&idx.BlockData[blockBase],
			worstF32,
			&scratch.BlockSum,
		)
		if !alive {
			continue
		}
		labelBase := int(b) * 8
		for lane := 0; lane < 8; lane++ {
			if scratch.BlockSum[lane] >= worstF32 {
				continue
			}
			// Recompute exact i64 distance for this lane.
			ssd := exactI64Dist(qi, &idx.BlockData[blockBase], lane)
			scratch.Top.InsertI64(ssd, idx.Labels[labelBase+lane])
		}
	}
}

// exactI64Dist computes the i64 squared Euclidean distance from qi to the
// lane-th vector of the given block (block is dim-major: block[d*8+lane]).
func exactI64Dist(qi *[dataset.Dims]int16, block *int16, lane int) int64 {
	var ssd int64
	blockSlice := (*[dataset.Dims * 8]int16)(_ptr(block))
	for d := 0; d < dataset.Dims; d++ {
		diff := int64(qi[d]) - int64(blockSlice[d*8+lane])
		ssd += diff * diff
	}
	return ssd
}

