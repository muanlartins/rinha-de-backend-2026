package ivf

import (
	"math"
	"math/rand/v2"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// K is the number of IVF clusters. Phase 30 experimented with K=8192
// (smaller clusters, ~366 vec/cluster) and K=2048 (larger clusters,
// ~1465 vec/cluster). Both produced a persistent FN=1 that no practical
// top-N could recover — the full-K AABB-LB sweep catches the entry via
// bbox pruning but ranking by centroid distance does not include the
// containing cluster within any tractable N.
//
// K=4096 turned out to be a structural sweet spot for our top-N
// escalation strategy. Both directions regress. See JOURNEY phase 30.
const K = 4096

// SampleSize is the row count used to train Lloyd's algorithm. We do not
// train on the full 3M; we train on a sample, then do a single full-data
// assignment pass. Picking ~16*K keeps every cluster well-represented while
// keeping each Lloyd iteration under a second.
const SampleSize = 65536

// MaxIters is the cap on Lloyd iterations. Six is enough on this dataset —
// `andrade-cpp-ivf` (rank #1) uses six. Beyond that the inertia improvement
// is negligible.
const MaxIters = 6

// rngSeed is fixed so two `docker build`s on the same machine produce a
// byte-identical index. See lecture 09 § Determinism.
var rngSeed = [2]uint64{0xCAFEBABEDEADBEEF, 0xC0FFEE0123456789}

// TrainKMeans runs k-means++ initialization on a sample of the dataset, then
// Lloyd's algorithm to refine the centroids. Returns K centroids each of
// length 14 (one per dim).
//
// The input `vectors` is the flat int16 slab of length count*14 used by the
// rest of the project. We convert int16 → float32 internally because Lloyd's
// updates require accumulating means in fp.
func TrainKMeans(vectors []int16, count int) (centroids [K][dataset.Dims]float32) {
	rng := rand.New(rand.NewPCG(rngSeed[0], rngSeed[1]))

	sample := drawSample(vectors, count, SampleSize, rng)

	initKMeansPlusPlus(sample, &centroids, rng)

	for iter := 0; iter < MaxIters; iter++ {
		assign := assignSample(sample, &centroids)
		updateCentroids(sample, assign, &centroids)
	}

	return centroids
}

// drawSample picks SampleSize random vector ids from [0, count) without
// replacement and returns their f32 dim values laid out row-major.
//
// We use a partial Fisher-Yates: O(SampleSize) random picks against an
// implicit identity permutation, swapping the picked id into the front. We
// do NOT materialize the full permutation array — at 3M * 4B = 12 MB it'd
// fit, but the partial-swap is half the memory traffic.
func drawSample(vectors []int16, count, n int, rng *rand.Rand) []float32 {
	if n > count {
		n = count
	}
	ids := make([]uint32, n)
	pickedSet := make(map[uint32]struct{}, n)
	for i := 0; i < n; i++ {
		var id uint32
		for {
			id = uint32(rng.IntN(count))
			if _, dup := pickedSet[id]; !dup {
				break
			}
		}
		pickedSet[id] = struct{}{}
		ids[i] = id
	}

	out := make([]float32, n*dataset.Dims)
	for i, id := range ids {
		src := int(id) * dataset.Stride
		dst := i * dataset.Dims
		for d := 0; d < dataset.Dims; d++ {
			out[dst+d] = float32(vectors[src+d])
		}
	}
	return out
}

// initKMeansPlusPlus seeds K centroids using the k-means++ rule: the first
// centroid is uniform-random, and each subsequent centroid is chosen with
// probability proportional to D(x)^2 where D(x) is x's distance to its
// currently-nearest centroid.
//
// Reference: Arthur & Vassilvitskii, SODA 2007.
func initKMeansPlusPlus(sample []float32, centroids *[K][dataset.Dims]float32, rng *rand.Rand) {
	n := len(sample) / dataset.Dims

	first := rng.IntN(n)
	copyVec(centroids[0][:], sample[first*dataset.Dims:])

	d2 := make([]float64, n)
	for i := 0; i < n; i++ {
		d2[i] = sqDistF32(sample[i*dataset.Dims:i*dataset.Dims+dataset.Dims], centroids[0][:])
	}

	for k := 1; k < K; k++ {
		total := 0.0
		for _, v := range d2 {
			total += v
		}
		if total <= 0 {
			pick := rng.IntN(n)
			copyVec(centroids[k][:], sample[pick*dataset.Dims:])
			continue
		}
		r := rng.Float64() * total
		acc := 0.0
		pick := n - 1
		for i, v := range d2 {
			acc += v
			if acc >= r {
				pick = i
				break
			}
		}
		copyVec(centroids[k][:], sample[pick*dataset.Dims:])
		updateNearest(sample, &centroids[k], d2)
	}
}

// updateNearest refreshes d2[i] := min(d2[i], dist(sample[i], newCentroid)^2).
func updateNearest(sample []float32, newCentroid *[dataset.Dims]float32, d2 []float64) {
	n := len(d2)
	for i := 0; i < n; i++ {
		d := sqDistF32(sample[i*dataset.Dims:i*dataset.Dims+dataset.Dims], newCentroid[:])
		if d < d2[i] {
			d2[i] = d
		}
	}
}

// assignSample is one full pass of the Lloyd "assign" step over the sample:
// for each sample vector, find the nearest centroid and record the cluster
// id. Returns a per-sample uint16 (K=4096 fits in 16 bits).
func assignSample(sample []float32, centroids *[K][dataset.Dims]float32) []uint16 {
	n := len(sample) / dataset.Dims
	out := make([]uint16, n)
	for i := 0; i < n; i++ {
		v := sample[i*dataset.Dims : i*dataset.Dims+dataset.Dims]
		best := 0
		bestD := math.Inf(+1)
		for c := 0; c < K; c++ {
			d := sqDistF32(v, centroids[c][:])
			if d < bestD {
				bestD = d
				best = c
			}
		}
		out[i] = uint16(best)
	}
	return out
}

// updateCentroids is one full pass of the Lloyd "update" step: each centroid
// becomes the mean of the vectors assigned to it. Empty clusters keep their
// previous centroid (rare for k-means++ init; not worth re-seeding).
func updateCentroids(sample []float32, assign []uint16, centroids *[K][dataset.Dims]float32) {
	var sums [K][dataset.Dims]float64
	var counts [K]uint32

	n := len(assign)
	for i := 0; i < n; i++ {
		c := assign[i]
		base := i * dataset.Dims
		for d := 0; d < dataset.Dims; d++ {
			sums[c][d] += float64(sample[base+d])
		}
		counts[c]++
	}

	for c := 0; c < K; c++ {
		if counts[c] == 0 {
			continue
		}
		inv := 1.0 / float64(counts[c])
		for d := 0; d < dataset.Dims; d++ {
			centroids[c][d] = float32(sums[c][d] * inv)
		}
	}
}

// AssignAll buckets every reference vector by its nearest centroid. This is
// the single full-data pass we do *after* training; everything else uses the
// 65K sample. Returns a per-vector cluster id (K fits in u16).
func AssignAll(vectors []int16, count int, centroids *[K][dataset.Dims]float32) []uint16 {
	out := make([]uint16, count)
	var q [dataset.Dims]float32
	for i := 0; i < count; i++ {
		base := i * dataset.Stride
		for d := 0; d < dataset.Dims; d++ {
			q[d] = float32(vectors[base+d])
		}
		best := 0
		bestD := math.Inf(+1)
		for c := 0; c < K; c++ {
			d := sqDistF32(q[:], centroids[c][:])
			if d < bestD {
				bestD = d
				best = c
			}
		}
		out[i] = uint16(best)
	}
	return out
}

// Inertia is the within-cluster sum of squared distances. Decreases
// monotonically under Lloyd's. Used by tests to confirm convergence.
func Inertia(sample []float32, assign []uint16, centroids *[K][dataset.Dims]float32) float64 {
	n := len(assign)
	total := 0.0
	for i := 0; i < n; i++ {
		v := sample[i*dataset.Dims : i*dataset.Dims+dataset.Dims]
		c := assign[i]
		total += sqDistF32(v, centroids[c][:])
	}
	return total
}

func sqDistF32(a, b []float32) float64 {
	s := 0.0
	for d := 0; d < dataset.Dims; d++ {
		diff := float64(a[d]) - float64(b[d])
		s += diff * diff
	}
	return s
}

func copyVec(dst []float32, src []float32) {
	for d := 0; d < dataset.Dims; d++ {
		dst[d] = src[d]
	}
}
