package search

import (
	"math/rand/v2"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// TestScoreAllCentroidsAsmParity compares the AVX2 asm path (production
// path on linux/amd64 and darwin/amd64 via Rosetta) against the pure-Go
// reference fold. Any divergence indicates a bug in the asm.
//
// Tolerance: f32 sums are order-dependent; we allow up to 1e-3 relative
// error per centroid. The asm uses the same FMA op as the autovec
// generic, so the result should be bit-identical for most centroids,
// with rare differences due to compiler-driven reordering in the
// generic path.
func TestScoreAllCentroidsAsmParity(t *testing.T) {
	const K = 4096
	rng := rand.New(rand.NewPCG(0xfeedf00d, 0xdeadbeef))

	centroids := make([]float32, dataset.Dims*K)
	for i := range centroids {
		centroids[i] = float32(rng.IntN(20000) - 10000)
	}

	var q [dataset.Dims]float32
	wantOut := make([]float32, K)
	gotOut := make([]float32, K)

	for trial := 0; trial < 50; trial++ {
		for d := 0; d < dataset.Dims; d++ {
			q[d] = float32(rng.IntN(20000) - 10000)
		}
		scoreAllCentroidsGeneric(&q, centroids, K, wantOut)
		ScoreAllCentroids(&q, centroids, K, gotOut)

		var maxRel float32
		for c := 0; c < K; c++ {
			w := wantOut[c]
			g := gotOut[c]
			if w == 0 && g == 0 {
				continue
			}
			diff := g - w
			if diff < 0 {
				diff = -diff
			}
			ref := w
			if ref < 0 {
				ref = -ref
			}
			rel := diff / ref
			if rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > 1e-3 {
			t.Errorf("trial %d: max relative error %g exceeds tolerance", trial, maxRel)
		}
	}
}

// BenchmarkScoreAllCentroids measures the per-call cost — the universal
// path runs this on every query. Useful to compare asm vs generic.
func BenchmarkScoreAllCentroids(b *testing.B) {
	const K = 4096
	rng := rand.New(rand.NewPCG(1, 2))
	centroids := make([]float32, dataset.Dims*K)
	for i := range centroids {
		centroids[i] = float32(rng.IntN(20000) - 10000)
	}
	var q [dataset.Dims]float32
	for d := 0; d < dataset.Dims; d++ {
		q[d] = float32(rng.IntN(20000) - 10000)
	}
	out := make([]float32, K)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScoreAllCentroids(&q, centroids, K, out)
	}
}

func BenchmarkScoreAllCentroidsGeneric(b *testing.B) {
	const K = 4096
	rng := rand.New(rand.NewPCG(1, 2))
	centroids := make([]float32, dataset.Dims*K)
	for i := range centroids {
		centroids[i] = float32(rng.IntN(20000) - 10000)
	}
	var q [dataset.Dims]float32
	for d := 0; d < dataset.Dims; d++ {
		q[d] = float32(rng.IntN(20000) - 10000)
	}
	out := make([]float32, K)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scoreAllCentroidsGeneric(&q, centroids, K, out)
	}
}
