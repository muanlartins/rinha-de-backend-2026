package api

import (
	"math/rand/v2"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
	"github.com/muanlartins/rinha-de-backend-2026/internal/search"
)

// Warmup runs `iters` synthetic IVF searches before the listener opens.
// Goal: warm L1/L2/L3 caches, the branch predictor, and bring the mmap'd
// index pages into RAM. With MADV_POPULATE_READ the pages are already
// resident, but the CPU caches are cold; warmup loads them with hot
// kernel + index data before the first real request.
//
// Uses synthetic int16 query vectors (no JSON parse) — the kernel and
// cluster sweep are what we want to warm, not the parser. Deterministic
// PCG seed so warmup is reproducible.
//
// See docs/lectures/13-mmap-madvise.md § What "warmup" does on top of mmap.
func Warmup(idx *ivf.IVFIndex, iters int) time.Duration {
	if idx == nil || iters <= 0 {
		return 0
	}
	t0 := time.Now()

	rng := rand.New(rand.NewPCG(0xC0FFEE, 0xCAFE))
	var scratch search.IVFScratch
	var qi [dataset.Dims]int16
	var qf [dataset.Dims]float32

	for i := 0; i < iters; i++ {
		for d := 0; d < dataset.Dims; d++ {
			// Distribute across the full int16 quantization range.
			qi[d] = int16(rng.IntN(int(dataset.QuantScale*2)) - int(dataset.QuantScale))
			qf[d] = float32(qi[d])
		}
		_ = search.FraudCountIVF(&qf, &qi, idx, &scratch)
	}

	return time.Since(t0)
}
