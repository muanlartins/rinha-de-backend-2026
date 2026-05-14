package search

import (
	"math/rand/v2"
	"testing"
)

// BenchmarkPickNextNUnscanned measures the escalation picker — top-N
// over K=4096 with one cluster masked out by the scanned bitmap.
// Drives Phase 34's decision on whether to asm-ify this path.
func BenchmarkPickNextNUnscanned(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	const K = 4096
	dists := make([]float32, K)
	for i := range dists {
		dists[i] = float32(rng.IntN(2000000)) - 1000000
	}
	scanned := make([]uint64, K/64)
	// Simulate fast-tier scanned: 1 cluster marked.
	scanned[7] |= 1 << 13

	picked := make([]uint16, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PickNextNUnscanned(dists, scanned, 32, picked)
	}
}
