//go:build amd64

package search

// argminAVX2 is the asm fast path. K must be a positive multiple of 8
// (production K=4096; tests cover smaller values). Returns the index
// of the smallest dists[i], with lowest-index winning on ties.
//
//go:noescape
func argminAVX2(dists *float32, K int) uint16

// pickArgminFast is the platform-specific dispatch used by
// PickTopNCentroids when n == 1. On amd64 it goes through AVX2 asm;
// on other architectures (see argmin_other.go) it falls back to the
// scalar path.
func pickArgminFast(dists []float32) uint16 {
	if len(dists)&7 != 0 || len(dists) == 0 {
		// asm requires K = positive multiple of 8 (production K=4096
		// always satisfies). For other lengths, fall back to scalar.
		return pickArgminScalar(dists)
	}
	return argminAVX2(&dists[0], len(dists))
}
