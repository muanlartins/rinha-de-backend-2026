package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// AABBLowerBoundF32 returns the lower bound on the squared distance from q
// to any point inside the axis-aligned bounding box [bmin, bmax]. See
// lecture 11 § Why AABB-LB is sound for the proof.
//
// q is the float32 dequantized query. bmin and bmax are 16-wide int16
// (lanes 14-15 are zero-padded in the index but only lanes 0..13 are
// touched here).
//
// We tested an AVX2 asm version (Phase 39); it ran ~50% slower than the
// scalar Go (18.7 ns vs 12.05 ns on amd64 Docker) because the Go
// compiler's autovectorizer produces excellent code for this branchy
// switch-loop, and the asm pays an unavoidable horizontal-sum reduction
// at the end. Kept the scalar version.
func AABBLowerBoundF32(q *[dataset.Dims]float32, bmin, bmax *[16]int16) float32 {
	var sum float32
	for d := 0; d < dataset.Dims; d++ {
		qd := q[d]
		bnLo := float32(bmin[d])
		bnHi := float32(bmax[d])
		var t float32
		switch {
		case qd > bnHi:
			t = qd - bnHi
		case qd < bnLo:
			t = bnLo - qd
		default:
			t = 0
		}
		sum += t * t
	}
	return sum
}
