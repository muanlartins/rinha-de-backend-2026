//go:build amd64

package search

import (
	"unsafe"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// scoreAllCentroidsAVX2 is the asm implementation of ScoreAllCentroids
// for amd64 (Phase 28). See centroid_amd64.s. K must be a positive
// multiple of 8 (we have K=4096 always; the asm has no tail handler).
//
//go:noescape
func scoreAllCentroidsAVX2(q *float32, centroids *float32, K int, out *float32)

// ScoreAllCentroids dispatches to the AVX2 asm path on amd64. The query
// argument is a fixed-size 14-dim array (the runtime always supplies 14
// dims); centroids must be at least K*14 long, and out at least K.
func ScoreAllCentroids(q *[dataset.Dims]float32, centroids []float32, K int, out []float32) {
	if len(out) < K {
		panic("out buffer too small")
	}
	scoreAllCentroidsAVX2(&q[0], (*float32)(unsafe.Pointer(&centroids[0])), K, &out[0])
}
