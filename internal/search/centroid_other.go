//go:build !amd64

package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// ScoreAllCentroids on non-amd64 builds falls through to the pure-Go
// reference implementation.
func ScoreAllCentroids(q *[dataset.Dims]float32, centroids []float32, K int, out []float32) {
	scoreAllCentroidsGeneric(q, centroids, K, out)
}
