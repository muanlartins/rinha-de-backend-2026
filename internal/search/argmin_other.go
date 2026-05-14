//go:build !amd64

package search

// pickArgminFast is the non-amd64 fallback — uses the scalar argmin.
func pickArgminFast(dists []float32) uint16 { return pickArgminScalar(dists) }
