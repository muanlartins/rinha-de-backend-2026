//go:build !amd64

package search

// Non-amd64 builds have no SIMD path; we leave sqdistAVX2 undefined and force
// useAVX2 = false so the scalar kernel always runs.
var useAVX2 = false

func sqdistAVX2(query, ref *int16) int64 { panic("sqdistAVX2 called on non-amd64") }
