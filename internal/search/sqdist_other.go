//go:build !amd64

package search

// Non-amd64 builds have no SIMD path; we leave sqdistAVX2 undefined and force
// useAVX2 = false so the scalar kernel always runs.
var useAVX2 = false

// UseAVX2 reports whether the SIMD kernel is selected at runtime. Exposed
// for /debug/info.
func UseAVX2() bool { return useAVX2 }

func sqdistAVX2(query, ref *int16) int64 { panic("sqdistAVX2 called on non-amd64") }
