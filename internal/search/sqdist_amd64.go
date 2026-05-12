//go:build amd64

package search

import "golang.org/x/sys/cpu"

// sqdistAVX2 computes the squared Euclidean distance between two 16-int16
// vectors using AVX2 (VPSUBW + VPMADDWD + horizontal sum). Implementation in
// sqdist_amd64.s.
func sqdistAVX2(query, ref *int16) int64

// blockScan8AVX2 computes 8 squared distances from one query to 8 reference
// vectors stored in dim-major block layout (16 dims × 8 vectors). Output is
// 8 int64 distances. Implementation in blockdist_amd64.s.
func blockScan8AVX2(query, block *int16, out *[8]int64)

// useAVX2 is set at init by detecting CPU feature support. On the Rinha test
// env (Mac Mini Late 2014 = Haswell) this is always true; on Rosetta 2 it
// depends on macOS version.
var useAVX2 = cpu.X86.HasAVX2

// UseAVX2 reports whether the SIMD kernel is selected at runtime. Exposed
// for /debug/info.
func UseAVX2() bool { return useAVX2 }
