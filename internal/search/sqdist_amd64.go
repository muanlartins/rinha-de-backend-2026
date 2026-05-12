//go:build amd64

package search

import "golang.org/x/sys/cpu"

// sqdistAVX2 computes the squared Euclidean distance between two 16-int16
// vectors using AVX2 (VPSUBW + VPMADDWD + horizontal sum). Implementation in
// sqdist_amd64.s.
func sqdistAVX2(query, ref *int16) int64

// useAVX2 is set at init by detecting CPU feature support. On the Rinha test
// env (Mac Mini Late 2014 = Haswell) this is always true; on Rosetta 2 it
// depends on macOS version.
var useAVX2 = cpu.X86.HasAVX2
