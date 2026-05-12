//go:build amd64

package search

import (
	"math/rand/v2"
	"testing"

	"golang.org/x/sys/cpu"
)

// blockScalar: reference squared-distance for 8 vectors stored in dim-major
// block layout, matching the asm semantics.
func blockScalar(query *[16]int16, block *[16 * 8]int16, out *[8]int64) {
	for v := 0; v < 8; v++ {
		var sum int64
		for d := 0; d < 16; d++ {
			t := int64(query[d]) - int64(block[d*8+v])
			sum += t * t
		}
		out[v] = sum
	}
}

func TestBlockScan8MatchesScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not supported")
	}
	rng := rand.New(rand.NewPCG(11, 22))
	var q [16]int16
	var blk [16 * 8]int16
	for trial := 0; trial < 5000; trial++ {
		for d := 0; d < 14; d++ {
			q[d] = int16(rng.IntN(32001))
		}
		q[14], q[15] = 0, 0
		for d := 0; d < 14; d++ {
			for v := 0; v < 8; v++ {
				blk[d*8+v] = int16(rng.IntN(32001))
			}
		}
		for v := 0; v < 8; v++ {
			blk[14*8+v] = 0
			blk[15*8+v] = 0
		}

		var simd, scalar [8]int64
		blockScan8AVX2(&q[0], &blk[0], &simd)
		blockScalar(&q, &blk, &scalar)
		for v := 0; v < 8; v++ {
			if simd[v] != scalar[v] {
				t.Fatalf("trial %d v %d: simd=%d scalar=%d", trial, v, simd[v], scalar[v])
			}
		}
	}
}
