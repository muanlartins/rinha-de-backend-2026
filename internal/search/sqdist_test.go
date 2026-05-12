//go:build amd64

package search

import (
	"math/rand/v2"
	"testing"

	"golang.org/x/sys/cpu"
)

// sqdistScalar mirrors the kernel inner loop on a pair of 16-int16 vectors,
// for the cross-check below.
func sqdistScalar(query, ref *[16]int16) int64 {
	var sum int64
	for d := 0; d < 16; d++ {
		t := int64(query[d]) - int64(ref[d])
		sum += t * t
	}
	return sum
}

// TestSqdistAVX2MatchesScalar feeds realistic inputs through both kernels.
// "Realistic" means: same range as the production data — values in [0, 32000]
// or both equal to the sentinel — so the per-lane diff stays inside int16
// and PMADDWL doesn't see overflow.
func TestSqdistAVX2MatchesScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not supported on this CPU")
	}
	rng := rand.New(rand.NewPCG(42, 84))
	var q, r [16]int16
	for trial := 0; trial < 10000; trial++ {
		for d := 0; d < 14; d++ {
			q[d] = int16(rng.IntN(32001))
			r[d] = int16(rng.IntN(32001))
		}
		// Padded dims always zero in both.
		q[14], q[15], r[14], r[15] = 0, 0, 0, 0

		got := sqdistAVX2(&q[0], &r[0])
		want := sqdistScalar(&q, &r)
		if got != want {
			t.Fatalf("trial %d: got=%d want=%d\n  q=%v\n  r=%v", trial, got, want, q, r)
		}
	}
}

// TestSqdistAVX2MaxDiff: every real dim hit its max diff (32000), with
// padded dims zeroed.
func TestSqdistAVX2MaxDiff(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not supported on this CPU")
	}
	var q, r [16]int16
	for i := 0; i < 14; i++ {
		q[i] = 32000
		r[i] = 0
	}
	want := int64(14) * 32000 * 32000
	got := sqdistAVX2(&q[0], &r[0])
	if got != want {
		t.Fatalf("got=%d want=%d", got, want)
	}
}
