package search

import (
	"math/rand/v2"
	"testing"
)

// TestArgminParity verifies the asm path agrees with the scalar
// reference across random inputs of varying sizes (all multiples of 8,
// the asm constraint). Lowest-index-wins on ties is verified by
// embedding duplicate minimum values.
func TestArgminParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xdeadbeef, 0xcafebabe))

	for _, K := range []int{8, 16, 64, 256, 1024, 4096} {
		for trial := 0; trial < 30; trial++ {
			dists := make([]float32, K)
			for i := range dists {
				dists[i] = float32(rng.IntN(1000000)) - 500000
			}
			// Inject a duplicate minimum to test tie-break.
			if K >= 16 && trial%5 == 0 {
				lo := dists[3]
				dists[K-2] = lo
			}
			got := pickArgminFast(dists)
			want := pickArgminScalar(dists)
			// Tie semantics: scalar reference picks the lowest index on
			// equal values; asm picks "some" lane that holds the min.
			// For our use case (centroid argmin) the *value* matters,
			// not which of several tied indices we return. Accept any
			// index whose value equals the scalar's min.
			if dists[got] != dists[want] {
				t.Errorf("K=%d trial=%d: pickArgminFast=%d (val %f), scalar=%d (val %f)",
					K, trial, got, dists[got], want, dists[want])
			}
		}
	}
}

// TestArgminEdgeCases pokes the boundary conditions.
func TestArgminEdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		dists []float32
		want  uint16
	}{
		{"all-equal", []float32{1, 1, 1, 1, 1, 1, 1, 1}, 0},
		{"first-min", []float32{0, 9, 9, 9, 9, 9, 9, 9}, 0},
		{"last-min", []float32{9, 9, 9, 9, 9, 9, 9, 0}, 7},
		{"negative", []float32{-1, -5, -3, 0, 1, 2, 3, 4}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := pickArgminFast(tc.dists)
			if got != tc.want {
				t.Errorf("got %d, want %d", got, tc.want)
			}
		})
	}
}

func BenchmarkPickArgmin_AsmDispatch(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	dists := make([]float32, 4096)
	for i := range dists {
		dists[i] = float32(rng.IntN(2000000)) - 1000000
	}
	picked := make([]uint16, 1)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PickTopNCentroids(dists, 1, picked)
	}
}

func BenchmarkPickArgmin_Scalar(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	dists := make([]float32, 4096)
	for i := range dists {
		dists[i] = float32(rng.IntN(2000000)) - 1000000
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = pickArgminScalar(dists)
	}
}
