package search

import (
	"math/rand/v2"
	"sort"
	"testing"
)

// pickNextNUnscannedReference is the pre-phase-34 sorted-insertion-array
// implementation, kept here as the parity oracle.
func pickNextNUnscannedReference(dists []float32, scanned []uint64, n int, picked []uint16) {
	if n > len(picked) {
		n = len(picked)
	}
	for i := 0; i < n; i++ {
		picked[i] = ^uint16(0)
	}
	worst := float32(1e38)

	for c := 0; c < len(dists); c++ {
		if scanned[c/64]&(1<<(uint(c)%64)) != 0 {
			continue
		}
		d := dists[c]
		if d >= worst {
			continue
		}
		var pos int
		for pos = 0; pos < n; pos++ {
			id := picked[pos]
			if id == ^uint16(0) || dists[id] > d {
				break
			}
		}
		if pos == n {
			continue
		}
		for i := n - 1; i > pos; i-- {
			picked[i] = picked[i-1]
		}
		picked[pos] = uint16(c)

		worstID := picked[n-1]
		if worstID == ^uint16(0) {
			worst = 1e38
		} else {
			worst = dists[worstID]
		}
	}
}

func TestPickNextNUnscannedHeapParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xdeadbeef, 0xfeedf00d))
	const K = 4096
	const N = 32

	for trial := 0; trial < 30; trial++ {
		dists := make([]float32, K)
		for i := range dists {
			dists[i] = float32(rng.IntN(2000000)) - 1000000
		}
		scanned := make([]uint64, K/64)
		// Randomly mark 1-3 clusters as scanned.
		marks := 1 + rng.IntN(3)
		for m := 0; m < marks; m++ {
			c := rng.IntN(K)
			scanned[c/64] |= 1 << uint(c%64)
		}

		gotPicked := make([]uint16, N)
		wantPicked := make([]uint16, N)
		PickNextNUnscanned(dists, scanned, N, gotPicked)
		pickNextNUnscannedReference(dists, scanned, N, wantPicked)

		// Both must contain the same set of indices (order may differ if
		// there are ties, but ties on f32 random distances are
		// vanishingly rare). To be safe, compare by SORTED distance
		// values rather than raw indices.
		gotDs := make([]float32, 0, N)
		wantDs := make([]float32, 0, N)
		for _, c := range gotPicked {
			if c != ^uint16(0) {
				gotDs = append(gotDs, dists[c])
			}
		}
		for _, c := range wantPicked {
			if c != ^uint16(0) {
				wantDs = append(wantDs, dists[c])
			}
		}
		sort.Slice(gotDs, func(i, j int) bool { return gotDs[i] < gotDs[j] })
		sort.Slice(wantDs, func(i, j int) bool { return wantDs[i] < wantDs[j] })
		if len(gotDs) != len(wantDs) {
			t.Fatalf("trial %d: len mismatch got=%d want=%d", trial, len(gotDs), len(wantDs))
		}
		for i := range gotDs {
			if gotDs[i] != wantDs[i] {
				t.Errorf("trial %d: index %d: got dist %f, want %f", trial, i, gotDs[i], wantDs[i])
			}
		}
	}
}

func BenchmarkPickNextNUnscannedReference(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	const K = 4096
	dists := make([]float32, K)
	for i := range dists {
		dists[i] = float32(rng.IntN(2000000)) - 1000000
	}
	scanned := make([]uint64, K/64)
	scanned[7] |= 1 << 13
	picked := make([]uint16, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pickNextNUnscannedReference(dists, scanned, 32, picked)
	}
}
