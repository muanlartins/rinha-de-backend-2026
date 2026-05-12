package kernel

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestScanBlock8AVX2MatchesGeneric is the cross-check. On amd64 it runs the
// AVX2 + FMA kernel against the pure-Go reference for 1000 random
// (q, block, worst) tuples; on other archs both sides are the generic
// fallback, so the test passes trivially but at least confirms the
// generic kernel is internally consistent.
func TestScanBlock8AVX2MatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 13))

	const lanes = 8
	const dims = 14

	for trial := 0; trial < 1000; trial++ {
		var q [dims]float32
		var block [dims * lanes]int16
		for d := 0; d < dims; d++ {
			q[d] = float32(rng.IntN(32000))
		}
		for i := range block {
			block[i] = int16(rng.IntN(32000))
		}
		// Set worst to something a bit above the median lane distance so
		// some trials trigger early-exit and others don't.
		var probe [lanes]float32
		_ = ScanBlock8Generic(&q[0], &block[0], math.MaxFloat32, &probe)
		median := probe[lanes/2]
		worst := median * (0.5 + rng.Float32()*1.5)

		var sumAsm, sumGen [lanes]float32
		aliveAsm := ScanBlock8AVX2(&q[0], &block[0], worst, &sumAsm)
		aliveGen := ScanBlock8Generic(&q[0], &block[0], worst, &sumGen)

		// If both kernels decided everyone is dead at some checkpoint, the
		// partial sums may differ at the still-untouched dims because the
		// asm kernel may have computed one more dim past the cutoff before
		// it tested. We still expect aliveAsm == aliveGen.
		if aliveAsm != aliveGen {
			t.Fatalf("trial %d: aliveAsm=%v aliveGen=%v worst=%g",
				trial, aliveAsm, aliveGen, worst)
		}

		if aliveAsm {
			// Lanes alive (i.e., still below worst): both sums must match
			// to f32 tolerance.
			for lane := 0; lane < lanes; lane++ {
				if sumAsm[lane] >= worst && sumGen[lane] >= worst {
					continue // dead lane; partial sum was not retained
				}
				rel := relErr(sumAsm[lane], sumGen[lane])
				if rel > 1e-3 {
					t.Fatalf("trial %d lane %d: sumAsm=%g sumGen=%g (rel=%g)",
						trial, lane, sumAsm[lane], sumGen[lane], rel)
				}
			}
		}
	}
}

// TestScanBlock8EarlyExitSemantics confirms that when worst is below the
// minimum possible accumulated distance after 4 dims, the kernel reports
// not-alive and the caller knows not to insert.
func TestScanBlock8EarlyExitSemantics(t *testing.T) {
	const lanes = 8
	const dims = 14
	var q [dims]float32
	var block [dims * lanes]int16
	for d := 0; d < dims; d++ {
		q[d] = 0
		for lane := 0; lane < lanes; lane++ {
			block[d*lanes+lane] = 30000 // huge diff vs q=0
		}
	}

	worst := float32(1.0) // way below the actual squared distance
	var sum [lanes]float32
	alive := ScanBlock8AVX2(&q[0], &block[0], worst, &sum)
	if alive {
		t.Fatalf("expected all-dead, got alive=%v sum=%v", alive, sum)
	}
}

// TestScanBlock8AllAlive: when worst is huge, the kernel must complete all
// 14 dims and report alive.
func TestScanBlock8AllAlive(t *testing.T) {
	const lanes = 8
	const dims = 14
	var q [dims]float32
	var block [dims * lanes]int16
	for d := 0; d < dims; d++ {
		q[d] = 10
		for lane := 0; lane < lanes; lane++ {
			block[d*lanes+lane] = int16(20 + lane)
		}
	}

	var sum [lanes]float32
	alive := ScanBlock8AVX2(&q[0], &block[0], math.MaxFloat32, &sum)
	if !alive {
		t.Fatal("expected alive=true with huge worst")
	}

	// Spot-check lane 0: (10-20)^2 * 14 = 1400
	want := float32((10 - 20) * (10 - 20) * dims)
	if math.Abs(float64(sum[0]-want)) > 1e-3 {
		t.Fatalf("lane 0: got %g, want ~%g", sum[0], want)
	}
}

func relErr(a, b float32) float64 {
	if a == b {
		return 0
	}
	denom := math.Abs(float64(a)) + math.Abs(float64(b))
	if denom < 1e-12 {
		return 0
	}
	return 2 * math.Abs(float64(a)-float64(b)) / denom
}

// BenchmarkScanBlock8AVX2 and BenchmarkScanBlock8Generic measure the
// per-call cost of the two kernels. Useful when tuning the early-exit
// strategy in phase 16.
func BenchmarkScanBlock8AVX2(b *testing.B) {
	const lanes = 8
	const dims = 14
	rng := rand.New(rand.NewPCG(1, 2))
	var q [dims]float32
	var block [dims * lanes]int16
	for d := 0; d < dims; d++ {
		q[d] = float32(rng.IntN(32000))
	}
	for i := range block {
		block[i] = int16(rng.IntN(32000))
	}
	var sum [lanes]float32

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScanBlock8AVX2(&q[0], &block[0], math.MaxFloat32, &sum)
	}
}

func BenchmarkScanBlock8Generic(b *testing.B) {
	const lanes = 8
	const dims = 14
	rng := rand.New(rand.NewPCG(1, 2))
	var q [dims]float32
	var block [dims * lanes]int16
	for d := 0; d < dims; d++ {
		q[d] = float32(rng.IntN(32000))
	}
	for i := range block {
		block[i] = int16(rng.IntN(32000))
	}
	var sum [lanes]float32

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ScanBlock8Generic(&q[0], &block[0], math.MaxFloat32, &sum)
	}
}
