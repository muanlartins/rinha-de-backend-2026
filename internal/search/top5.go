package search

// Top5 holds the 5 nearest neighbors by squared distance, stored as int64
// to preserve the exact integer arithmetic of int16-quantized vectors.
// The asm kernel runs in f32 (faster), then the caller reranks alive lanes
// with exact i64 distance — see lecture 11 § The cascading top-5 insert.
//
// dist[0] is the closest, dist[4] is the worst (largest distance among the
// 5). The struct is small enough to keep on the stack across a query.
type Top5 struct {
	dist  [5]int64
	label [5]uint8
}

// kernelSafety is added to the f32 worst before being passed to the AVX2
// kernel and to the AABB-LB check. The f32 partial sums can be off by up
// to ~14 * 3600 ≈ 50K at magnitudes of 3e10 (worst case for 14 squared
// i16 diffs where each diff is up to 64000). 64K covers it with room.
// See lecture 11 § f32 vs i64 precision.
const kernelSafety = 65536

// Reset sets all distances to the largest possible int64 so the first 5
// candidates fill the heap.
func (t *Top5) Reset() {
	for i := 0; i < 5; i++ {
		t.dist[i] = 1 << 62
		t.label[i] = 0
	}
}

// WorstI64 returns the largest of the 5 current distances (i64).
func (t *Top5) WorstI64() int64 { return t.dist[4] }

// WorstF32 is the f32 threshold the kernel uses for its early-exit check.
// We add kernelSafety so a candidate whose true i64 distance is within
// rounding error of WorstI64 is not pruned by the kernel.
func (t *Top5) WorstF32() float32 {
	return float32(t.dist[4]) + kernelSafety
}

// InsertI64 places (d, l) into the sorted-ascending top-5 if d < WorstI64.
// Linear scan + shift. ~10-15 ns per call in practice.
func (t *Top5) InsertI64(d int64, l uint8) {
	if d >= t.dist[4] {
		return
	}
	pos := 4
	for pos > 0 && t.dist[pos-1] > d {
		t.dist[pos] = t.dist[pos-1]
		t.label[pos] = t.label[pos-1]
		pos--
	}
	t.dist[pos] = d
	t.label[pos] = l
}

// FraudCount returns how many of the 5 labels are 1 (fraud).
func (t *Top5) FraudCount() uint8 {
	return t.label[0] + t.label[1] + t.label[2] + t.label[3] + t.label[4]
}
