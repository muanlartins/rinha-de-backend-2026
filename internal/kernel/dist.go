// Package kernel holds the SIMD distance kernels used by the IVF search
// hot path. The amd64 build provides AVX2 + FMA assembly; other architectures
// get a pure-Go fallback used only for development on Apple Silicon.
//
// All public functions accept layout-specific pointers, not slices, so the
// caller can avoid the per-call slice-header construction overhead.
//
// Layout assumptions (see docs/lectures/10-plan9-simd.md):
//
//   - q points to 14 contiguous float32 — the query, dequantized.
//   - block points to 14*8 contiguous int16 — one block of 8 reference
//     vectors, dim-major within the block: block[d*8 + lane] is dim d of
//     reference lane `lane` of this block.
//   - sum is filled with the 8 squared-distance lane sums on return.
//   - worst is broadcast across the 8 lanes for the early-exit check.
//
// The return value is true if at least one lane is still strictly less than
// worst after the kernel runs. The caller uses this to decide whether to
// insert any of the lanes into the running top-5 heap.
package kernel
