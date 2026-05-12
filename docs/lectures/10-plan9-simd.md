# Lecture 10 — Plan 9 SIMD: AVX2 + FMA in Go without cgo

This phase writes the inner distance kernel in assembly. The kernel computes 8 squared Euclidean distances in parallel against a query vector, with a 3-stage early-exit at dims 4, 6, and 8. It's the workhorse of the hot path.

Before any code, here's the model — what Plan 9 assembly is, why Go uses it, and how to write AVX2 + FMA instructions correctly without confusing the assembler.

## Why not cgo?

cgo gives us access to compiler intrinsics, but the per-call cost is on the order of 50-200 ns: argument marshalling, switching to a C stack, and back. Our kernel runs ~5000 times per query (one call per 8-vector block × ~600 blocks scanned per query). That's 250 µs to 1 ms of overhead per request, just from the cgo bridge. We'd never hit the 1 ms p99 target.

The Go calling convention for assembly functions is much cheaper: a direct branch with no stack switching. The cost is that we have to write the kernel in the Plan 9 dialect, which is more austere than Intel/AT&T syntax but no less expressive for our purposes.

## What "Plan 9 assembly" is

Go's assembler is a fork of the Plan 9 toolchain. The mnemonics are mostly identical to Intel's (`VMOVDQU`, `VFMADD231PS`, `VPSUBW`), but the **operand order is reversed** from Intel — the destination is on the right, like AT&T. There are also some syntactic differences:

- Registers are named without the `%` prefix: `AX`, `BX`, `Y0`, `Y15`.
- 64-bit immediates use the suffix `Q` (e.g. `MOVQ`, `PUSHQ`).
- 32-bit and 16-bit suffixes are `L` and `W` (`MOVL`, `MOVW`).
- The `SB` register holds the program counter for symbol references: `MOVQ ·symbol(SB), AX` loads the address of a Go-level symbol into AX.
- The `FP` register holds the frame pointer, used for argument access: `MOVQ q+0(FP), AX` loads the first argument.
- Frame headers look like `TEXT ·FuncName(SB), NOSPLIT, $localsize-argsize`. NOSPLIT tells the runtime "don't insert a stack-overflow check on entry" — safe for our kernel because it does no function calls and uses a fixed amount of stack.

## How Go calls assembly

When a Go function with signature `func F(args...) results...` is implemented in assembly, the Go compiler emits a stub that adapts the caller's register-ABI args (Go 1.17+) into stack-laid-out args for the assembly function. Inside the assembly, all arguments live in the caller's frame at fixed offsets, accessed via `FP`.

For our kernel:

```go
//go:noescape
func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) (anyAlive bool)
```

The frame layout, computed by the Go ABI rules (8-byte pointer alignment, 4-byte float32 alignment, return bool packed at the end):

| Name | Offset | Size |
|---|---:|---:|
| q | 0 | 8 |
| block | 8 | 8 |
| worst | 16 | 4 |
| sum | 24 | 8 |
| anyAlive | 32 | 1 |
| (padding) | 33-39 | 7 |

Total frame size: 40 bytes. We write `TEXT ·ScanBlock8AVX2(SB), NOSPLIT, $0-40`. The Go assembler verifies that `argsize` matches the declared signature; mismatch is a compile error.

The `//go:noescape` annotation tells the Go compiler that the function does not retain any of its pointer arguments after returning. This lets the compiler keep `sum` on the caller's stack rather than spilling it to the heap.

## The AVX2 instructions we need

We use a small subset:

| Mnemonic | Meaning |
|---|---|
| `VMOVDQU` | 256-bit unaligned move (memory <-> ymm or ymm <-> ymm). |
| `VBROADCASTSS` | Broadcast a single f32 from memory to all 8 lanes of a ymm. |
| `VPMOVSXWD` | Sign-extend 8 packed int16 (xmm = 128 bit) into 8 packed int32 (ymm = 256 bit). |
| `VCVTDQ2PS` | Convert 8 packed int32 (ymm) to 8 packed f32 (ymm). |
| `VSUBPS` | Per-lane f32 subtract: `dst = a - b`. |
| `VFMADD231PS` | Fused multiply-add: `dst = src1 * src2 + dst`. (The "231" refers to the operand encoding scheme.) |
| `VCMPPS` | Compare two ymms of 8 f32 lanes per a predicate immediate; produce an 8-bit-per-lane mask in a ymm. |
| `VMOVMSKPS` | Extract the high bit of each of the 8 f32 lanes from a ymm into an 8-bit value in a general-purpose register. |
| `VXORPS` | Bitwise XOR of two ymms — used to zero the accumulator. |

The early-exit pattern uses VCMPPS + VMOVMSKPS:

```
VCMPPS    $0x01, Y15, Y0, Y3       ; Y3[i] = -1 if Y0[i] < Y15[i] else 0
VMOVMSKPS Y3, R8                   ; R8 = 8-bit OR of "is lane i still alive"
TESTL     R8, R8
JZ        all_lanes_dead           ; if 0, no lane can still beat worst
```

The predicate immediate `$0x01` means "ordered less-than" — lane is alive iff its accumulated sum is strictly less than the broadcast `worst` value.

## The inner loop

The block has 14 × 8 = 112 int16 values laid out **dim-major**: 8 lanes of dim 0, then 8 lanes of dim 1, then ... then 8 lanes of dim 13. So `block[d*8 + lane]` is byte offset `d*16 + lane*2` (each int16 is 2 bytes).

Per dim, the work is:

```
; Load block dim d into an xmm half-register, sign-extend to int32 ymm
VPMOVSXWD   d*16(BX), Y1            ; Y1 = signed-extend(block[d*8..d*8+8])
VCVTDQ2PS   Y1, Y1                  ; Y1 = (f32)(int32 vector)
VBROADCASTSS d*4(AX), Y2            ; Y2 = {q[d]}x8
VSUBPS      Y1, Y2, Y2              ; Y2 = q[d] - lane
VFMADD231PS Y2, Y2, Y0              ; Y0 += Y2 * Y2
```

After dims 0..4 (5 dims accumulated), we do the first early-exit check. Then dims 5..6 (7 dims), then dim 7..8 (9 dims). After dim 8, we usually have enough partial sum to make the early-exit fire on dead lanes, but the bookkeeping cost is small enough that we always complete dims 9-13 once we've cleared the dim-8 gate.

The lecture 08 survey shows top submissions use checkpoints at dims 4, 6, and 8. We follow suit. The empirical reason: after 4 dims of squared accumulation, the f32 ranges are usually large enough that a "dead lane" diverges from the worst-of-top-5 by an unmistakable margin. Earlier checks would just burn cycles in the VCMPPS itself.

## Pitfalls and gotchas

A few things that bite the first time:

- **Plan 9 reversed operand order**. `VFMADD231PS Y2, Y1, Y0` does `Y0 = Y1 * Y2 + Y0`. The Intel manual writes the *same* instruction as `VFMADD231PS YMM0, YMM1, YMM2`. Reading Intel docs and translating naively will produce a kernel that computes the right multiplications with the wrong accumulator.

- **VPMOVSXWD wants a 128-bit memory operand** (xmm-sized) and produces a 256-bit ymm result. The memory side is 8 int16 = 16 bytes = 128 bits. The Plan 9 assembler accepts `VPMOVSXWD (BX), Y1` and figures out the source is xmm-sized from the destination width. No explicit memory-size cast required.

- **VBROADCASTSS wants a 32-bit memory operand or a 128-bit xmm register**. Either `VBROADCASTSS (AX), Y2` (broadcasts the float at *AX) or `VBROADCASTSS X3, Y2` works. The 8-bit "broadcast a register" only loads from xmm, not gpr.

- **VMOVMSKPS destination must be a general-purpose register**, not a ymm. The 8-bit mask is bit-packed into the low 8 bits of the GPR. Use a 32-bit GPR like `R8L` (Plan 9 syntax: `R8`). Subsequent `TESTL R8, R8 ; JZ early_exit` is the standard idiom.

- **Plan 9 immediate prefix is `$`**, both for instruction immediates and for symbolic constants. `VCMPPS $0x01, Y15, Y0, Y3`.

- **Function frame size**. If `$0-N` and `N` is wrong, `go vet` flags the function as "frame size mismatch with signature". We declare `$0-40` matching the layout above.

- **NOSPLIT** is required if the function uses no extra stack space and makes no Go-runtime calls. Without it, the runtime inserts a stack-overflow check on entry; in our case that check is wasted work because we use 0 bytes of local stack.

## Cross-checking against the generic fallback

For darwin/arm64 (our development machine) the `_amd64.s` file is excluded from the build by file-name convention. A `dist_generic.go` file provides the same `ScanBlock8AVX2` function in pure Go for any architecture. The generic implementation is the oracle.

The cross-check test:

```go
func TestScanBlock8AVX2MatchesGeneric(t *testing.T) {
    if !cpu.X86.HasAVX2 { t.Skip(...) }
    for trial := 0; trial < 1000; trial++ {
        // generate random q, block, worst
        // ...
        var sumAsm, sumGen [8]float32
        aliveAsm := ScanBlock8AVX2(&q[0], &block[0], worst, &sumAsm)
        aliveGen := ScanBlock8Generic(&q[0], &block[0], worst, &sumGen)
        // Compare sums up to f32 tolerance, compare alive flags exactly.
    }
}
```

The tolerance check accounts for FMA's single-rounding semantics: a fused `a*b + c` produces a slightly different last bit than separate `a*b` followed by addition. The generic Go path does separate multiplication and addition; the FMA assembly path fuses them. Empirically, the relative error stays under `1e-3` on our data ranges, and we test against that bound.

The test is skipped automatically on non-amd64 platforms, but is part of the CI suite that runs in Docker against linux/amd64.

## What we are *not* doing

- **VPMADDWD with int16 arithmetic.** A common pattern in image-processing kernels: stay in int16, multiply pairs of int16 lanes into int32 lanes with VPMADDWD, accumulate in int32. Saves the f32 conversion. We don't do it because the contest's exact-match requirement makes f32 accumulation the safe choice — int32 accumulators would need rescaling logic to compare to the worst-of-top-5 (which is naturally f32).

- **AVX-512.** The contest hardware is Haswell, which has AVX2 but not AVX-512. We rely on `GOAMD64=v3` to guarantee the v3 baseline (AVX2 + FMA + BMI2) is available.

- **Multi-block batching.** We compute one block of 8 vectors per call. A multi-block kernel (e.g. 32 vectors per call) would amortize the function-call overhead, but the per-call overhead in Go's asm ABI is already tiny (~3 ns). Profile says it's not where time goes.

## Code layout

The implementation in this phase:

- `internal/kernel/dist_amd64.s` — the AVX2 + FMA kernel.
- `internal/kernel/dist_amd64.go` — the function declaration with `//go:noescape`.
- `internal/kernel/dist_generic.go` — scalar fallback for non-amd64.
- `internal/kernel/dist_test.go` — cross-check + benchmarks.

The kernel is used by phase 16's search code. Phase 15 stops here — we don't wire it to the request path yet.

## References

- Go assembler doc: https://go.dev/doc/asm
- Intel Intrinsics Guide (mnemonic semantics): https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Plan 9 mnemonics table in Go source: `src/cmd/asm/internal/arch/avx_optabs.go`
- Lecture 08 § 3 for the layout rationale.
