//go:build amd64

#include "textflag.h"

// func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) (anyAlive bool)
//
// Frame layout (40 bytes total):
//   q          +0   (*float32, 8B)
//   block      +8   (*int16, 8B)
//   worst      +16  (float32, 4B)  -- pad bytes 20-23 for ptr alignment
//   sum        +24  (*[8]float32, 8B)
//   anyAlive   +32  (bool, 1B)     -- pad to 40 for arg-frame alignment
//
// Computes 8 squared Euclidean distances between q (14 f32) and 8 ref
// vectors stored dim-major in block (14*8 int16). Early-exit gate at
// dim 8 cumulative against worst; final check after dim 14.
//
// Phase 38: split FMA dependency chain into two parallel accumulators.
//
//   Y0 = even-dim accumulator (dims 0, 2, 4, 6, 8, 10, 12)
//   Y4 = odd-dim accumulator  (dims 1, 3, 5, 7, 9, 11, 13)
//
// Per-dim work is unchanged (14 dims × 5 ops = 70 ops total), but the
// critical path is halved: instead of 14 FMA-latency-5 hops serialized
// into one accumulator (70 cycles on Haswell), each chain is 7 hops
// (35 cycles), and they run in parallel on independent pipelines.
//
// At each checkpoint we merge Y4 → Y0 for the threshold compare. The
// merge is a single VADDPS (3 cycles, throughput 0.5), trivial.
//
// Register usage:
//   AX = &q[0]
//   BX = &block[0]
//   CX = &sum[0]
//   R8 = alive mask (low 8 bits)
//   Y0 = squared-distance accumulator, even dims
//   Y4 = squared-distance accumulator, odd dims
//   Y1 = scratch (loaded block dim, converted to f32)
//   Y2 = scratch (broadcast query dim, then diff)
//   Y3 = scratch (cmp mask) / merged total at checkpoint
//   Y15 = broadcast worst across 8 lanes

TEXT ·ScanBlock8AVX2(SB), NOSPLIT, $0-40
    MOVQ         q+0(FP), AX
    MOVQ         block+8(FP), BX
    VBROADCASTSS worst+16(FP), Y15
    MOVQ         sum+24(FP), CX

    // HW prefetch the next block (offset +224) so it lands in L1d while
    // this iteration's FMAs are in flight. Three lines cover the 192
    // bytes that aren't already hot from the current block's tail line
    // [192..256). PREFETCHT0 is a hint — safe on invalid addresses at
    // end-of-cluster / end-of-mmap.
    PREFETCHT0 256(BX)
    PREFETCHT0 320(BX)
    PREFETCHT0 384(BX)

    VXORPS Y0, Y0, Y0     // even-dim acc
    VXORPS Y4, Y4, Y4     // odd-dim acc

    // --- Dims 0..3 ---------------------------------------------------
    // dim 0 → Y0
    VPMOVSXWD    (BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS (AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y0

    // dim 1 → Y4
    VPMOVSXWD    16(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 4(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y4

    // dim 2 → Y0
    VPMOVSXWD    32(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 8(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y0

    // dim 3 → Y4
    VPMOVSXWD    48(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 12(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y4

    // --- Dims 4..7 ---------------------------------------------------
    // dim 4 → Y0
    VPMOVSXWD    64(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 16(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y0

    // dim 5 → Y4
    VPMOVSXWD    80(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 20(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y4

    // dim 6 → Y0
    VPMOVSXWD    96(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 24(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y0

    // dim 7 → Y4
    VPMOVSXWD    112(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 28(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y4

    // --- Checkpoint at 8 dims accumulated -----------------------------
    // Merge Y4 → Y0 to get the cumulative 8-dim sum, then compare.
    VADDPS    Y4, Y0, Y3
    VCMPPS    $0x01, Y15, Y3, Y3
    VMOVMSKPS Y3, R8
    TESTL     R8, R8
    JZ        dead

    // --- Dims 8..13 ---------------------------------------------------
    // dim 8 → Y0
    VPMOVSXWD    128(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 32(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y0

    // dim 9 → Y4
    VPMOVSXWD    144(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 36(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y4

    // dim 10 → Y0
    VPMOVSXWD    160(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 40(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y0

    // dim 11 → Y4
    VPMOVSXWD    176(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 44(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y4

    // dim 12 → Y0
    VPMOVSXWD    192(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 48(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y0

    // dim 13 → Y4
    VPMOVSXWD    208(BX), Y1
    VCVTDQ2PS    Y1, Y1
    VBROADCASTSS 52(AX), Y2
    VSUBPS       Y1, Y2, Y2
    VFMADD231PS  Y2, Y2, Y4

    // --- Final merge + liveness --------------------------------------
    // Combine both accs into Y0 for the final compare and store.
    VADDPS    Y4, Y0, Y0
    VCMPPS    $0x01, Y15, Y0, Y3
    VMOVMSKPS Y3, R8
    TESTL     R8, R8
    JZ        dead

    VMOVUPS Y0, (CX)
    MOVB    $1, anyAlive+32(FP)
    VZEROUPPER
    RET

dead:
    // On the dead path, Y0 may not have the merged total if we
    // exited at the dim-8 checkpoint. The Top5 cascading-insert path
    // only uses BlockSum on the alive path, but we still write
    // *something* to BlockSum so the contract that anyAlive=false
    // means "don't read sum" is preserved on alive=true. For
    // robustness we merge here too.
    VADDPS  Y4, Y0, Y0
    VMOVUPS Y0, (CX)
    MOVB    $0, anyAlive+32(FP)
    VZEROUPPER
    RET
