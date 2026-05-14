//go:build amd64

#include "textflag.h"

// func scoreAllCentroidsAVX2(q *float32, centroids *float32, K int, out *float32)
//
// Equivalent to:
//   for c := 0; c < K; c++ { out[c] = 0 }
//   for d := 0; d < 14; d++ {
//       qd := q[d]
//       base := d * K
//       for c := 0; c < K; c++ {
//           diff := qd - centroids[base+c]
//           out[c] += diff * diff
//       }
//   }
//
// Loop order: outer over d, inner over c. This is cache-friendly because
// for each dim we stream through K=4096 sequential float32s (16 KB per
// dim, fits in L1d). The accumulator out[] is also streamed sequentially.
//
// A version with outer-c, inner-d was tried but ran 3× slower because
// touching all 14 dim slabs in sequence (14 × K_stride = 14 × 16 KB =
// 224 KB) thrashes L1 cache.
//
// The hand-coded asm wins vs the autovec generic only marginally because
// the autovec already emits VFMADD231PS. The wins come from:
//   - explicit unroll-by-2 inner-c loop (2 batches per iter = 16 lanes)
//   - skip the explicit zero loop (use VXORPS in dim 0 path)
//
// Frame layout (32 bytes total):
//   q          +0   (*float32, 8B)
//   centroids  +8   (*float32, 8B)
//   K          +16  (int, 8B)
//   out        +24  (*float32, 8B)

TEXT ·scoreAllCentroidsAVX2(SB), NOSPLIT, $0-32
    MOVQ q+0(FP), AX
    MOVQ centroids+8(FP), BX
    MOVQ K+16(FP), CX           // K (count)
    MOVQ out+24(FP), DX

    // R8 = byte stride per dim = K * 4
    MOVQ CX, R8
    SHLQ $2, R8

    // ============================================
    // First dim (d=0) — write out[c] = (q0-c)^2 with VXORPS-free path
    // ============================================
    VBROADCASTSS (AX), Y0       // Y0 = q[0]
    XORQ R9, R9                 // R9 = c byte offset
d0_loop:
    CMPQ R9, R8
    JGE  d0_done

    VMOVUPS (BX)(R9*1), Y1
    VSUBPS  Y1, Y0, Y1
    VMULPS  Y1, Y1, Y1
    VMOVUPS Y1, (DX)(R9*1)

    VMOVUPS 32(BX)(R9*1), Y2
    VSUBPS  Y2, Y0, Y2
    VMULPS  Y2, Y2, Y2
    VMOVUPS Y2, 32(DX)(R9*1)

    ADDQ $64, R9
    JMP  d0_loop
d0_done:

    // ============================================
    // Dims 1..13 — out[c] += (q[d]-c)^2 with FMA
    // ============================================
    MOVQ BX, R10                // R10 = &centroids[d*K] (advances each dim)
    ADDQ R8, R10                // start from dim 1

    // Loop d=1..13. Unrolled below.

    // -------- dim 1 --------
    VBROADCASTSS 4(AX), Y0
    XORQ R9, R9
d1_loop:
    CMPQ R9, R8
    JGE  d1_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)

    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)

    ADDQ $64, R9
    JMP  d1_loop
d1_done:
    ADDQ R8, R10

    // -------- dim 2 --------
    VBROADCASTSS 8(AX), Y0
    XORQ R9, R9
d2_loop:
    CMPQ R9, R8
    JGE  d2_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d2_loop
d2_done:
    ADDQ R8, R10

    // -------- dim 3 --------
    VBROADCASTSS 12(AX), Y0
    XORQ R9, R9
d3_loop:
    CMPQ R9, R8
    JGE  d3_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d3_loop
d3_done:
    ADDQ R8, R10

    // -------- dim 4 --------
    VBROADCASTSS 16(AX), Y0
    XORQ R9, R9
d4_loop:
    CMPQ R9, R8
    JGE  d4_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d4_loop
d4_done:
    ADDQ R8, R10

    // -------- dim 5 --------
    VBROADCASTSS 20(AX), Y0
    XORQ R9, R9
d5_loop:
    CMPQ R9, R8
    JGE  d5_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d5_loop
d5_done:
    ADDQ R8, R10

    // -------- dim 6 --------
    VBROADCASTSS 24(AX), Y0
    XORQ R9, R9
d6_loop:
    CMPQ R9, R8
    JGE  d6_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d6_loop
d6_done:
    ADDQ R8, R10

    // -------- dim 7 --------
    VBROADCASTSS 28(AX), Y0
    XORQ R9, R9
d7_loop:
    CMPQ R9, R8
    JGE  d7_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d7_loop
d7_done:
    ADDQ R8, R10

    // -------- dim 8 --------
    VBROADCASTSS 32(AX), Y0
    XORQ R9, R9
d8_loop:
    CMPQ R9, R8
    JGE  d8_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d8_loop
d8_done:
    ADDQ R8, R10

    // -------- dim 9 --------
    VBROADCASTSS 36(AX), Y0
    XORQ R9, R9
d9_loop:
    CMPQ R9, R8
    JGE  d9_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d9_loop
d9_done:
    ADDQ R8, R10

    // -------- dim 10 --------
    VBROADCASTSS 40(AX), Y0
    XORQ R9, R9
d10_loop:
    CMPQ R9, R8
    JGE  d10_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d10_loop
d10_done:
    ADDQ R8, R10

    // -------- dim 11 --------
    VBROADCASTSS 44(AX), Y0
    XORQ R9, R9
d11_loop:
    CMPQ R9, R8
    JGE  d11_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d11_loop
d11_done:
    ADDQ R8, R10

    // -------- dim 12 --------
    VBROADCASTSS 48(AX), Y0
    XORQ R9, R9
d12_loop:
    CMPQ R9, R8
    JGE  d12_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d12_loop
d12_done:
    ADDQ R8, R10

    // -------- dim 13 --------
    VBROADCASTSS 52(AX), Y0
    XORQ R9, R9
d13_loop:
    CMPQ R9, R8
    JGE  d13_done
    VMOVUPS (R10)(R9*1), Y1
    VMOVUPS (DX)(R9*1), Y2
    VSUBPS  Y1, Y0, Y1
    VFMADD231PS Y1, Y1, Y2
    VMOVUPS Y2, (DX)(R9*1)
    VMOVUPS 32(R10)(R9*1), Y3
    VMOVUPS 32(DX)(R9*1), Y4
    VSUBPS  Y3, Y0, Y3
    VFMADD231PS Y3, Y3, Y4
    VMOVUPS Y4, 32(DX)(R9*1)
    ADDQ $64, R9
    JMP  d13_loop
d13_done:

    VZEROUPPER
    RET
