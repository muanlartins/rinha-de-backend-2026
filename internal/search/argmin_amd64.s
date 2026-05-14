//go:build amd64

#include "textflag.h"

// func argminAVX2(dists *float32, K int) uint16
//
// Returns the index of the smallest element in dists[0..K]. K must be a
// positive multiple of 8 (production K=4096). On ties the lowest index
// wins (consistent with strict-less-than comparisons in VCMPLTPS).
//
// Algorithm: 8 parallel min-streams using VBLENDVPS-driven update.
//
//   Y0 = current min values per lane (init +INF)
//   Y1 = current min indices per lane as i32 (init 0..7, then +8 per step)
//   Y2 = scratch (loaded dists, then mask)
//   Y3 = step vector [8,8,8,8,8,8,8,8] (broadcast i32)
//   Y4 = scratch (mask cast)
//
// After the streaming loop, fold the 8 lanes into a single min by
// repeatedly halving via VEXTRACTF128 / VSHUFPS, choosing min and
// matching idx at each step. Finally VPEXTRD lane 0 returns the
// scalar index.
//
// Frame layout (32 bytes):
//   dists  +0   (*float32, 8B)
//   K      +8   (int, 8B)
//   ret    +16  (uint16, 2B)  -- padded to 32 by Go ABI

DATA argminInitIdx<>+0(SB)/4, $0
DATA argminInitIdx<>+4(SB)/4, $1
DATA argminInitIdx<>+8(SB)/4, $2
DATA argminInitIdx<>+12(SB)/4, $3
DATA argminInitIdx<>+16(SB)/4, $4
DATA argminInitIdx<>+20(SB)/4, $5
DATA argminInitIdx<>+24(SB)/4, $6
DATA argminInitIdx<>+28(SB)/4, $7
GLOBL argminInitIdx<>(SB), RODATA|NOPTR, $32

DATA argminStep8<>+0(SB)/4, $8
DATA argminStep8<>+4(SB)/4, $8
DATA argminStep8<>+8(SB)/4, $8
DATA argminStep8<>+12(SB)/4, $8
DATA argminStep8<>+16(SB)/4, $8
DATA argminStep8<>+20(SB)/4, $8
DATA argminStep8<>+24(SB)/4, $8
DATA argminStep8<>+28(SB)/4, $8
GLOBL argminStep8<>(SB), RODATA|NOPTR, $32

DATA argminInfMax<>+0(SB)/4, $0x7F800000   // +INF f32
DATA argminInfMax<>+4(SB)/4, $0x7F800000
DATA argminInfMax<>+8(SB)/4, $0x7F800000
DATA argminInfMax<>+12(SB)/4, $0x7F800000
DATA argminInfMax<>+16(SB)/4, $0x7F800000
DATA argminInfMax<>+20(SB)/4, $0x7F800000
DATA argminInfMax<>+24(SB)/4, $0x7F800000
DATA argminInfMax<>+28(SB)/4, $0x7F800000
GLOBL argminInfMax<>(SB), RODATA|NOPTR, $32

TEXT ·argminAVX2(SB), NOSPLIT, $0-24
    MOVQ dists+0(FP), AX
    MOVQ K+8(FP), CX

    VMOVDQU argminInfMax<>(SB), Y0      // Y0 = min values  (+INF)
    VMOVDQU argminInitIdx<>(SB), Y1     // Y1 = min indices (0..7)
    VMOVDQU argminInitIdx<>(SB), Y5     // Y5 = current index counter (0..7)
    VMOVDQU argminStep8<>(SB), Y3       // Y3 = step (+8 per iter)

    SHLQ $2, CX                          // CX = K * 4 (byte count)
    XORQ R8, R8                          // R8 = byte offset

loop:
    CMPQ R8, CX
    JGE  reduce

    VMOVUPS    (AX)(R8*1), Y2           // load 8 floats
    VCMPPS     $1, Y0, Y2, Y4            // Y4 = mask where Y2 < Y0  (CMP_LT_OS)
    VBLENDVPS  Y4, Y2, Y0, Y0           // Y0 = where mask: Y2 else Y0
    VBLENDVPS  Y4, Y5, Y1, Y1           // Y1 = where mask: Y5 else Y1
    VPADDD     Y3, Y5, Y5                // Y5 += 8 (advance cur index)

    ADDQ $32, R8
    JMP  loop

reduce:
    // Reduce 8 lanes (Y0/Y1) → 1 lane via three halvings.
    //
    // Step 1: compare upper 128 vs lower 128.
    VEXTRACTF128 $1, Y0, X6              // X6 = Y0[hi]
    VEXTRACTF128 $1, Y1, X7              // X7 = Y1[hi]
    VCMPPS       $1, X0, X6, X8          // X8 = (X6 < X0_lo)
    VBLENDVPS    X8, X6, X0, X0          // X0 = min of 8 lanes folded to 4
    VBLENDVPS    X8, X7, X1, X1

    // Step 2: compare pair-of-2 (swap halves of X0 to compare lanes [0,1] vs [2,3]).
    VPSHUFD   $0x4e, X0, X6              // X6 = X0 with halves swapped
    VPSHUFD   $0x4e, X1, X7
    VCMPPS    $1, X0, X6, X8
    VBLENDVPS X8, X6, X0, X0
    VBLENDVPS X8, X7, X1, X1

    // Step 3: compare neighbours (lanes [0] vs [1]).
    VPSHUFD   $0xb1, X0, X6              // swap adjacent pairs
    VPSHUFD   $0xb1, X1, X7
    VCMPPS    $1, X0, X6, X8
    VBLENDVPS X8, X7, X1, X1             // we only need the index now

    VPEXTRD $0, X1, AX                   // scalar i32 winning index
    MOVW    AX, ret+16(FP)
    VZEROUPPER
    RET
