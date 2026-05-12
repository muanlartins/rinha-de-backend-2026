// Squared Euclidean distance over 16 int16 lanes using SSE4.1.
//
// PMADDWL produces 4 int32 sums per 8-int16 input. Each output lane is at most
// 2·(32000)² ≈ 2.05e9 — fits int32 (barely). But two PMADDWL chunks then
// produce 8 int32 lanes; summing them in int32 (PADDD) can overflow. We
// expand each chunk's 4 int32s to 4 int64s via PMOVSXDQ, then accumulate in
// int64, then reduce to scalar.

#include "textflag.h"

// func sqdistAVX2(query, ref *int16) int64
TEXT ·sqdistAVX2(SB), NOSPLIT, $0-24
	MOVQ query+0(FP), AX
	MOVQ ref+8(FP), BX

	// Chunk 0: lanes [0..7]
	MOVOU (AX), X0
	MOVOU (BX), X1
	PSUBW X1, X0
	PMADDWL X0, X0          // X0 = 4 int32 lanes, each = (q[2k]-r[2k])²+(q[2k+1]-r[2k+1])²

	// Chunk 1: lanes [8..15] (lanes 14, 15 are zero in both inputs)
	MOVOU 16(AX), X1
	MOVOU 16(BX), X2
	PSUBW X2, X1
	PMADDWL X1, X1          // X1 = 4 int32 lanes

	// Sign-extend each chunk's 4 int32 to 2× 2 int64s, then sum them all in int64.
	PMOVSXDQ X0, X2         // X2 = [X0[0], X0[1]] as 2 int64
	PSHUFD $0x0E, X0, X3    // X3 = [X0[2], X0[3], -, -]
	PMOVSXDQ X3, X3         // X3 = [X0[2], X0[3]] as 2 int64
	PADDQ X3, X2            // X2 += X3 → X2 = [X0[0]+X0[2], X0[1]+X0[3]]

	PMOVSXDQ X1, X3         // X3 = [X1[0], X1[1]] as 2 int64
	PSHUFD $0x0E, X1, X4    // X4 = [X1[2], X1[3], -, -]
	PMOVSXDQ X4, X4         // X4 = [X1[2], X1[3]] as 2 int64
	PADDQ X4, X3            // X3 += X4 → X3 = [X1[0]+X1[2], X1[1]+X1[3]]

	PADDQ X3, X2            // X2 holds 2 int64s = total split into [even, odd] halves

	MOVQ X2, AX
	PEXTRQ $1, X2, CX
	ADDQ CX, AX
	MOVQ AX, ret+16(FP)
	RET
