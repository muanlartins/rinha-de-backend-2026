// BlockScan8: squared Euclidean distances from one int16 query to 8
// reference vectors stored in dim-major block layout (16 dims × 8 vectors).
//
// Per-dim pattern (16 dims for Stride=16):
//   VPBROADCASTW [query+d*2], X2  // 8 copies of query[d]
//   VMOVDQU      [block+d*16], X3 // 8 int16s of dim d, one per vector
//   VPSUBW       X2, X3, X3       // X3 = ref - query (8 int16 diffs)
//   VPMOVSXWD    X3, Y3           // sign-extend to 8 int32s
//   VPMULLD      Y3, Y3, Y3       // Y3 = diff² (8 int32, fits — max ~1.07e9)
//   VEXTRACTI128 $0, Y3, X4
//   VEXTRACTI128 $1, Y3, X5
//   VPMOVSXDQ    X4, Y4           // 4 int64s (low half of Y3)
//   VPMOVSXDQ    X5, Y5           // 4 int64s (high half of Y3)
//   VPADDQ       Y4, Y0, Y0       // accumulators: Y0 for vectors 0..3
//   VPADDQ       Y5, Y1, Y1       //               Y1 for vectors 4..7
//
// Final reduction: write Y0 + Y1 (8 int64s = 64 bytes) into out[8].

#include "textflag.h"

#define DIM(d)                       \
    VPBROADCASTW (d*2)(AX), X2     ; \
    VMOVDQU      (d*16)(BX), X3    ; \
    VPSUBW       X2, X3, X3        ; \
    VPMOVSXWD    X3, Y3            ; \
    VPMULLD      Y3, Y3, Y3        ; \
    VEXTRACTI128 $0, Y3, X4        ; \
    VEXTRACTI128 $1, Y3, X5        ; \
    VPMOVSXDQ    X4, Y4            ; \
    VPMOVSXDQ    X5, Y5            ; \
    VPADDQ       Y4, Y0, Y0        ; \
    VPADDQ       Y5, Y1, Y1

// func blockScan8AVX2(query, block *int16, out *[8]int64)
TEXT ·blockScan8AVX2(SB), NOSPLIT, $0-24
	MOVQ query+0(FP), AX
	MOVQ block+8(FP), BX
	MOVQ out+16(FP), CX

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1

	DIM(0)
	DIM(1)
	DIM(2)
	DIM(3)
	DIM(4)
	DIM(5)
	DIM(6)
	DIM(7)
	DIM(8)
	DIM(9)
	DIM(10)
	DIM(11)
	DIM(12)
	DIM(13)
	DIM(14)
	DIM(15)

	VMOVDQU Y0, 0(CX)
	VMOVDQU Y1, 32(CX)
	VZEROUPPER
	RET
