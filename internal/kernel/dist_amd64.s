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
// The kernel computes 8 squared Euclidean distances between q (14 f32) and
// 8 reference vectors stored dim-major in block (14*8 int16). Early-exit
// at dims 4, 6, 8 against worst.
//
// Register usage:
//   AX = &q[0]
//   BX = &block[0]
//   CX = &sum[0]
//   R8 = alive mask (low 8 bits)
//   Y0 = squared-distance accumulator, 8 f32 lanes
//   Y1 = scratch (loaded block dim, converted to f32)
//   Y2 = scratch (broadcast query dim, then diff)
//   Y3 = scratch (cmp mask)
//   Y15 = broadcast worst across 8 lanes

TEXT ·ScanBlock8AVX2(SB), NOSPLIT, $0-40
	MOVQ         q+0(FP), AX
	MOVQ         block+8(FP), BX
	VBROADCASTSS worst+16(FP), Y15
	MOVQ         sum+24(FP), CX

	VXORPS Y0, Y0, Y0

	// --- Dims 0..3 -----------------------------------------------------
	VPMOVSXWD    (BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS (AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    16(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 4(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    32(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 8(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    48(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 12(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// --- Checkpoint 1: 4 dims accumulated ------------------------------
	VCMPPS    $0x01, Y15, Y0, Y3
	VMOVMSKPS Y3, R8
	TESTL     R8, R8
	JZ        dead

	// --- Dims 4..5 -----------------------------------------------------
	VPMOVSXWD    64(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 16(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    80(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 20(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// --- Checkpoint 2: 6 dims accumulated ------------------------------
	VCMPPS    $0x01, Y15, Y0, Y3
	VMOVMSKPS Y3, R8
	TESTL     R8, R8
	JZ        dead

	// --- Dims 6..7 -----------------------------------------------------
	VPMOVSXWD    96(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 24(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    112(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 28(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// --- Checkpoint 3: 8 dims accumulated ------------------------------
	VCMPPS    $0x01, Y15, Y0, Y3
	VMOVMSKPS Y3, R8
	TESTL     R8, R8
	JZ        dead

	// --- Dims 8..13 (no further checkpoints) ---------------------------
	VPMOVSXWD    128(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 32(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    144(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 36(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    160(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 40(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    176(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 44(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    192(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 48(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	VPMOVSXWD    208(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VBROADCASTSS 52(AX), Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// --- Final liveness ------------------------------------------------
	VCMPPS    $0x01, Y15, Y0, Y3
	VMOVMSKPS Y3, R8
	TESTL     R8, R8
	JZ        dead

	VMOVUPS Y0, (CX)
	MOVB    $1, anyAlive+32(FP)
	VZEROUPPER
	RET

dead:
	VMOVUPS Y0, (CX)
	MOVB    $0, anyAlive+32(FP)
	VZEROUPPER
	RET
