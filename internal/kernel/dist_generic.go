package kernel

import "unsafe"

// ScanBlock8Generic is the pure-Go reference implementation of the kernel.
// Used as the cross-check oracle on amd64 and as the production hot-path
// implementation on non-amd64 (development on darwin/arm64).
//
// We use 32-bit float accumulation with separate multiplies and adds (no
// FMA), matching how the early Go compiler lowers a*b+c on platforms
// without hardware FMA. The amd64 assembly uses VFMADD231PS which performs
// the multiply-add with a single rounding; the two outputs agree to within
// ~1e-3 relative on data ranges we encounter.
func ScanBlock8Generic(q *float32, block *int16, worst float32, sum *[8]float32) (anyAlive bool) {
	const dims = 14
	const lanes = 8
	qarr := (*[dims]float32)(unsafe.Pointer(q))
	barr := (*[dims * lanes]int16)(unsafe.Pointer(block))

	for lane := 0; lane < lanes; lane++ {
		sum[lane] = 0
	}

	// Cadence 8 (single gate) — phase 24 alignment with the amd64 asm,
	// which removed the dim-4 and dim-6 checkpoints. Keeps the generic
	// path as a faithful oracle.
	checkpoints := [1]int{8}
	cpIdx := 0

	for d := 0; d < dims; d++ {
		qd := qarr[d]
		for lane := 0; lane < lanes; lane++ {
			diff := qd - float32(barr[d*lanes+lane])
			sum[lane] += diff * diff
		}

		if cpIdx < len(checkpoints) && d+1 == checkpoints[cpIdx] {
			cpIdx++
			alive := false
			for lane := 0; lane < lanes; lane++ {
				if sum[lane] < worst {
					alive = true
					break
				}
			}
			if !alive {
				return false
			}
		}
	}

	for lane := 0; lane < lanes; lane++ {
		if sum[lane] < worst {
			return true
		}
	}
	return false
}
