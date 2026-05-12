//go:build amd64

package kernel

// ScanBlock8AVX2 computes 8 squared Euclidean distances between q and the 8
// reference vectors of a dim-major block. The 3-stage early-exit (after
// dims 4, 6, 8) is performed inside the kernel; the function returns
// anyAlive=false when all 8 lanes have already exceeded worst, in which
// case the contents of sum past the exit point are partial sums and the
// caller must not insert them.
//
//go:noescape
func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) (anyAlive bool)
