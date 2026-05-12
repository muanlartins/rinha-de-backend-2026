//go:build !amd64

package kernel

// On non-amd64 the AVX2 name is wired to the generic implementation so call
// sites can use a single symbol without build tags.
func ScanBlock8AVX2(q *float32, block *int16, worst float32, sum *[8]float32) (anyAlive bool) {
	return ScanBlock8Generic(q, block, worst, sum)
}
