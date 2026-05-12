package dataset

import "unsafe"

// unsafeSlice reinterprets a *int16 + total byte count as a []byte. Used to
// serialize/deserialize int16 slabs with no copy. Caller must guarantee the
// underlying memory is alive for the lifetime of the returned slice.
func unsafeSlice(p *int16, lenBytes int) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(p)), lenBytes)
}
