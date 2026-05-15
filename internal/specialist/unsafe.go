package specialist

import "unsafe"

// ptr is the standard "pointer-to-anything" shim used in unsafe casts.
// Kept as a tiny helper to keep the use of unsafe.Pointer narrow and
// auditable.
func ptr(p *int16) unsafe.Pointer { return unsafe.Pointer(p) }
