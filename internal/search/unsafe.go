package search

import "unsafe"

// _ptr reinterprets an int16 pointer as a raw pointer for array casts.
func _ptr(p *int16) unsafe.Pointer { return unsafe.Pointer(p) }
