//go:build !linux

package api

import (
	"net"
	"os"
)

// adoptFD on non-linux: keep the previous net.FileConn wrapping. The
// direct-syscall path (rawfd_linux.go) is linux-only because we
// integrate with Go's runtime via LockOSThread + blocking syscalls;
// macOS Docker testing uses this fallback.
func adoptFD(fd int, h *Handler) {
	f := os.NewFile(uintptr(fd), "scm-fd")
	c, err := net.FileConn(f)
	_ = f.Close()
	if err != nil {
		return
	}
	tuneSCMConn(c)
	go handleRawConn(c, h)
}
