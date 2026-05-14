//go:build !linux

package api

import "net"

// tuneSCMConn is a no-op on non-linux builds. The production target is
// linux/amd64 (the rinha bot Mac Mini runs Ubuntu); the darwin variant
// exists only for local tests and benches.
func tuneSCMConn(c net.Conn) { _ = c }
