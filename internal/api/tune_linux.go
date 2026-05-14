//go:build linux

package api

import (
	"net"
	"syscall"
)

// tuneSCMConn applies low-latency TCP options to a connection adopted
// from an SCM_RIGHTS fd-passing channel. The LB (so-no-forevis) accepts
// the TCP socket, then hands the raw fd to us; once we wrap it via
// os.NewFile + net.FileConn we get back a net.Conn (a *net.TCPConn for
// TCP sockets, although the static type erasure means we go through
// SyscallConn to be sure).
//
// Two options:
//
//   TCP_NODELAY  (Nagle off) — prevents the kernel from buffering small
//     writes waiting for an ACK on a previous segment. Our /fraud-score
//     responses are 35–36 bytes; without NODELAY, pipelined responses
//     on the same keep-alive connection can be held until the prior
//     ACK arrives (up to ~RTT). crepao-da-massa sets this on its
//     accepted clients (`server.cpp:421`).
//
//   TCP_QUICKACK — tells the kernel to send ACKs immediately, no
//     200 ms delayed-ack window. Linux-only. Affects how fast our
//     ACKs reach the client after their request; matters when k6
//     pipelines. crepao sets this at `server.cpp:423`.
//
// Phase 33. One-shot per connection (called once before the per-conn
// goroutine starts); zero recurring cost.
func tuneSCMConn(c net.Conn) {
	if tc, ok := c.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}
	if rc, ok := c.(interface {
		SyscallConn() (syscall.RawConn, error)
	}); ok {
		raw, err := rc.SyscallConn()
		if err != nil {
			return
		}
		_ = raw.Control(func(fd uintptr) {
			// TCP_QUICKACK = 12 on Linux. golang.org/x/sys/unix exposes
			// it but we already depend on syscall, so use the constant
			// directly to avoid an extra import dependency.
			const tcpQuickAck = 12
			_ = syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, tcpQuickAck, 1)
		})
	}
}
