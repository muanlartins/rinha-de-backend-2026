//go:build linux

package api

import (
	"runtime"
	"syscall"
)

// handleRawFD is the Phase 37 hot-path connection handler: read parse
// write loop with direct syscalls on the raw fd, bypassing Go's
// netpoller entirely.
//
// Why this exists:
//
// The previous handleRawConn path wraps the SCM_RIGHTS-received fd with
// os.NewFile + net.FileConn, producing a *net.TCPConn. Every Read/Write
// then goes through Go's netpoller machinery:
//   - poll.FD lock acquisition
//   - non-blocking syscall.Read; if EAGAIN, park on epoll
//   - deadline tracking, interface dispatch
//
// For SCM_RIGHTS-passed sockets, none of that is necessary. The fd is
// just an int; the kernel can block the OS thread directly. We:
//   1. fcntl the fd to blocking mode.
//   2. LockOSThread so the runtime can spawn other M's for other
//      goroutines while this thread is blocked in syscall.
//   3. Use syscall.Read / syscall.Write directly.
//
// This eliminates ~200-500 ns of netpoller integration per Read/Write,
// plus the wakeup-park-wakeup cycle when data isn't yet available.
// Estimated bot p99 reduction: 5-15 µs universal (every request).
//
// Trade-offs vs the netpoller path:
//   - Each connection consumes one OS thread while it has any
//     in-flight syscall. Bounded by active connection count (typically
//     ~100 under k6 load). OS-thread cost is ~16 KB stack + Go-side
//     state = trivial within our 165 MB cgroup.
//   - GOMAXPROCS=1 still applies: only one goroutine runs Go code at
//     a time. M's blocked in syscall don't count. Other goroutines
//     are still serialized.
//
// Correctness:
//   - syscall.Read returns (n, err); n=0 && err==nil is EOF (matches
//     net.Conn semantics).
//   - Short writes are handled with a loop (rare on small responses
//     but possible).
//   - EINTR is retried (signal-handler interruption; never fatal).
func handleRawFD(fd int, h *Handler) {
	defer syscall.Close(fd)

	// Pin this goroutine to its OS thread. Blocking syscalls below
	// will block that thread without stalling the runtime, because
	// Go spawns another M to keep other goroutines running. No
	// UnlockOSThread call: when the goroutine returns (connection
	// closes), the thread terminates with it.
	runtime.LockOSThread()

	// Ensure blocking mode. so-no-forevis typically hands TCP sockets
	// in blocking mode already, but explicitly forcing it avoids any
	// stray non-blocking flag from the LB.
	if err := syscall.SetNonblock(fd, false); err != nil {
		return
	}

	// Phase 33 socket tuning, set directly on the fd (no need to go
	// through net.Conn.SyscallConn). TCP_NODELAY disables Nagle on
	// small responses; TCP_QUICKACK disables the 40 ms delayed-ACK
	// window. Both one-shot per connection.
	const tcpNoDelay = 1
	const tcpQuickAck = 12
	_ = syscall.SetsockoptInt(fd, syscall.IPPROTO_TCP, tcpNoDelay, 1)
	_ = syscall.SetsockoptInt(fd, syscall.IPPROTO_TCP, tcpQuickAck, 1)

	bufRef := rawReadBufPool.Get().(*[]byte)
	buf := *bufRef
	used := 0
	pos := 0
	defer func() {
		if cap(buf) <= maxRequestSize {
			*bufRef = buf[:cap(buf)]
			rawReadBufPool.Put(bufRef)
		}
	}()

	for {
		// Find header end in buf[pos:used]. Buffer may already contain
		// data from a previous read (pipelined keep-alive — see Phase 35a).
		var headEnd int
		for {
			if idx := indexHeaderEnd(buf[pos:used]); idx >= 0 {
				headEnd = pos + idx + 4
				break
			}
			// Need more data. Compact / grow if buffer is full.
			if used == len(buf) {
				if pos > 0 {
					copy(buf, buf[pos:used])
					used -= pos
					pos = 0
				} else if used >= maxRequestSize {
					return
				} else {
					nb := make([]byte, len(buf)*2)
					copy(nb, buf[:used])
					buf = nb
				}
			}
			n, err := syscall.Read(fd, buf[used:])
			if n > 0 {
				used += n
				continue
			}
			if err == syscall.EINTR {
				continue
			}
			return // EOF or fatal error
		}

		path, contentLen := parseRequestLine(buf[pos:headEnd])
		if contentLen > maxRequestSize-(headEnd-pos) {
			return
		}
		bodyEnd := headEnd + contentLen
		for used < bodyEnd {
			if used == len(buf) {
				if pos > 0 {
					copy(buf, buf[pos:used])
					used -= pos
					headEnd -= pos
					bodyEnd -= pos
					pos = 0
				} else {
					nb := make([]byte, len(buf)*2)
					copy(nb, buf[:used])
					buf = nb
				}
			}
			n, err := syscall.Read(fd, buf[used:])
			if n > 0 {
				used += n
				continue
			}
			if err == syscall.EINTR {
				continue
			}
			return
		}

		resp := h.RouteRaw(path, buf[headEnd:bodyEnd])

		// Write the response, looping on short writes / EINTR.
		woff := 0
		for woff < len(resp) {
			n, err := syscall.Write(fd, resp[woff:])
			if n > 0 {
				woff += n
				continue
			}
			if err == syscall.EINTR {
				continue
			}
			return
		}

		// Advance to next request in buffer (pipelined case) — same
		// pos-cursor pattern as Phase 35a in handleRawConn.
		pos = bodyEnd
		if pos == used {
			pos = 0
			used = 0
		}
	}
}

// adoptFD is the platform-specific dispatcher: on linux, it takes the
// raw fd and starts a syscall-based handler goroutine. ServeFDChannel
// calls this for each fd received over SCM_RIGHTS.
func adoptFD(fd int, h *Handler) {
	go handleRawFD(fd, h)
}
