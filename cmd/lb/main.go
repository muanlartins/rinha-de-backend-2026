//go:build linux

package main

import (
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

// SCM_RIGHTS fd-passing load balancer.
//
// Accepts TCP on $LB_PORT (default 9999) and round-robins each new client
// connection to one of the upstream APIs listed in $UPSTREAMS (comma-
// separated paths to each api's .sock.ctrl SOCK_STREAM unix socket). The
// TCP fd is handed off via sendmsg(SCM_RIGHTS); after the handoff the LB
// closes its copy of the fd, removes itself from the data path, and the
// API talks to the client directly. See internal/fdpass/recv_linux.go
// for the matching receiver and docs/lectures/12-scm-rights.md.
//
// Single-threaded by design: at the bot's 0.10 LB CPU budget, all the
// LB ever does per request is one accept4 + one sendmsg + one close.
// No netpoller, no goroutines, no allocations on the hot path.

func main() {
	runtime.GOMAXPROCS(1)

	port := 9999
	if p := os.Getenv("LB_PORT"); p != "" {
		if n, err := strconv.Atoi(p); err == nil && n > 0 && n < 65536 {
			port = n
		}
	}

	ups := os.Getenv("UPSTREAMS")
	if ups == "" {
		log.Fatal("UPSTREAMS env required (comma-separated ctrl socket paths)")
	}
	upstreams := strings.Split(ups, ",")
	for i, u := range upstreams {
		upstreams[i] = strings.TrimSpace(u)
	}

	ctrlFds := make([]int, len(upstreams))
	for i, p := range upstreams {
		fd, err := dialCtrl(p)
		if err != nil {
			log.Fatalf("dial ctrl %s: %v", p, err)
		}
		ctrlFds[i] = fd
	}

	listenFd, err := unix.Socket(unix.AF_INET, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		log.Fatalf("socket: %v", err)
	}
	_ = unix.SetsockoptInt(listenFd, unix.SOL_SOCKET, unix.SO_REUSEADDR, 1)
	if err := unix.Bind(listenFd, &unix.SockaddrInet4{Port: port}); err != nil {
		log.Fatalf("bind :%d: %v", port, err)
	}
	if err := unix.Listen(listenFd, 65535); err != nil {
		log.Fatalf("listen: %v", err)
	}

	log.Printf("lb listening on :%d, upstreams=%v", port, upstreams)

	var next atomic.Uint32

	for {
		connFd, _, err := unix.Accept4(listenFd, unix.SOCK_CLOEXEC)
		if err != nil {
			if err == unix.EINTR {
				continue
			}
			log.Printf("accept: %v", err)
			continue
		}

		i := int(next.Add(1)-1) % len(upstreams)

		if err := sendFd(ctrlFds[i], connFd); err != nil {
			log.Printf("sendmsg upstream %d (%s): %v; reconnecting", i, upstreams[i], err)
			_ = unix.Close(ctrlFds[i])
			if nfd, derr := dialCtrl(upstreams[i]); derr == nil {
				ctrlFds[i] = nfd
			} else {
				log.Printf("reconnect upstream %d failed: %v", i, derr)
				ctrlFds[i] = -1
			}
		}
		_ = unix.Close(connFd)
	}
}

func dialCtrl(path string) (int, error) {
	fd, err := unix.Socket(unix.AF_UNIX, unix.SOCK_STREAM|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return -1, err
	}
	if err := unix.Connect(fd, &unix.SockaddrUnix{Name: path}); err != nil {
		_ = unix.Close(fd)
		return -1, err
	}
	return fd, nil
}

func sendFd(ctrlFd, payloadFd int) error {
	if ctrlFd < 0 {
		return unix.EBADF
	}
	oob := unix.UnixRights(payloadFd)
	return unix.Sendmsg(ctrlFd, []byte{0}, oob, nil, 0)
}
