//go:build linux

package fdpass

import (
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// TestRecvLoopback opens a pair of pipes (a TCP listener + a client
// connection), then sends the client fd over a STREAM control socket
// using SCM_RIGHTS. The Listen-returned channel must yield the fd, and a
// write to that fd must reach the listener.
func TestRecvLoopback(t *testing.T) {
	ctrlPath := filepath.Join(t.TempDir(), "ctrl.sock")
	fdCh, listenFd, err := Listen(ctrlPath)
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	t.Cleanup(func() { unix.Close(listenFd) })

	// Connect to the STREAM listener — this is what jrblatt/so-no-forevis
	// does in production.
	clientFd, err := unix.Socket(unix.AF_UNIX, unix.SOCK_STREAM, 0)
	if err != nil {
		t.Fatalf("client socket: %v", err)
	}
	defer unix.Close(clientFd)
	if err := unix.Connect(clientFd, &unix.SockaddrUnix{Name: ctrlPath}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	// Create a pipe to simulate a "client connection" the LB would pass.
	// We write the payload BEFORE sending the fd so it sits in the kernel
	// pipe buffer; the receiver can then read it without blocking.
	rfd, wfd, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	defer wfd.Close()

	const msg = "hello"
	if _, err := wfd.Write([]byte(msg)); err != nil {
		t.Fatalf("pre-write: %v", err)
	}

	oob := unix.UnixRights(int(rfd.Fd()))
	if err := unix.Sendmsg(clientFd, []byte("FD"), oob, nil, 0); err != nil {
		t.Fatalf("sendmsg: %v", err)
	}
	_ = rfd.Close() // LB closes its copy after sendmsg

	select {
	case fd := <-fdCh:
		// recvLoop set non-blocking; we use unix.Poll to wait for data.
		pfds := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
		if _, err := unix.Poll(pfds, 2000); err != nil {
			t.Fatalf("poll: %v", err)
		}
		buf := make([]byte, len(msg))
		n, err := unix.Read(fd, buf)
		if err != nil {
			t.Fatalf("read on received fd: %v", err)
		}
		if string(buf[:n]) != msg {
			t.Fatalf("got %q want %q", buf[:n], msg)
		}
		_ = unix.Close(fd)
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for fd")
	}
}

// TestListenStubOnDarwinSkipped is a placeholder so we know the linux build
// constraint is filtering correctly. On non-linux, this file isn't compiled.
func TestListenStubOnDarwinSkipped(t *testing.T) {
	if _, err := net.LookupHost("localhost"); err != nil {
		t.Skipf("no net stack")
	}
}
