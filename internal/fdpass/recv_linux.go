//go:build linux

package fdpass

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// Listen opens a SOCK_SEQPACKET listener at ctrlPath and yields received
// file descriptors on the returned channel. Each Recvmsg may carry one or
// more fds via SCM_RIGHTS; all of them get pushed to the channel.
//
// The channel is closed when the listener exits (typically on shutdown).
// Returns the listener fd so the caller can Close it during shutdown.
func Listen(ctrlPath string) (<-chan int, int, error) {
	_ = unix.Unlink(ctrlPath)

	fd, err := unix.Socket(unix.AF_UNIX, unix.SOCK_SEQPACKET|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return nil, 0, fmt.Errorf("socket: %w", err)
	}
	if err := unix.Bind(fd, &unix.SockaddrUnix{Name: ctrlPath}); err != nil {
		_ = unix.Close(fd)
		return nil, 0, fmt.Errorf("bind %s: %w", ctrlPath, err)
	}
	_ = os.Chmod(ctrlPath, 0o666)
	if err := unix.Listen(fd, 16); err != nil {
		_ = unix.Close(fd)
		return nil, 0, fmt.Errorf("listen: %w", err)
	}

	out := make(chan int, 256)
	go acceptLoop(fd, out)
	return out, fd, nil
}

// acceptLoop accepts new control connections from the LB. Each control
// connection can carry many fds over its lifetime (the LB re-uses the
// connection across requests). We spin up one recvLoop per connection.
func acceptLoop(listenFd int, out chan<- int) {
	defer close(out)
	for {
		connFd, _, err := unix.Accept4(listenFd, unix.SOCK_CLOEXEC)
		if err != nil {
			if err == unix.EINTR {
				continue
			}
			return
		}
		go recvLoop(connFd, out)
	}
}

// recvLoop reads ancillary-data messages on the given UDS connection.
// Every fd carried via SCM_RIGHTS gets set non-blocking (required by the
// Go netpoller) and pushed to out.
func recvLoop(connFd int, out chan<- int) {
	defer unix.Close(connFd)

	// 16-byte data buffer: the LB usually sends 1-4 bytes of header/tag
	// alongside the fd. We don't care about the contents.
	buf := make([]byte, 16)
	// Ancillary buffer big enough for 4 fds (the LB may batch).
	oobCap := unix.CmsgSpace(4 * 4)
	oob := make([]byte, oobCap)

	for {
		_, oobn, _, _, err := unix.Recvmsg(connFd, buf, oob, 0)
		if err != nil {
			if err == unix.EINTR {
				continue
			}
			return
		}
		if oobn == 0 {
			continue
		}
		msgs, err := unix.ParseSocketControlMessage(oob[:oobn])
		if err != nil {
			continue
		}
		for i := range msgs {
			fds, err := unix.ParseUnixRights(&msgs[i])
			if err != nil {
				continue
			}
			for _, fd := range fds {
				if err := unix.SetNonblock(fd, true); err != nil {
					_ = unix.Close(fd)
					continue
				}
				out <- fd
			}
		}
	}
}
