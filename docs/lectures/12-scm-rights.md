# Lecture 12 — SCM_RIGHTS: How the Top 4 Skip a Userspace Hop

The top 4 submissions on the leaderboard share one trick that nobody else uses: instead of a normal HAProxy-style proxy, they ship a tiny load balancer that **passes the accepted client file descriptor directly to the backend over a Unix socket**. Once handed off, the backend speaks HTTP to the client; the LB is out of the data path for the rest of the connection.

This lecture covers the concept end-to-end: what `SCM_RIGHTS` is, why fd passing works in the Linux kernel, the protocol `so-no-forevis` uses, and how to plug a recvmsg loop into a Go epoll server.

## The framing: where does proxy time go?

A traditional HAProxy/nginx proxy does this per request:

1. `accept()` the client TCP connection on `:9999`.
2. `connect()` an upstream socket (TCP or UDS) to the backend.
3. `read()` from the client, `write()` to the backend.
4. `read()` from the backend, `write()` to the client.

Each `read`/`write` is a kernel syscall with userspace buffer copying. For a 500-byte JSON request and 30-byte response, that's at minimum:

- 1 client `read` (request enters proxy userspace)
- 1 backend `write` (request leaves proxy userspace)
- 1 backend `read` (response enters proxy userspace)
- 1 client `write` (response leaves proxy userspace)

Plus the syscalls for `accept`, `connect`, `close`. Total: ~6 syscalls and ~1 KB of byte copying per request, ~20-40 µs at minimum on Haswell.

`splice(2)` removes the userspace copying via in-kernel pipes — that's what `steixeira93/rinha-lb` does. But it still needs the proxy to mediate the read/write between two sockets.

**Fd-passing skips it entirely.** After `accept`, the LB hands the client fd over a control socket and is done. No `connect` to backend, no proxying, no copying. The backend's epoll loop sees the client fd appear and services it directly.

## What `SCM_RIGHTS` does

Unix Domain Sockets support "ancillary data" via `sendmsg(2)`. One ancillary type is `SCM_RIGHTS`: a payload of file descriptors.

When you `sendmsg` with `SCM_RIGHTS` over a UDS:

1. The kernel reads the fd numbers from the sender's fd table.
2. For each fd, the kernel duplicates the underlying file (kernel `struct file`) into the receiver's fd table at the next-available fd number.
3. The receiver gets *new* fd numbers (different from the sender's), but they reference the same open files.
4. The sender can `close()` its copies; the receiver's still works.

This is how the kernel implements fd inheritance across `fork()` and `exec()`, generalized to arbitrary processes connected by a UDS. The receiver process doesn't even need to be a child of the sender.

For sockets specifically: when an fd points to an open TCP connection (after `accept`), passing it via SCM_RIGHTS gives the receiver a fully-formed `struct sock` in their fd table. They can `read`/`write` it, register it with epoll/io_uring, etc. The kernel's TCP stack doesn't care which userspace process holds the fd — it routes packets to and from the socket as long as someone has it open.

## The protocol `so-no-forevis` uses

`jrblatt/so-no-forevis:v1.0.0` is the public LB image used by rank #1, #3, #5. The protocol it expects from upstreams:

1. The upstream listens on a regular UDS at `/sockets/api-1.sock` (etc.) for the standard `accept` + `read`/`write` HTTP flow as a fallback.
2. The upstream also listens on a **control** UDS at `/sockets/api-1.sock.ctrl` in `SOCK_SEQPACKET` mode.
3. When the LB accepts a connection from a client on `:9999`, it picks a backend (round-robin or random), then does:
   ```c
   struct msghdr msg = {0};
   struct iovec iov = { .iov_base = "FD", .iov_len = 2 };  // optional small data
   msg.msg_iov = &iov;
   msg.msg_iovlen = 1;
   char cbuf[CMSG_SPACE(sizeof(int))];
   msg.msg_control = cbuf;
   msg.msg_controllen = sizeof(cbuf);
   struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
   cmsg->cmsg_level = SOL_SOCKET;
   cmsg->cmsg_type = SCM_RIGHTS;
   cmsg->cmsg_len = CMSG_LEN(sizeof(int));
   *(int *)CMSG_DATA(cmsg) = client_fd;
   sendmsg(ctrl_fd, &msg, 0);
   close(client_fd);  // LB's copy of the fd
   ```
4. The upstream `recvmsg`s on `.ctrl`, extracts the fd, registers it with its epoll loop, and proceeds as if it had just `accept`ed the client itself.

`SOCK_SEQPACKET` is used for the control channel because it guarantees message boundaries — each `sendmsg` produces exactly one `recvmsg` on the other side, with ancillary data intact. `SOCK_STREAM` would also work but you'd have to manage your own message framing.

The "FD" data byte in the iovec is irrelevant to the fd-passing logic — it's just there so `recvmsg` doesn't return zero bytes (Linux requires non-empty payload to deliver ancillary data on some kernels). Both peers ignore the byte's value.

## Implementing in Go

Go's `syscall` package and `golang.org/x/sys/unix` together expose everything we need:

```go
package fdpass

import (
    "golang.org/x/sys/unix"
    "net"
)

// Listen opens a SEQPACKET UDS at ctrlPath and yields received fds on the
// returned channel. The channel closes when the listener errors out
// (typically on shutdown).
func Listen(ctrlPath string) (<-chan int, error) {
    // Remove stale socket if present.
    _ = unix.Unlink(ctrlPath)

    fd, err := unix.Socket(unix.AF_UNIX, unix.SOCK_SEQPACKET, 0)
    if err != nil { return nil, err }
    if err := unix.Bind(fd, &unix.SockaddrUnix{Name: ctrlPath}); err != nil {
        unix.Close(fd)
        return nil, err
    }
    if err := unix.Listen(fd, 16); err != nil {
        unix.Close(fd)
        return nil, err
    }

    out := make(chan int, 64)
    go acceptLoop(fd, out)
    return out, nil
}

func acceptLoop(listenFd int, out chan<- int) {
    defer close(out)
    for {
        connFd, _, err := unix.Accept(listenFd)
        if err != nil { return }
        // Each control connection can pass many fds over its lifetime.
        go recvLoop(connFd, out)
    }
}

func recvLoop(connFd int, out chan<- int) {
    defer unix.Close(connFd)
    buf := make([]byte, 16)
    oob := make([]byte, unix.CmsgSpace(4))  // room for one int
    for {
        _, oobn, _, _, err := unix.Recvmsg(connFd, buf, oob, 0)
        if err != nil { return }
        if oobn == 0 { continue }
        msgs, err := unix.ParseSocketControlMessage(oob[:oobn])
        if err != nil { continue }
        for _, m := range msgs {
            fds, err := unix.ParseUnixRights(&m)
            if err != nil { continue }
            for _, fd := range fds {
                // Set non-blocking — required by the Go runtime.
                if err := unix.SetNonblock(fd, true); err != nil {
                    unix.Close(fd)
                    continue
                }
                out <- fd
            }
        }
    }
}
```

To consume the fds in our raw HTTP server:

```go
fdCh, _ := fdpass.Listen(socketPath + ".ctrl")

// In the main accept loop:
for {
    select {
    case fd := <-fdCh:
        // Wrap as net.Conn — we can use Go's poller and connection lifecycle.
        f := os.NewFile(uintptr(fd), "client")
        c, err := net.FileConn(f)
        f.Close()  // os.NewFile dup'd; FileConn dup'd again. Close our copy.
        if err != nil { continue }
        go handle(c)
    case c := <-listenCh:
        // Normal accept path (fallback if LB falls back to TCP proxy).
        go handle(c)
    }
}
```

The `net.FileConn` step converts a raw fd to a `*net.UnixConn` (for UDS-passed fds) or `*net.TCPConn` (for TCP-passed fds). The Go runtime's netpoller integrates the fd into its goroutine-aware I/O dispatch. Performance-wise this matches a fresh `accept`-then-`handle` flow.

## Gotchas

A few things that bite the first time:

- **Set non-blocking explicitly**. The kernel-level fd inherits its blocking mode from the sender. Go's netpoller requires non-blocking fds; if you pass without `SetNonblock`, the read goroutine deadlocks on the first `read` syscall.

- **Close the LB's copy or you leak FDs**. The sender holds a copy of the fd until it `close`s. `so-no-forevis` does this correctly, but if you write your own LB, remember.

- **Ancillary buffer must use `CmsgSpace`**, not `CmsgLen`. `CmsgSpace(N)` accounts for kernel alignment; `CmsgLen(N)` is what the kernel writes. Using `CmsgLen` for the buffer size truncates and the kernel returns `EMSGSIZE`.

- **The control socket type must be SEQPACKET on Linux**. Some BSDs accept STREAM. SOCK_DGRAM doesn't preserve ancillary data on Linux. STREAM works but you have to do your own framing.

- **`os.NewFile` and `net.FileConn` each dup the fd**. So `f := os.NewFile(uintptr(fd), ...); c, _ := net.FileConn(f); f.Close()` ends up with `c` holding one fd, `f.Close()` releases another, and our caller is responsible for closing `c` when done. Don't double-close.

- **The LB's `connect()` to upstream `.ctrl` must be SEQPACKET too**. If the LB opens a STREAM connection, the upstream's SEQPACKET `accept` returns `EOPNOTSUPP`.

## What we're NOT doing

- **Writing our own SCM_RIGHTS LB**. The `jrblatt/so-no-forevis:v1.0.0` image is public; we just reference it in compose. Saves us from re-implementing the LB's accept loop, io_uring tuning, and SCM_RIGHTS sender protocol. We only have to implement the *receiver* side, which is what the lecture above covers.

- **Falling back to HAProxy if so-no-forevis fails**. If the fd-passing path fails, the upstream's normal UDS listener still accepts regular `connect` traffic. The LB doesn't fall back automatically, but the upstream is ready for either.

## What the kernel does under the hood

For curiosity: when sendmsg with SCM_RIGHTS is called, the kernel:

1. Walks the cmsg buffer, finds SCM_RIGHTS messages.
2. For each fd in the SCM_RIGHTS payload, calls `__fget(fd)` in the sender's task struct — gets a `struct file *` with refcount incremented.
3. Stores the `struct file *` in a per-message "passed_fds" array attached to the unix socket's skb (socket buffer).
4. Queues the skb on the receiver's socket.

When recvmsg is called:

1. Dequeues the skb.
2. Walks the passed_fds, calls `__fd_install(receiver_task, file *)` — allocates a new fd in the receiver's task struct pointing to the same `struct file *`.
3. Writes the new fd numbers into the receiver's ancillary cmsg buffer.

The `struct file *` is shared between sender and receiver until both close their copies. The TCP/IP stack doesn't care which task owns the fd; it operates on `struct sock *` referenced from `struct file *->private_data`.

This is how systemd's socket activation works, and how nginx pre-fork workers share a listening socket. Same kernel primitive.

## References

- `man 3 cmsg` — Linux ancillary data API.
- `unix(7)` — SOCK_SEQPACKET semantics on Linux.
- `golang.org/x/sys/unix` — Go bindings for sendmsg/recvmsg.
- `so-no-forevis` source — referenced in [lecture 08](08-top10-survey.md#scm-rights-fd-passing) survey of top submissions.
- Linux kernel `net/unix/scm.c` — the implementation of SCM_RIGHTS dispatch on a unix socket.
