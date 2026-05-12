package api

import (
	"net"
	"os"
	"sync"
)

const (
	maxRequestSize = 8 * 1024
	readBufSize    = 4 * 1024
)

// rawhttpResponses[count] is the FULL HTTP/1.1 response — status line +
// headers + body — for fraud_count ∈ 0..5. Pre-built once at startup so
// the request loop just writes the bytes to the socket.
var rawhttpResponses [6][]byte

// readyOK is the response for GET /ready when the dataset is loaded.
var readyOK = []byte("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

// readyNotYet is the 503 for GET /ready while loading.
var readyNotYet = []byte("HTTP/1.1 503 Service Unavailable\r\nContent-Length: 9\r\n\r\nnot ready")

// methodNotAllowed for non-POST on /fraud-score.
var methodNotAllowed = []byte("HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")

// notFound for unknown paths.
var notFound = []byte("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")

func init() {
	bodies := [6]string{
		`{"approved":true,"fraud_score":0}`,
		`{"approved":true,"fraud_score":0.2}`,
		`{"approved":true,"fraud_score":0.4}`,
		`{"approved":false,"fraud_score":0.6}`,
		`{"approved":false,"fraud_score":0.8}`,
		`{"approved":false,"fraud_score":1}`,
	}
	for i, b := range bodies {
		rawhttpResponses[i] = buildResp(b)
	}
}

func buildResp(body string) []byte {
	clen := itoa(len(body))
	out := make([]byte, 0, 64+len(body))
	out = append(out, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "...)
	out = append(out, clen...)
	out = append(out, "\r\n\r\n"...)
	out = append(out, body...)
	return out
}

func itoa(n int) []byte {
	if n == 0 {
		return []byte{'0'}
	}
	var buf [16]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return append([]byte(nil), buf[i:]...)
}

var rawReadBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, readBufSize)
		return &b
	},
}

// ListenRaw starts a custom HTTP/1.1 server on the Unix socket at path.
// Each new connection runs handleRawConn in its own goroutine, which loops
// on keep-alive (one socket → many requests).
func ListenRaw(path string, h *Handler) error {
	_ = os.Remove(path)
	ln, err := net.Listen("unix", path)
	if err != nil {
		return err
	}
	_ = os.Chmod(path, 0o666)
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go handleRawConn(conn, h)
		}
	}()
	return nil
}

// ServeFDChannel adopts fds received over an SCM_RIGHTS control channel and
// runs handleRawConn on each, exactly like the local accept path. fdCh is
// closed by the fdpass listener on shutdown; this function returns when
// the channel closes.
//
// The fd is wrapped via os.NewFile + net.FileConn. The os.File is closed
// after FileConn returns; the kernel keeps the underlying socket alive via
// the net.Conn's own refcount. See docs/lectures/12-scm-rights.md.
func ServeFDChannel(fdCh <-chan int, h *Handler) {
	for fd := range fdCh {
		f := os.NewFile(uintptr(fd), "scm-fd")
		c, err := net.FileConn(f)
		_ = f.Close()
		if err != nil {
			continue
		}
		go handleRawConn(c, h)
	}
}

func handleRawConn(conn net.Conn, h *Handler) {
	defer conn.Close()
	bufRef := rawReadBufPool.Get().(*[]byte)
	buf := *bufRef
	used := 0
	defer func() {
		if cap(buf) <= maxRequestSize {
			*bufRef = buf[:cap(buf)]
			rawReadBufPool.Put(bufRef)
		}
	}()

	for {
		var headEnd int
		for {
			if used >= len(buf) {
				if used >= maxRequestSize {
					return
				}
				nb := make([]byte, len(buf)*2)
				copy(nb, buf[:used])
				buf = nb
			}
			n, err := conn.Read(buf[used:])
			if n > 0 {
				used += n
				if idx := indexHeaderEnd(buf[:used]); idx >= 0 {
					headEnd = idx + 4
					break
				}
			}
			if err != nil {
				return
			}
		}

		path, contentLen := parseRequestLine(buf[:headEnd])
		if contentLen > maxRequestSize-headEnd {
			return
		}
		bodyEnd := headEnd + contentLen
		for used < bodyEnd {
			if used >= len(buf) {
				nb := make([]byte, len(buf)*2)
				copy(nb, buf[:used])
				buf = nb
			}
			n, err := conn.Read(buf[used:])
			if n > 0 {
				used += n
			}
			if err != nil {
				return
			}
		}

		resp := h.RouteRaw(path, buf[headEnd:bodyEnd])
		if _, err := conn.Write(resp); err != nil {
			return
		}

		copy(buf, buf[bodyEnd:used])
		used -= bodyEnd
	}
}

func indexHeaderEnd(b []byte) int {
	for i := 0; i+3 < len(b); i++ {
		if b[i] == '\r' && b[i+1] == '\n' && b[i+2] == '\r' && b[i+3] == '\n' {
			return i
		}
	}
	return -1
}

// parseRequestLine extracts the request path and Content-Length. We only
// need these two; everything else gets ignored.
func parseRequestLine(buf []byte) (path []byte, contentLen int) {
	i := 0
	for i < len(buf) && buf[i] != ' ' {
		i++
	}
	i++
	pathStart := i
	for i < len(buf) && buf[i] != ' ' {
		i++
	}
	path = buf[pathStart:i]
	contentLen = findContentLength(buf)
	return
}

func findContentLength(buf []byte) int {
	for i := 0; i+16 < len(buf); i++ {
		if (buf[i] == 'C' || buf[i] == 'c') && isContentLengthPrefix(buf[i:]) {
			j := i + 16
			for j < len(buf) && buf[j] == ' ' {
				j++
			}
			n := 0
			for j < len(buf) && buf[j] >= '0' && buf[j] <= '9' {
				n = n*10 + int(buf[j]-'0')
				j++
			}
			return n
		}
	}
	return 0
}

func isContentLengthPrefix(b []byte) bool {
	const name = "content-length: "
	if len(b) < len(name) {
		return false
	}
	for i := 0; i < 15; i++ {
		c := b[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != name[i] {
			return false
		}
	}
	return true
}
