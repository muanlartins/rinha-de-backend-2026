package api

import (
	"bytes"
	"errors"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeConn implements net.Conn with scripted reads and a write buffer.
// readChunks are returned one per Read call. After they're exhausted,
// Read returns io.EOF.
type fakeConn struct {
	mu         sync.Mutex
	readChunks [][]byte
	written    bytes.Buffer
	closed     bool
}

func (c *fakeConn) Read(b []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.readChunks) == 0 {
		return 0, io.EOF
	}
	chunk := c.readChunks[0]
	n := copy(b, chunk)
	if n < len(chunk) {
		c.readChunks[0] = chunk[n:]
	} else {
		c.readChunks = c.readChunks[1:]
	}
	return n, nil
}

func (c *fakeConn) Write(b []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, errors.New("closed")
	}
	return c.written.Write(b)
}

func (c *fakeConn) Close() error {
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
	return nil
}

func (c *fakeConn) LocalAddr() net.Addr                { return nil }
func (c *fakeConn) RemoteAddr() net.Addr               { return nil }
func (c *fakeConn) SetDeadline(t time.Time) error      { return nil }
func (c *fakeConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *fakeConn) SetWriteDeadline(t time.Time) error { return nil }

// makeRequest builds a minimal valid HTTP/1.1 request.
func makeRequest(path, body string) string {
	return "POST " + path + " HTTP/1.1\r\nHost: x\r\nContent-Length: " +
		itoaStr(len(body)) + "\r\n\r\n" + body
}

func itoaStr(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [16]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// TestHandleRawConn_SingleRequest is the minimal sanity check: one
// request → one response written.
func TestHandleRawConn_SingleRequest(t *testing.T) {
	body := `{"hello":"world"}`
	req := makeRequest("/ready", body)
	conn := &fakeConn{readChunks: [][]byte{[]byte(req)}}
	h := NewHandler()
	h.MarkReady()
	handleRawConn(conn, h)
	got := conn.written.String()
	if !strings.HasPrefix(got, "HTTP/1.1 ") {
		t.Fatalf("no HTTP response written: %q", got)
	}
	if strings.Count(got, "HTTP/1.1") != 1 {
		t.Errorf("expected exactly 1 response, got %d", strings.Count(got, "HTTP/1.1"))
	}
}

// TestHandleRawConn_PipelinedRequests is the regression test for
// Phase 35a: multiple requests arrive in one Read; we must handle them
// all without losing data or corrupting the parser. The previous
// memmove-per-request code passed this trivially; the new offset-
// based code must too.
func TestHandleRawConn_PipelinedRequests(t *testing.T) {
	body := `{"x":1}`
	req := makeRequest("/ready", body)
	// 5 requests glued into one buffer.
	all := req + req + req + req + req
	conn := &fakeConn{readChunks: [][]byte{[]byte(all)}}
	h := NewHandler()
	h.MarkReady()
	handleRawConn(conn, h)
	got := conn.written.String()
	if n := strings.Count(got, "HTTP/1.1"); n != 5 {
		t.Errorf("expected 5 responses, got %d (resp: %q)", n, got)
	}
}

// TestHandleRawConn_FragmentedRead checks the case where one request's
// bytes arrive across multiple Read() calls — common in real TCP.
func TestHandleRawConn_FragmentedRead(t *testing.T) {
	body := `{"y":2}`
	req := []byte(makeRequest("/ready", body))
	// Split into 4 chunks at arbitrary positions.
	chunks := [][]byte{
		req[:10],
		req[10:30],
		req[30:50],
		req[50:],
	}
	conn := &fakeConn{readChunks: chunks}
	h := NewHandler()
	h.MarkReady()
	handleRawConn(conn, h)
	got := conn.written.String()
	if n := strings.Count(got, "HTTP/1.1"); n != 1 {
		t.Errorf("expected 1 response, got %d (resp: %q)", n, got)
	}
}

// TestHandleRawConn_PipelinedAndFragmented mixes both patterns to
// stress the offset logic: first chunk has 2 full requests + start of
// 3rd; second chunk completes 3rd.
func TestHandleRawConn_PipelinedAndFragmented(t *testing.T) {
	body := `{"z":3}`
	req := makeRequest("/ready", body)
	combined := req + req + req
	split := len(req)*2 + 30
	conn := &fakeConn{
		readChunks: [][]byte{
			[]byte(combined[:split]),
			[]byte(combined[split:]),
		},
	}
	h := NewHandler()
	h.MarkReady()
	handleRawConn(conn, h)
	got := conn.written.String()
	if n := strings.Count(got, "HTTP/1.1"); n != 3 {
		t.Errorf("expected 3 responses, got %d (resp: %q)", n, got)
	}
}
