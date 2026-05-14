package api

import (
	"bytes"
	"io"
	"net"
	"strings"
	"testing"
	"time"
)

// pipelinedConn replays a fixed buffer of N pipelined requests for the
// duration of the bench, looping at EOF. We measure throughput
// (requests/sec) rather than wall-time per call to amortize startup.
type pipelinedConn struct {
	full   []byte
	rd     int
	target int
	served int
	out    bytes.Buffer
}

func (c *pipelinedConn) Read(b []byte) (int, error) {
	if c.served >= c.target {
		return 0, io.EOF
	}
	if c.rd == len(c.full) {
		c.rd = 0
	}
	n := copy(b, c.full[c.rd:])
	c.rd += n
	return n, nil
}

func (c *pipelinedConn) Write(b []byte) (int, error) {
	c.served += bytes.Count(b, []byte("HTTP/1.1"))
	c.out.Write(b)
	return len(b), nil
}

func (c *pipelinedConn) Close() error                       { return nil }
func (c *pipelinedConn) LocalAddr() net.Addr                { return nil }
func (c *pipelinedConn) RemoteAddr() net.Addr               { return nil }
func (c *pipelinedConn) SetDeadline(t time.Time) error      { return nil }
func (c *pipelinedConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *pipelinedConn) SetWriteDeadline(t time.Time) error { return nil }

// BenchmarkHandleRawConn_Pipelined measures per-request cost under
// keep-alive pipelining with /ready (no IVF work). Used to isolate
// the HTTP/handler shim cost.
func BenchmarkHandleRawConn_Pipelined(b *testing.B) {
	body := `{"hello":"world"}`
	req := makeRequest("/ready", body)
	all := strings.Repeat(req, 64)

	h := NewHandler()
	h.MarkReady()

	b.ReportAllocs()
	b.ResetTimer()
	conn := &pipelinedConn{full: []byte(all), target: b.N}
	handleRawConn(conn, h)
	if conn.served < b.N {
		b.Fatalf("served %d, target %d", conn.served, b.N)
	}
}

// BenchmarkHandleRawConn_FraudPipelined exercises the actual production
// hot path: pipelined POST /fraud-score on a keep-alive conn, going
// through the load shedder. Designed to catch per-request allocations
// like time.After (Phase 35b).
//
// Index is NOT loaded (h.idx is nil) — the fraudScoreRaw early-returns
// fraudResponses[0]; this still exercises the shedder + parse path
// without the IVF cost dominating.
func BenchmarkHandleRawConn_FraudPipelined(b *testing.B) {
	// A realistic body from the test data is ~600 bytes. Use a stub
	// here; the parser will fail and we'll return fraudResponses[0],
	// but only AFTER passing through the shedder — which is what we
	// want to measure.
	body := `{"transaction":{"amount":100.0,"installments":1,"requested_at":"2026-03-15T10:00:00Z"}}`
	req := makeRequest("/fraud-score", body)
	all := strings.Repeat(req, 64)

	h := NewHandler()
	h.MarkReady()

	b.ReportAllocs()
	b.ResetTimer()
	conn := &pipelinedConn{full: []byte(all), target: b.N}
	handleRawConn(conn, h)
	if conn.served < b.N {
		b.Fatalf("served %d, target %d", conn.served, b.N)
	}
}
