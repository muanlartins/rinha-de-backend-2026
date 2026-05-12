package api

import (
	"fmt"
	"io"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/search"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

// Pooled 2 KB read buffer. Spec bodies are 500–700 bytes; io.ReadAll would
// grow-and-allocate per request.
var bodyBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 2048)
		return &b
	},
}

func readBodyInto(r io.Reader, buf []byte) ([]byte, error) {
	buf = buf[:cap(buf)]
	total := 0
	for {
		if total == len(buf) {
			grown := make([]byte, len(buf)*2)
			copy(grown, buf[:total])
			buf = grown
		}
		n, err := r.Read(buf[total:])
		total += n
		if err == io.EOF {
			return buf[:total], nil
		}
		if err != nil {
			return buf[:total], err
		}
	}
}

// fraudResponses[count] is the pre-built body for fraud_count ∈ 0..5
// (fraud_score = count/5, approved = fraud_score < 0.6).
var fraudResponses = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1}`),
}

type Handler struct {
	ready atomic.Bool
	ds    atomic.Pointer[dataset.Dataset]
}

func NewHandler() *Handler {
	return &Handler{}
}

func (h *Handler) SetDataset(ds *dataset.Dataset) {
	h.ds.Store(ds)
	h.ready.Store(true)
}

// MarkReady flips ready without a dataset for the smoke-only path (no
// /resources mount). /fraud-score then returns a fixed stub.
func (h *Handler) MarkReady() {
	h.ready.Store(true)
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/ready":
		h.handleReady(w, r)
	case "/fraud-score":
		h.handleFraudScore(w, r)
	case "/debug/info":
		h.handleDebugInfo(w, r)
	default:
		http.NotFound(w, r)
	}
}

// handleDebugInfo reports runtime state — used to verify on the rinha test
// env that the SIMD path is active, the dataset is loaded, and heap is
// where we expect.
func (h *Handler) handleDebugInfo(w http.ResponseWriter, _ *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	ds := h.ds.Load()
	var count int
	if ds != nil {
		count = ds.Count
	}
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w,
		`{"useAVX2":%t,"dataset_count":%d,"ready":%t,"heap_inuse_mb":%d,"alloc_total_mb":%d,"goarch":"%s","goos":"%s","gomaxprocs":%d}`,
		search.UseAVX2(),
		count,
		h.ready.Load(),
		m.HeapInuse/(1<<20),
		m.TotalAlloc/(1<<20),
		runtime.GOARCH,
		runtime.GOOS,
		runtime.GOMAXPROCS(0),
	)
}

func (h *Handler) handleReady(w http.ResponseWriter, _ *http.Request) {
	if h.ready.Load() {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	_, _ = w.Write([]byte("not ready"))
}

func (h *Handler) handleFraudScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !h.ready.Load() {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}

	bufp := bodyBufPool.Get().(*[]byte)
	body, err := readBodyInto(r.Body, *bufp)
	if err != nil {
		*bufp = body[:0]
		bodyBufPool.Put(bufp)
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer func() {
		*bufp = body[:0]
		bodyBufPool.Put(bufp)
	}()

	ds := h.ds.Load()
	if ds == nil {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(fraudResponses[0])
		return
	}

	var query [dataset.Stride]int16
	if !vector.VectorizeFast(body, &query) {
		// HTTP 5xx weighs 5 in E; a misclassification weighs 1 or 3. On a
		// parse miss, prefer approved=true / score=0 over a 5xx.
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(fraudResponses[0])
		return
	}

	frauds := search.FraudCountGrid(&query, ds)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(fraudResponses[frauds])
}
