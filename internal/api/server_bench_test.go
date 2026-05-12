package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// loadDataset returns a Dataset built from the local mirror, or nil if it
// isn't available (CI / fresh checkout without the references submodule).
func loadDataset(tb testing.TB) *dataset.Dataset {
	tb.Helper()
	path := "../../references/rinha-official/resources/references.json.gz"
	if _, err := os.Stat(path); err != nil {
		tb.Skipf("references unavailable (%v)", err)
		return nil
	}
	ds, err := dataset.LoadFromGzipJSON(path)
	if err != nil {
		tb.Fatalf("load: %v", err)
	}
	ds.Partition()
	ds.BuildGrid()
	return ds
}

// loadHandlerPayloads pulls request bodies out of the test-data file. Only
// for benchmarking; we never look at the expected labels.
func loadHandlerPayloads(tb testing.TB, n int) [][]byte {
	tb.Helper()
	path := "../../references/rinha-official/test/test-data.json"
	f, err := os.Open(path)
	if err != nil {
		tb.Skipf("test data unavailable (%v)", err)
		return nil
	}
	defer f.Close()
	var top struct {
		Entries []struct {
			Request json.RawMessage `json:"request"`
		} `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&top); err != nil {
		tb.Fatalf("decode: %v", err)
	}
	if n > len(top.Entries) {
		n = len(top.Entries)
	}
	out := make([][]byte, n)
	for i := 0; i < n; i++ {
		out[i] = append([]byte(nil), top.Entries[i].Request...)
	}
	return out
}

// BenchmarkHandlerFraudScore runs the full /fraud-score handler end-to-end:
// body read, fast parse, grid search, write response. Used both as a perf
// number and for `-cpuprofile` capture to feed PGO.
func BenchmarkHandlerFraudScore(b *testing.B) {
	ds := loadDataset(b)
	if ds == nil {
		return
	}
	payloads := loadHandlerPayloads(b, 1024)
	if payloads == nil {
		return
	}

	h := NewHandler()
	h.SetDataset(ds)

	w := httptest.NewRecorder()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		body := payloads[i&1023]
		req := httptest.NewRequest("POST", "/fraud-score", bytes.NewReader(body))
		// httptest's NewRecorder zeroes between calls; reuse the body buffer
		// by resetting the recorder body.
		w.Body = nil
		h.ServeHTTP(w, req)
		_, _ = io.Copy(io.Discard, w.Result().Body)
	}
}
