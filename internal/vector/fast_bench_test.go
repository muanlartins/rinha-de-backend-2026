package vector

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

func loadSamplePayloads(b *testing.B, n int) [][]byte {
	b.Helper()
	f, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		b.Skipf("test data unavailable (%v)", err)
		return nil
	}
	defer f.Close()
	var top struct {
		Entries []struct {
			Request json.RawMessage `json:"request"`
		} `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&top); err != nil {
		b.Fatalf("decode: %v", err)
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

func BenchmarkVectorizeFast(b *testing.B) {
	payloads := loadSamplePayloads(b, 1024)
	if payloads == nil {
		return
	}
	var out [dataset.Stride]int16
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = VectorizeFast(payloads[i&1023], &out)
	}
}

func BenchmarkVectorizeSlow(b *testing.B) {
	payloads := loadSamplePayloads(b, 1024)
	if payloads == nil {
		return
	}
	var out [dataset.Stride]int16
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = vectorizeSlow(payloads[i&1023], &out)
	}
}
