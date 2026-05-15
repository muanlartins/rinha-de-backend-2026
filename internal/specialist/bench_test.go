package specialist

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/search"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

// BenchmarkSpecialistFraudCount runs the specialist search through
// 1024 real test-data.json queries in a tight loop. Per-iteration cost
// includes vectorize + search. Compared head-to-head with the IVF
// benchmark below to see whether specialist is faster end-to-end.
func BenchmarkSpecialistFraudCount(b *testing.B) {
	loadFixtures(b)
	if sharedSpec == nil {
		return
	}
	queries := loadQueriesBench(b, 1024)
	if queries == nil {
		return
	}

	var sc SearchScratch
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qi := &queries[i&1023].v
		var qf [dataset.Dims]float32
		var qiArr [dataset.Dims]int16
		for d := 0; d < dataset.Dims; d++ {
			qf[d] = float32(qi[d])
			qiArr[d] = qi[d]
		}
		_ = FraudCount(&qf, &qiArr, sharedSpec, &sc)
	}
}

func BenchmarkIVFFraudCount(b *testing.B) {
	loadFixtures(b)
	if sharedIVF == nil {
		return
	}
	queries := loadQueriesBench(b, 1024)
	if queries == nil {
		return
	}

	var sc search.IVFScratch
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qi := &queries[i&1023].v
		var qf [dataset.Dims]float32
		var qiArr [dataset.Dims]int16
		for d := 0; d < dataset.Dims; d++ {
			qf[d] = float32(qi[d])
			qiArr[d] = qi[d]
		}
		_ = search.FraudCountIVF(&qf, &qiArr, sharedIVF, &sc)
	}
}

func loadQueriesBench(tb testing.TB, n int) []struct {
	v [dataset.Dims]int16
} {
	tb.Helper()
	f, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		tb.Skipf("test data unavailable: %v", err)
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
	out := make([]struct {
		v [dataset.Dims]int16
	}, n)
	for i := 0; i < n; i++ {
		var qi [dataset.Stride]int16
		if !vector.VectorizeFast(top.Entries[i%len(top.Entries)].Request, &qi) {
			tb.Fatalf("vectorize fail at %d", i)
		}
		for d := 0; d < dataset.Dims; d++ {
			out[i].v[d] = qi[d]
		}
	}
	return out
}
