package search

import (
	"encoding/json"
	"os"
	"sync"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

var (
	loadOnce sync.Once
	sharedDS *dataset.Dataset
)

func loadDS(tb testing.TB) *dataset.Dataset {
	loadOnce.Do(func() {
		path := "../../references/rinha-official/resources/references.json.gz"
		if _, err := os.Stat(path); err != nil {
			return
		}
		ds, err := dataset.LoadFromGzipJSON(path)
		if err != nil {
			tb.Fatalf("load: %v", err)
		}
		ds.Partition()
		ds.BuildIVF()
		sharedDS = ds
	})
	if sharedDS == nil {
		tb.Skip("references unavailable")
	}
	return sharedDS
}

// TestIVFMatchesBruteForce: verify IVF gives the same fraud count as exact
// brute force on a sample of test payloads. They should agree on at least
// ~99% — the IVF is approximate (probes top clusters by centroid distance
// then LB-prunes the rest, can miss when the true nearest is in a cluster
// whose bbox happens to skip the LB filter under tight thresholds).
func TestIVFMatchesBruteForce(t *testing.T) {
	ds := loadDS(t)
	if ds == nil {
		return
	}

	f, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		t.Skip("test data unavailable")
		return
	}
	defer f.Close()

	var top struct {
		Entries []struct {
			Request json.RawMessage `json:"request"`
		} `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&top); err != nil {
		t.Fatal(err)
	}

	const sampleSize = 2000
	if len(top.Entries) > sampleSize {
		top.Entries = top.Entries[:sampleSize]
	}

	mismatch := 0
	for i, entry := range top.Entries {
		var q [dataset.Stride]int16
		if !vector.VectorizeFast(entry.Request, &q) {
			t.Fatalf("entry %d: parse failed", i)
		}
		ivf := FraudCountIVF(&q, ds)
		bf := FraudCount(&q, ds)
		if ivf != bf {
			mismatch++
			if mismatch <= 5 {
				t.Logf("entry %d: ivf=%d bruteforce=%d", i, ivf, bf)
			}
		}
	}
	rate := float64(mismatch) / float64(len(top.Entries))
	t.Logf("checked %d entries, %d mismatches (%.2f%%)", len(top.Entries), mismatch, rate*100)
	if rate > 0.05 {
		t.Fatalf("IVF / bruteforce disagreement rate %.2f%% > 5%%", rate*100)
	}
}
