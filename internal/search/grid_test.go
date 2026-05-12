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
		ds.BuildGrid()
		sharedDS = ds
	})
	if sharedDS == nil {
		tb.Skip("references unavailable")
	}
	return sharedDS
}

// TestGridMatchesBrute: the grid algorithm with LB pruning is exact, so
// for every payload the fraud_count must equal brute-force. Any
// divergence is a bug (skipping a cell the LB shouldn't have skipped, or
// an early-exit kernel rounding error).
func TestGridMatchesBrute(t *testing.T) {
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

	// Subsample to keep the test under a minute. 1000 random-but-deterministic
	// payloads is enough to catch any LB pruning bug — there's no payload that
	// "rarely" misbehaves; either the algorithm is exact or it isn't.
	const sampleCap = 1000
	step := 1
	if len(top.Entries) > sampleCap {
		step = len(top.Entries) / sampleCap
	}

	mismatch := 0
	for i := 0; i < len(top.Entries); i += step {
		var q [dataset.Stride]int16
		if !vector.VectorizeFast(top.Entries[i].Request, &q) {
			t.Fatalf("entry %d: parse failed", i)
		}
		grid := FraudCount(&q, ds)
		brute := FraudCountBrute(&q, ds)
		if grid != brute {
			mismatch++
			if mismatch <= 10 {
				t.Logf("entry %d: grid=%d brute=%d", i, grid, brute)
			}
		}
	}
	if mismatch > 0 {
		t.Fatalf("grid disagrees with brute on %d of %d entries — algorithm not exact",
			mismatch, len(top.Entries)/step)
	}
}

// TestGridFullPayloadSet: optional, runs over the full 54k entries to
// confirm the FP/FN count vs test labels. Skipped by default (long-running).
func TestGridFullDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("long-running; run with -run TestGridFullDataset")
	}
	ds := loadDS(t)
	if ds == nil {
		return
	}

	f, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		t.Skip("test data unavailable")
	}
	defer f.Close()

	type entry struct {
		Request          json.RawMessage `json:"request"`
		ExpectedApproved bool            `json:"expected_approved"`
	}
	var top struct {
		Entries []entry `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&top); err != nil {
		t.Fatal(err)
	}

	parseFails, fp, fn := 0, 0, 0
	for _, e := range top.Entries {
		var q [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &q) {
			parseFails++
			continue
		}
		fc := FraudCount(&q, ds)
		approved := (float64(fc) / 5.0) < 0.6
		switch {
		case approved && !e.ExpectedApproved:
			fn++ // we approved a fraud
		case !approved && e.ExpectedApproved:
			fp++ // we denied a legit
		}
	}
	t.Logf("%d entries: parse_fails=%d FP=%d FN=%d", len(top.Entries), parseFails, fp, fn)
}
