package search

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

// TestFullDatasetMismatches scans every test-data.json entry, runs the
// production IVF, compares to the labeled expected_approved, and reports
// every mismatch + every parse failure. The goal: identify the single FN
// and the single Err we see consistently on the rinha bot.
func TestFullDatasetMismatches(t *testing.T) {
	if testing.Short() {
		t.Skip("long; run with -short=false")
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
		ExpectedScore    float64         `json:"expected_fraud_score"`
	}
	var top struct {
		Entries []entry `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&top); err != nil {
		t.Fatal(err)
	}

	parseFails := 0
	mismatches := 0
	fpCount, fnCount := 0, 0
	for i, e := range top.Entries {
		var q [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &q) {
			parseFails++
			if parseFails <= 3 {
				t.Logf("parse fail entry %d: %s", i, e.Request)
			}
			continue
		}
		fc := FraudCountIVF(&q, ds)
		gotApproved := (float64(fc)/5.0) < 0.6
		if gotApproved != e.ExpectedApproved {
			mismatches++
			if e.ExpectedApproved && !gotApproved {
				fpCount++ // expected approve, we denied
			} else {
				fnCount++ // expected deny, we approved
			}
			if mismatches <= 5 {
				t.Logf("entry %d MISMATCH ourFraudCount=%d (approved=%v) expected_approved=%v expected_score=%v",
					i, fc, gotApproved, e.ExpectedApproved, e.ExpectedScore)
				t.Logf("  body=%s", e.Request)
			}
		}
	}
	t.Logf("checked %d entries:\n  parse fails: %d\n  total mismatches: %d (FP=%d, FN=%d)",
		len(top.Entries), parseFails, mismatches, fpCount, fnCount)
}
