package search

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

// TestFullDatasetMismatches scans every test-data.json entry, runs the
// production IVF, compares to the labeled expected_approved.
func TestFullDatasetMismatches(t *testing.T) {
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

	parseFails := 0
	fpCount, fnCount := 0, 0
	for i, e := range top.Entries {
		var q [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &q) {
			parseFails++
			continue
		}
		fc := FraudCountIVF(&q, ds)
		gotApproved := (float64(fc)/5.0) < 0.6
		if gotApproved != e.ExpectedApproved {
			if e.ExpectedApproved && !gotApproved {
				fpCount++ // expected approve, we denied
			} else {
				fnCount++ // expected deny, we approved
			}
			if fpCount+fnCount <= 5 {
				t.Logf("entry %d MISMATCH ourFraudCount=%d (approved=%v) expected_approved=%v",
					i, fc, gotApproved, e.ExpectedApproved)
			}
		}
	}
	t.Logf("checked %d entries: parse_fails=%d FP=%d FN=%d",
		len(top.Entries), parseFails, fpCount, fnCount)
}
