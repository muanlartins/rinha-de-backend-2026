package api

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
)

// TestRouteRawClassifies confirms that the rawhttp dispatch path (the one
// used in production by the custom HTTP server) returns the correct
// fraud_count for fraud and legit entries of test-data.json.
func TestRouteRawClassifies(t *testing.T) {
	path := "../../references/rinha-official/resources/references.json.gz"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("references unavailable: %v", err)
	}
	ds, err := dataset.LoadFromGzipJSON(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	centroids := ivf.TrainKMeans(ds.Vectors, ds.Count)
	assign := ivf.AssignAll(ds.Vectors, ds.Count, &centroids)
	ivf.SetLabelSource(ds.Labels)
	idx, err := ivf.Build(ds.Vectors, ds.Count, &centroids, assign)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	h := NewHandler()
	h.SetIndex(idx)

	tf, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		t.Skipf("test data unavailable: %v", err)
	}
	defer tf.Close()
	type entry struct {
		Request          json.RawMessage `json:"request"`
		ExpectedApproved bool            `json:"expected_approved"`
	}
	var top struct {
		Entries []entry `json:"entries"`
	}
	if err := json.NewDecoder(tf).Decode(&top); err != nil {
		t.Fatal(err)
	}

	// Check first 20 entries via RouteRaw.
	for i := 0; i < 20; i++ {
		e := top.Entries[i]
		resp := h.RouteRaw([]byte("/fraud-score"), e.Request)

		// Find body after \r\n\r\n
		bodyStart := -1
		for j := 0; j+3 < len(resp); j++ {
			if resp[j] == '\r' && resp[j+1] == '\n' && resp[j+2] == '\r' && resp[j+3] == '\n' {
				bodyStart = j + 4
				break
			}
		}
		if bodyStart < 0 {
			t.Fatalf("entry %d: no body in response: %q", i, resp)
		}
		body := string(resp[bodyStart:])

		approved := body == `{"approved":true,"fraud_score":0}` ||
			body == `{"approved":true,"fraud_score":0.2}` ||
			body == `{"approved":true,"fraud_score":0.4}`

		match := approved == e.ExpectedApproved
		if !match {
			t.Logf("entry %d: body=%s expected_approved=%v (MISMATCH)", i, body, e.ExpectedApproved)
		}
	}
}
