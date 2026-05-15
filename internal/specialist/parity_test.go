package specialist

import (
	"encoding/json"
	"os"
	"sync"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
	"github.com/muanlartins/rinha-de-backend-2026/internal/search"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

var (
	loadOnce    sync.Once
	sharedDS    *dataset.Dataset
	sharedSpec  *SpecialistIndex
	sharedIVF   *ivf.IVFIndex
)

func loadFixtures(tb testing.TB) {
	loadOnce.Do(func() {
		path := "../../references/rinha-official/resources/references.json.gz"
		if _, err := os.Stat(path); err != nil {
			return
		}
		ds, err := dataset.LoadFromGzipJSON(path)
		if err != nil {
			tb.Fatalf("load dataset: %v", err)
		}
		sharedDS = ds
		// Build specialist index.
		sharedSpec = Build(ds)
		tb.Logf("specialist: N=%d parts=%d nodes=%d blocks=%d",
			sharedSpec.N, sharedSpec.PartCount, sharedSpec.NodeCount, sharedSpec.BlockCount)

		// Build IVF for cross-check.
		centroids := ivf.TrainKMeans(ds.Vectors, ds.Count)
		assign := ivf.AssignAll(ds.Vectors, ds.Count, &centroids)
		ivf.SetLabelSource(ds.Labels)
		idx, err := ivf.Build(ds.Vectors, ds.Count, &centroids, assign)
		if err != nil {
			tb.Fatalf("build ivf: %v", err)
		}
		sharedIVF = idx
	})
	if sharedDS == nil {
		tb.Skip("references unavailable")
	}
}

// TestSpecialistFP_FN runs every entry in test-data.json through the
// specialist search and reports FP / FN counts against expected_approved.
// This is the strict correctness gate before shipping. FP+FN must be 0.
func TestSpecialistFP_FN(t *testing.T) {
	loadFixtures(t)
	if sharedSpec == nil {
		return
	}

	f, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		t.Skipf("test-data unavailable: %v", err)
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
		t.Fatalf("decode: %v", err)
	}

	var sc SearchScratch
	fp, fn := 0, 0
	for _, e := range top.Entries {
		var qi [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &qi) {
			continue
		}
		var qf [dataset.Dims]float32
		var qiArr [dataset.Dims]int16
		for d := 0; d < dataset.Dims; d++ {
			qf[d] = float32(qi[d])
			qiArr[d] = qi[d]
		}
		count := FraudCount(&qf, &qiArr, sharedSpec, &sc)
		approved := (float64(count) / 5.0) < 0.6
		switch {
		case approved && !e.ExpectedApproved:
			fn++
		case !approved && e.ExpectedApproved:
			fp++
		}
	}
	t.Logf("specialist full set: FP=%d FN=%d", fp, fn)
	if fp+fn > 0 {
		t.Errorf("expected FP=FN=0, got FP=%d FN=%d", fp, fn)
	}
}

// TestSpecialistVsIVF compares specialist's classification against IVF's
// on a stride sample of test-data.json. Their fraud_count may differ
// on ties (different top-5 sets) but the approved/denied classification
// must agree. Disagreement indicates one of the algorithms is missing
// a true neighbor; we expect specialist to be at least as accurate as
// IVF (since both are exact within their search scope and specialist
// has finer-grained pruning).
func TestSpecialistVsIVF(t *testing.T) {
	loadFixtures(t)
	if sharedSpec == nil || sharedIVF == nil {
		return
	}

	f, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		t.Skipf("test-data unavailable: %v", err)
	}
	defer f.Close()
	var top struct {
		Entries []struct {
			Request json.RawMessage `json:"request"`
		} `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&top); err != nil {
		t.Fatalf("decode: %v", err)
	}

	step := 50 // ~1k samples; long tests run unsampled
	if testing.Verbose() {
		step = 5
	}

	var specSc SearchScratch
	var ivfSc search.IVFScratch
	classifMismatch := 0
	countMismatch := 0
	for i := 0; i < len(top.Entries); i += step {
		var qi [dataset.Stride]int16
		if !vector.VectorizeFast(top.Entries[i].Request, &qi) {
			continue
		}
		var qf [dataset.Dims]float32
		var qiArr [dataset.Dims]int16
		for d := 0; d < dataset.Dims; d++ {
			qf[d] = float32(qi[d])
			qiArr[d] = qi[d]
		}
		specCount := FraudCount(&qf, &qiArr, sharedSpec, &specSc)
		ivfCount := search.FraudCountIVF(&qf, &qiArr, sharedIVF, &ivfSc)
		if specCount != ivfCount {
			countMismatch++
			specApproved := (float64(specCount) / 5.0) < 0.6
			ivfApproved := (float64(ivfCount) / 5.0) < 0.6
			if specApproved != ivfApproved {
				classifMismatch++
				if classifMismatch <= 5 {
					t.Logf("entry %d: spec=%d ivf=%d (CLASS MISMATCH)", i, specCount, ivfCount)
				}
			}
		}
	}
	t.Logf("step=%d count_mismatch=%d class_mismatch=%d", step, countMismatch, classifMismatch)
	if classifMismatch > 0 {
		t.Errorf("specialist classification differs from IVF on %d entries", classifMismatch)
	}
}
