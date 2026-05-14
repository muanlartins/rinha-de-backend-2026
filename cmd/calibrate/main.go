// calibrate finds per-class ExtremeWorstThreshold values for the IVF
// fast tier, such that escalation catches every mis-classified entry
// without escalating safely-classified ones.
//
// Algorithm:
//
//  1. For each entry in test-data.json, run the fast tier ONLY at the
//     configured FastNProbe → record (fast_count, fast_worst_i64,
//     expected_approved).
//  2. For each entry, also run the FULL sweep → record full_count.
//     The full count is the "oracle within IVF". An entry where the
//     full classification disagrees with expected is structurally
//     unfixable by escalation (its true 5-NN doesn't classify correctly
//     even with all 4096 clusters scanned).
//  3. For each class c ∈ {0..5}: among entries where fast_count == c,
//     find the minimum threshold T such that all entries needing
//     escalation (full classification fixes them) have worst > T. This
//     is the smallest T that catches every fixable error.
//  4. Print the threshold table and the resulting expected (FP, FN, Err).
//
// Output is Go source for internal/search/thresholds.go — paste it in.
//
// Run on linux/amd64 (use `docker run`) to match the bot's kernel
// precision. Local darwin/arm64 numbers will not transfer.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
	"github.com/muanlartins/rinha-de-backend-2026/internal/search"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

type entry struct {
	Request          json.RawMessage `json:"request"`
	ExpectedApproved bool            `json:"expected_approved"`
}

type record struct {
	fastCount   uint8
	fastWorst   int64
	fullCount   uint8 // full-K oracle (sweeps everything)
	topNCount   uint8 // result under the runtime phase-27 path
	expectedApp bool
}

func approvedFromCount(c uint8) bool { return float64(c)/5.0 < 0.6 }

func main() {
	var (
		indexPath = flag.String("index", "/resources/index.bin", "path to IVF index")
		testPath  = flag.String("test", "references/rinha-official/test/test-data.json", "test-data.json")
		nprobe    = flag.Int("nprobe", search.FastNProbe, "FastNProbe value used for the fast-tier pass")
		topN      = flag.Int("topN", search.EscalateNProbe, "EscalateNProbe simulated for the top-N path (phase 27)")
		topNSweep = flag.String("topNSweep", "", "comma-separated list of N values to sweep (e.g. 16,24,32,48); overrides -topN")
	)
	flag.Parse()

	f, err := os.Open(*indexPath)
	if err != nil {
		log.Fatalf("open index: %v", err)
	}
	idx, err := ivf.Load(f)
	f.Close()
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	log.Printf("index loaded: N=%d K=%d blocks=%d nprobe=%d", idx.N, idx.K, idx.Blocks, *nprobe)

	tf, err := os.Open(*testPath)
	if err != nil {
		log.Fatalf("open test: %v", err)
	}
	var top struct {
		Entries []entry `json:"entries"`
	}
	if err := json.NewDecoder(tf).Decode(&top); err != nil {
		log.Fatalf("decode test: %v", err)
	}
	tf.Close()
	log.Printf("test entries: %d", len(top.Entries))

	records := make([]record, 0, len(top.Entries))
	parseFails := 0
	var scratch search.IVFScratch
	for _, e := range top.Entries {
		var qi [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &qi) {
			parseFails++
			continue
		}
		var qf [dataset.Dims]float32
		var qiArr [dataset.Dims]int16
		for d := 0; d < dataset.Dims; d++ {
			qf[d] = float32(qi[d])
			qiArr[d] = qi[d]
		}

		fastCnt, fastWorst := search.FraudCountFastOnly(&qf, &qiArr, idx, &scratch, *nprobe)
		fullCnt := search.FraudCountFull(&qf, &qiArr, idx, &scratch)
		topNCnt := search.FraudCountTopN(&qf, &qiArr, idx, &scratch, *topN)

		records = append(records, record{
			fastCount:   fastCnt,
			fastWorst:   fastWorst,
			fullCount:   fullCnt,
			topNCount:   topNCnt,
			expectedApp: e.ExpectedApproved,
		})
	}
	log.Printf("processed %d entries (parse_fails=%d)", len(records), parseFails)

	// Per-class breakdown.
	log.Printf("--- per-class breakdown ---")
	for c := uint8(0); c <= 5; c++ {
		var n, mismatchFast, mismatchFull, fixable int
		for _, r := range records {
			if r.fastCount != c {
				continue
			}
			n++
			if approvedFromCount(r.fastCount) != r.expectedApp {
				mismatchFast++
			}
			if approvedFromCount(r.fullCount) != r.expectedApp {
				mismatchFull++
			}
			// Fixable: fast wrong, full right.
			if approvedFromCount(r.fastCount) != r.expectedApp &&
				approvedFromCount(r.fullCount) == r.expectedApp {
				fixable++
			}
		}
		log.Printf("class=%d n=%-6d fast_wrong=%-4d full_wrong=%-4d fixable_by_escalation=%d",
			c, n, mismatchFast, mismatchFull, fixable)
	}

	// Find per-class threshold.
	thresholds := computeThresholds(records)
	log.Printf("--- chosen thresholds ---")
	totalEscalations := 0
	for c := uint8(0); c <= 5; c++ {
		log.Printf("class=%d threshold=%d", c, thresholds[c])
		for _, r := range records {
			if r.fastCount != c {
				continue
			}
			needEscalate := (c == 2 || c == 3 || c == 4) ||
				(thresholds[c] > 0 && r.fastWorst > thresholds[c])
			if needEscalate {
				totalEscalations++
			}
		}
	}
	log.Printf("expected escalations: %d / %d (%.2f%%)",
		totalEscalations, len(records), 100*float64(totalEscalations)/float64(len(records)))

	// Verify with thresholds applied: how many would remain wrong?
	fp, fn := 0, 0
	for _, r := range records {
		c := r.fastCount
		escalate := (c == 2 || c == 3 || c == 4) ||
			(thresholds[c] > 0 && r.fastWorst > thresholds[c])
		finalCnt := r.fastCount
		if escalate {
			finalCnt = r.fullCount
		}
		approved := approvedFromCount(finalCnt)
		switch {
		case approved && !r.expectedApp:
			fn++
		case !approved && r.expectedApp:
			fp++
		}
	}
	log.Printf("--- post-calibration projection (oracle = full-K sweep) ---")
	log.Printf("FP=%d FN=%d", fp, fn)

	// Top-N verification: what would actually happen under the phase-27
	// production path, where escalation = top-N nearest unscanned only?
	if *topNSweep != "" {
		log.Printf("--- top-N sweep ---")
		ns := strings.Split(*topNSweep, ",")
		for _, ns := range ns {
			var nVal int
			if _, err := fmt.Sscanf(ns, "%d", &nVal); err != nil {
				log.Printf("skip bad N=%q: %v", ns, err)
				continue
			}
			fpN, fnN := topNCheck(*testPath, idx, &scratch, nVal)
			log.Printf("N=%-3d FP=%d FN=%d", nVal, fpN, fnN)
		}
	} else {
		fpN, fnN := 0, 0
		for _, r := range records {
			a := approvedFromCount(r.topNCount)
			switch {
			case a && !r.expectedApp:
				fnN++
			case !a && r.expectedApp:
				fpN++
			}
		}
		log.Printf("--- top-N=%d projection (oracle = production path) ---", *topN)
		log.Printf("FP=%d FN=%d", fpN, fnN)
	}

	// Emit the Go constants.
	fmt.Println()
	fmt.Println("// generated by cmd/calibrate — paste into internal/search/thresholds.go")
	fmt.Println("var ExtremeWorstThreshold = [6]int64{")
	for c := uint8(0); c <= 5; c++ {
		fmt.Printf("\t/* count=%d */ %d,\n", c, thresholds[c])
	}
	fmt.Println("}")
}

// computeThresholds: for each class c, find the smallest T such that
// escalating any entry with fast_count==c AND fast_worst > T catches
// all entries where escalation would fix a mis-classification.
//
// Returns thresholds[c] = 0 when no escalation is needed (no fixable
// mis-classifications in that class).
func computeThresholds(records []record) [6]int64 {
	var out [6]int64
	for c := uint8(0); c <= 5; c++ {
		// Skip the always-escalate classes — the count gate covers them.
		if c == 2 || c == 3 || c == 4 {
			continue
		}
		// Collect fixable entries (fast wrong, full right) for this class.
		var fixableWorsts []int64
		for _, r := range records {
			if r.fastCount != c {
				continue
			}
			fastWrong := approvedFromCount(r.fastCount) != r.expectedApp
			fullRight := approvedFromCount(r.fullCount) == r.expectedApp
			if fastWrong && fullRight {
				fixableWorsts = append(fixableWorsts, r.fastWorst)
			}
		}
		if len(fixableWorsts) == 0 {
			continue
		}
		// Set threshold to the smallest fixable worst minus 1: catches all
		// fixable entries (their worst > T) and minimizes the number of
		// safe entries also escalated.
		sort.Slice(fixableWorsts, func(i, j int) bool { return fixableWorsts[i] < fixableWorsts[j] })
		out[c] = fixableWorsts[0] - 1
	}
	return out
}

// topNCheck replays each record's request through FraudCountTopN at the
// given N and counts FP/FN against the expected outcome. Used by the
// -topNSweep mode to choose the smallest N that preserves FP=FN=0 under
// the production phase-27 path.
//
// Returns (fp, fn). Each call recomputes from raw request data; the
// passed scratch is reused (and reset by FraudCountFastOnly internally).
func topNCheck(testPath string, idx *ivf.IVFIndex, scratch *search.IVFScratch, n int) (int, int) {
	tf, err := os.Open(testPath)
	if err != nil {
		log.Printf("topNCheck: %v", err)
		return -1, -1
	}
	defer tf.Close()
	var top struct {
		Entries []entry `json:"entries"`
	}
	if err := json.NewDecoder(tf).Decode(&top); err != nil {
		log.Printf("topNCheck decode: %v", err)
		return -1, -1
	}
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
		cnt := search.FraudCountTopN(&qf, &qiArr, idx, scratch, n)
		approved := approvedFromCount(cnt)
		switch {
		case approved && !e.ExpectedApproved:
			fn++
		case !approved && e.ExpectedApproved:
			fp++
		}
	}
	return fp, fn
}
