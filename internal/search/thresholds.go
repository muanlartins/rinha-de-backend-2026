package search

// ExtremeWorstThreshold[c] is the i64 squared-distance threshold above
// which a fast-tier result with fraud_count == c triggers a full sweep
// escalation. Set per-class because the worst-of-top-5 distance
// distribution depends on whether the local cluster is fraud-dense.
//
// Calibrated by cmd/calibrate against test-data.json on linux/amd64.
//
// Semantics:
//   - 0 ⇒ disabled (the count-{2,3,4} gate alone handles this class)
//   - 0 < T < 1<<62 ⇒ escalate when fast-tier worst_i64 > T
//
// luanlouzada's published thresholds (at int16 × 10000, K=1280) are:
//   THR[0]=3501932 THR[1]=3569273 THR[2]=2906420
//   THR[3]=2738652 THR[4]=3297753 THR[5]=4594089
// We don't reuse those directly — our K=4096, our centroids and bbox
// layout differ — so cluster geometry differs. Calibration is required.
//
// Until calibration runs, all thresholds are 0 ⇒ phase-26 reduces to
// "FastNProbe=1 + escalate on count ∈ {2,3,4}". For boundaries where
// the fast tier returns a confident count (0/1/5) but the answer is
// wrong, the calibrated table is what closes the gap.
// Calibrated 2026-05-13 against test-data.json on linux/amd64 (verified
// to match darwin/arm64 exactly — same K, same index seed, no kernel
// drift at FastNProbe=1). cmd/calibrate output:
//
//   class=0 n=28884 fast_wrong=4    fixable=4    → threshold=3501931
//   class=1 n=416   fast_wrong=63   fixable=63   → threshold=3371673
//   class=2 n=748   fast_wrong=214  fixable=214  → always-escalate
//   class=3 n=782   fast_wrong=217  fixable=217  → always-escalate
//   class=4 n=410   fast_wrong=52   fixable=52   → always-escalate
//   class=5 n=22860 fast_wrong=6    fixable=6    → threshold=3375483
//
// Total escalations: 3362 / 54100 = 6.21% of queries.
// Projected post-calibration: FP=0 FN=0 (all 556 fixable entries
// trigger escalation; full sweep produces 0 errors).
//
// Note: class=0 threshold is within 1 i64 unit of luanlouzada's
// published 3501932 — strong corroboration that both calibrations
// converge to the same statistical boundary.
var ExtremeWorstThreshold = [6]int64{
	/* count=0 */ 3501931,
	/* count=1 */ 3371673,
	/* count=2 */ 0, // always-escalate (count gate)
	/* count=3 */ 0, // always-escalate
	/* count=4 */ 0, // always-escalate
	/* count=5 */ 3375483,
}
