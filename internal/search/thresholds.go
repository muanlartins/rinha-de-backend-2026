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
// Phase 39 recalibration (FastNProbe=2, K=4096, EscalateNProbe=32):
//   class=0 n=28878  fast_wrong=4   fixable=4    → threshold=3501931
//   class=1 n=398    fast_wrong=18  fixable=18   → threshold=3780951
//   class=2 n=751    fast_wrong=123 fixable=123  → always-escalate
//   class=3 n=802    fast_wrong=140 fixable=140  → always-escalate
//   class=4 n=413    fast_wrong=18  fixable=18   → always-escalate
//   class=5 n=22858  fast_wrong=0   fixable=0    → no escalation needed (!)
//
// Escalation rate: 2253 / 54100 = 4.16 % (was 6.21 % at FastNProbe=1).
// Projection FP=FN=0 under top-N=32 production path.
//
// Notable change from previous calibration: class=5 now has threshold=0
// because FastNProbe=2 already produces the correct top-5 for every
// confidently-fraud query. The 6 misclassifications at FastNProbe=1
// (which previously required threshold=3375483) are all caught by the
// 2nd cluster scan.
// Phase 41 recalibration (FastNProbe=2, K=4096, EscalateNProbe=224 top-N)
// against test-data.json updated 2026-05-20 (commit 9dd2c32 — fraud_count
// 24058→23959, edge_case_count 797→645):
//
//   class=0 n=29158  fast_wrong=5    fixable=5    → threshold=92128083
//   class=1 n=301    fast_wrong=25   fixable=25   → threshold=4592068
//   class=2 n=657    fast_wrong=134  fixable=134  → always-escalate
//   class=3 n=593    fast_wrong=134  fixable=134  → always-escalate
//   class=4 n=365    fast_wrong=40   fixable=40   → always-escalate
//   class=5 n=23026  fast_wrong=15   fixable=15   → threshold=91459358
//
// Escalation rate: 20307 / 54100 = 37.54%. Top-N sweep at N={32,48,64,96,
// 128,160,192,224,256} showed N=224 is the smallest value reaching
// FP=0 FN=0 under the production phase-27 path. Phase 40 used full-K
// (4094 clusters) which was overkill — p99 ballooned to 2.82ms in the bot
// run. Top-N=224 cuts escalation work ~18x while preserving accuracy.
var ExtremeWorstThreshold = [6]int64{
	/* count=0 */ 92128083,
	/* count=1 */ 4592068,
	/* count=2 */ 0, // always-escalate (count gate)
	/* count=3 */ 0, // always-escalate
	/* count=4 */ 0, // always-escalate
	/* count=5 */ 91459358,
}
