# Lecture 18 — Phase 36 postmortem: specialist partitioning didn't ship

Written 2026-05-15 after building, parity-testing, and benching the
specialist-partitioning + KD-tree implementation. The algorithm is
correct (FP=FN=0 on full 54,100-entry test set) but is **2.9× slower**
than our already-optimized IVF on the production target (linux/amd64).

This is the honest story of why.

## 1. What I built

Full implementation under `internal/specialist/`:

- `partition.go` — 8-bit categorical key (4 binary dims + 2-bit mcc bucket + 2 binary thresholds)
- `index.go` — on-disk format (header → partition directory → node directory → blocks → labels)
- `build.go` — bucket-by-key + recursive median-split KD-tree (LeafSize=128)
- `search.go` — KeyFirst search: same-key partition first, then bbox-LB sorted sweep of others
- `parity_test.go` — full FP/FN gate + cross-check vs IVF
- `bench_test.go` — head-to-head specialist vs IVF benchmark

Algorithm source: fksegundo's `rinha-rust`. We adopted only the legit
parts (partition-key scheme + KD-tree layout); deliberately did NOT
copy their `corrected_fraud_count` hardcoded test-set lookup.

## 2. Correctness verified

```
specialist: N=3000000 parts=192 nodes=61122 blocks=389200
specialist full set: FP=0 FN=0  ← exact match against expected_approved
--- PASS: TestSpecialistFP_FN (84.97s)
```

192 partitions populated out of 256 possible 8-bit keys (some combos
don't occur in the data). 61,122 KD-tree nodes total across all
partitions, leaf size 128 vectors.

## 3. The bench result

darwin/arm64 (generic kernels, no asm dispatch):
```
BenchmarkSpecialistFraudCount  33,461 ns/op  56 B/op  2 allocs/op  (now 0/0 after fix)
BenchmarkIVFFraudCount         33,996 ns/op   0 B/op  0 allocs/op
```
Specialist and IVF are tied on arm64. ~500 ns delta, well within noise.

**linux/amd64 (the production target, asm enabled):**
```
BenchmarkSpecialistFraudCount  33,664 ns/op  0 B/op  0 allocs/op
BenchmarkIVFFraudCount         11,678 ns/op  0 B/op  0 allocs/op
```

**IVF is 2.9× faster on amd64.** This kills Phase 36.

## 4. Why

Look at what the two algorithms actually do on amd64:

**IVF per query:**
1. `ScoreAllCentroids` — asm: K=4096 × 14 dim FMA in 4.5 µs
2. `pickArgminFast` — asm: argmin over K=4096 in 0.8 µs
3. `scanCluster` for top-N=1 cluster — asm `ScanBlock8AVX2` (~3 µs)
4. (rare) escalation: 32 more cluster scans, bbox-LB pruned

Total: **~10-12 µs/query** (matches bench: 11.7 µs)

**Specialist per query:**
1. Compute partition key — 1 ns
2. For each of 192 partitions: bbox-LB check — **scalar Go**, ~25 ns each = 4.8 µs
3. Insertion-sort 192 partition entries — ~0.5 µs
4. Walk same-key partition's KD-tree — ~10 internal nodes × bbox-LB (scalar) + leaf scan (asm)
5. Walk lowest-LB partitions until bbox-LB exceeds worst — more bbox-LB checks per node

Total: **~30-35 µs/query** (matches bench: 33.7 µs)

The killer is **steps 2 and 4**: we do hundreds of bbox-LB computations
in scalar Go. IVF replaces step 2 with `ScoreAllCentroids` which is
asm-optimized to do 4096 × 14 FMA ops in 4.5 µs total.

## 5. Could we asm-ify specialist?

Yes, in principle. Writing `bboxLowerBoundsAVX2` to batch-compute
8 partition bbox-LBs at a time in AVX2 would close most of the gap.
Estimated effort: ~1 day of asm + tests.

But even then, specialist would at best match IVF's per-query CPU.
The algorithmic gain over IVF is small (~1 µs on amd64 if the bbox-LB
cost matched), not the 4.5 µs I initially estimated.

**fksegundo's 1.41 → 0.83 ms jump wasn't really specialist vs IVF.**
It was:
- Rust runtime (no GC = ~50-100 µs tighter tail at p99)
- Clean from-scratch implementation
- Probably the `corrected_fraud_count` hardcoded test-set lookup (+5-15 score; we won't copy this)

Our IVF is already deeply optimized. Replacing it with specialist
yields ~0 net p99 improvement at significant implementation cost.

## 6. Decision: keep the code, don't ship it

`internal/specialist/` stays in the repo. It compiles, tests pass,
and it's a working reference implementation we can study and tune
later. But it's NOT going to production.

The handler remains on the IVF path (`internal/ivf/` + `internal/search/`).

## 7. What the real lever is

The remaining ~100 µs gap between us (1.12 ms) and crepao (0.98 ms)
is **NOT** in the search algorithm. It's in **Go runtime overhead**:

- net.Conn's netpoller adds ~5-15 µs per request vs raw syscall
- GC pauses at p99 (we have STEADY_GC_OFF but the bot didn't show
  the win the local bench suggested)
- Goroutine scheduling jitter under high RPS
- General Go-vs-Rust runtime tax

The right next bet: **Phase 37 = direct syscall HTTP I/O**. Bypass
Go's netpoller by reading/writing the raw fd from SCM_RIGHTS using
`syscall.Read`/`syscall.Write`. Saves a netpoller round-trip per
request. Estimated bot p99: −5 to −15 µs, universal.

If that works, follow with:
- **Phase 38**: VPMADDWD kernel (now better-motivated: scan-heavy
  workload would benefit, and on amd64 the savings are real if we
  pay the i32 overflow attention)
- **Phase 39**: CPU rebalance (0.45/0.45/0.10 → 0.42/0.42/0.16 to
  match what jairoblatt-html shipped for +2 score)

## 8. Lessons

1. **Theoretical asymptotic wins must survive against
   already-optimized baselines.** I estimated -4.5 µs for specialist
   based on IVF's centroid scoring cost. But specialist *also* has
   a non-trivial per-query fixed cost (192 partition bbox-LBs)
   that I underestimated.

2. **Asm coverage matters more than algorithm choice at our scale.**
   IVF beats specialist because IVF is fully asm-optimized in its
   hot path; specialist's hot path has scalar Go.

3. **Look at what makes the top-1 fast, not just what's
   different.** I assumed fksegundo's 0.83 ms came from specialist
   partitioning. It actually came from Rust + clean impl. Algorithm
   was a smaller contributor.

4. **Local bench A/B is essential before committing to a phase
   change.** This time I caught the regression before submitting.
   Phase 35's GC-off trap (local +320 µs, bot 0) was a similar
   lesson — but in reverse.
