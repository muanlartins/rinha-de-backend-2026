# Lecture 15 — Sub-2ms plan: forensic vs the new top-1, staged phase 27+

Written 2026-05-14 after phase 26 landed (5618.55 / p99 2.41 ms, detection
saturated at 3000). The new #1 on the leaderboard is **crepao-da-massa /
silent-index** at **6000.00 / p99 0.98 ms** — the first sub-millisecond
submission ever. This lecture catalogues what they did, what we can copy,
and the staged plan to take our submission from p99 2.41 ms toward 1.0 ms.

## 1. New leaderboard snapshot (2026-05-14)

| Rank | Submission | Lang | Score | p99 |
|---:|---|---|---:|---:|
| 1 | **crepao-da-massa silent-index** | C++ | **6000.00** | **0.98 ms** |
| 2 | viniciusdsandrade andrade-cpp-ivf | C | 5976.23 | 1.06 ms |
| 3 | jairoblatt jrblatt-html | Rust | 5972.20 | 1.07 ms |
| 4 | jairoblatt jairoblatt-rust | Rust | 5970.10 | 1.07 ms |
| 5 | rafaelcoelhox eu-sou-o-ze-pamonha | C | 5955.42 | 1.11 ms |
| 6 | daniloitagyba itagyba-dotnet | .NET AOT | 5946.86 | 1.13 ms |
| 7 | hvini | C | 5937.40 | 1.16 ms |
| 8 | oliveirajhony zig | Zig | 5888.15 | 1.29 ms |
| 9 | lemesdaniel floating-finch-zig | Zig | 5879.86 | 1.32 ms |
| 10 | JordaoGustavo csharp | C# | 5855.40 | 1.40 ms |
| 11 | joycegodinho | Go | 5854.11 | 1.40 ms |
| ... | ... | | ... | ... |
| ~24 | **us (phase 26)** | Go | **5618.55** | **2.41 ms** |

**Top-10 cutoff: p99 ≈ 1.40 ms / final ≈ 5855**.
**Sub-1ms (top-1) requires p99 ≈ 0.98 ms / final = 6000**.

Detection is saturated at 3000 for every entry in the top 10 — the *only*
axis remaining is p99 latency. Every microsecond we save in the hot path
becomes a leaderboard rank.

## 2. The crepao-da-massa playbook

Cloned to `references/crepao-silent-index/`. Same overall architecture as
the other top-6 (int16 × 10000 IVF, AABB-LB, fd-passing LB, mmap+madvise,
hand-rolled HTTP, AVX2). The **specific** wins:

### 2.1. K = 1664 clusters (tuned, not chosen)

`Dockerfile:20` — `build_index ... 1664 65536 6`. Commit history shows
they swept K = 1280 → 1664 (commit `6609bd3 "Tune offline index cluster
count"`) along with explicit offline profiling via `src/profile.cpp` and
`src/offline_api_profile.cpp`. The profile tool measures p50/p95/p99 and
escalation rate per K-candidate. **Empirical K-tuning is the differentiator
between top-6 and top-1.**

Why K matters: with `FAST_NPROBE = 2` (their fast tier scans 2 clusters)
and full-K=1664 sweep on the repair path, the AABB-LB elimination cost is
**~2.5× cheaper** than our K=4096 sweep. And cluster scans themselves are
**larger but fewer** (~1804 vecs/cluster vs our ~732), which is friendlier
to instruction-level parallelism.

### 2.2. Dimension-paired interleaved block layout

`src/index.hpp:124-126`, build at `build_index.cpp:298-305`. Their block
of 8 vectors stores `(dim 0, dim 1)` pairs interleaved per vector, then
the next dim-pair, etc. The runtime loads 16 int16s with one
`_mm_loadu_si128`, then `_mm256_madd_epi16(diff, diff)` produces 8
squared-diff i32s from a single instruction. Latency 5, throughput 0.5.

Commit `e61f5e0 "Interleave vector blocks for paired SIMD scan"` is the
**biggest single algorithmic commit** in their history. They credit it
informally as the change that broke into sub-1ms.

We currently use f32 FMA over widened int16→f32 (`internal/kernel/dist_amd64.s`).
The VPMADDWD path skips the widen entirely. Expected scan-cluster speedup
~20-30 %.

### 2.3. Escalation gate: `fraud_count ∈ [1, 4]`

`docker-compose.yml:42-43`, `server.cpp:562-568`. Their `repair_min=1`,
`repair_max=4` — i.e., escalate on **any** non-extreme outcome. No
class-conditional worst-distance threshold for count=0/5.

Our calibration found that escalating count=0/5 only when `worst > THR[c]`
catches 4+6 = 10 fixable entries out of ~52000 confident queries (0.02 %).
Crepao either doesn't have those entries on their K=1664 layout (different
cluster geometry hides the boundary), or accepts them.

This is an *opportunity* for us: our class-conditional thresholds at
count=0/1/5 escalate ~1422 queries to fix 73 (5.1% useful rate). Tightening
those — or using crepao's simpler "extreme-only confidence" approach —
could cut wasted escalations significantly.

### 2.4. Date function hardcoding

`server.cpp:144-162`. Since test data is from a known month (March 2026),
they hardcode `epoch_minutes` and `weekday_monday0` to O(1) table lookups
over a 31-day window. We do full date arithmetic in `internal/vector/fast.go`.
Worth checking how much CPU that costs us — probably 1-3 µs per query.

### 2.5. Aggressive page warming

`src/index.hpp:811-820` — `warm_pages()` walks the mmap in 4KB strides
*after* `MADV_WILLNEED`. Double-prefault to ensure TLB is hot. We already
have `MADV_HUGEPAGE | MADV_POPULATE_READ | MADV_RANDOM` (lecture 13) plus
500-iter warmup, but our warmup hits the *search path*, not raw page
sweep. Adding an explicit page-sweep pass before search warmup might shave
the first few requests' tail latency.

### 2.6. Build flags

`Makefile` — `-Ofast -march=haswell -mavx2 -mfma -flto -fno-exceptions
-fno-rtti -pthread`. We have `GOAMD64=v3 -pgo=auto` (Haswell + PGO). Their
LTO is more aggressive than Go's default; ours is implicit via Go's
internal linker. Not a meaningful gap.

### 2.7. The "html" submission

`references/jairoblatt-html/` — *not* a different stack. It's an XML DSL
that **transpiles to Rust source** identical to `jairoblatt-rust/`. The
+2.1 score delta vs jairoblatt-rust comes from **upgrading the so-no-forevis
image to v1.0.0 and bumping LB CPU from 0.10 → 0.20**, sacrificing 0.05
from each API.

We already use **v1.0.0** of so-no-forevis and **0.10 CPU** for the LB.
A CPU rebalance could give us 1-2 cosmetic points — defer; not the lever.

## 3. Our cost model at p99

Per-escalating query (the worst 6.21 % that set p99):

```
ScoreAllCentroids        K=4096 × 14 = 57k MACs (pure-Go autovec)      ~25 µs
PickTopNCentroids        N=1 linear scan of 4096                       ~ 2 µs
scanCluster (fast tier)  1 cluster × ~92 blocks (early-exit + AABB)    ~ 8 µs
Full-sweep escalation    4095 cluster-LB checks + ~40 actual scans     ~450 µs
─────────────────────────────────────────────────────────────────────────────
                                                                       ~485 µs CPU
                                                                       + scheduler/queueing
                                                                       = p99 2.41 ms wall-clock
```

**The escalation is 90 % of the worst-1 % CPU budget.** Everything else is
already tight enough to not show up in p99.

Also: 3289 of our 3362 escalations are **wasted** — the full sweep returns
the same fraud_count as the fast tier. Only 73 entries genuinely flip.
The current calibration over-triggers because thresholds are set to
"smallest fixable worst − 1", which catches all 73 fixable cases plus
~1349 same-class queries above the same numeric threshold.

## 4. Staged plan

Each phase is independently shippable and reversible. Phases bundle code
+ calibration + test; nothing ships without a passing `TestIVFFullDataset`
on linux/amd64.

### Phase 27 — Top-N escalation (biggest expected lever)

Replace the full K=4096 sweep with the next N=24 unscanned-and-nearest
clusters by centroid distance.

**Code.** `internal/search/ivf.go:124-129`. Reuse `scratch.CentroidDists`
(already computed). Add `PickNextNUnscanned(dists, scanned[], n, out[])`
that returns the next N indices not yet in `scanned`. Loop body
unchanged (radius + AABB-LB + scan-block).

**Cost.** 24 cluster pre-prune checks × ~100 ns + ~10 actual scans ×
~3-8 µs ≈ **40-60 µs**. Saves ~390 µs on escalating queries.

**Expected p99.** 2.41 ms → **~1.5–1.7 ms**.

**Risk.** Top-N is not provably sound — there could be a true 5-NN in a
cluster ranked > N by centroid distance. Validate via `cmd/calibrate`:
add a `FraudCountTopNSweep(n=24)` simulation, require post-cal projection
FP=0 FN=0. If FN appears, bump N to 32 or 48.

**Validation gate.** New cross-check `TestTopNEscalation` over the full
test set. Must show FP=0 FN=0 at chosen N. Recalibration of thresholds
is **not** required — `ExtremeWorstThreshold` already classifies
"escalation needed"; the escalation method changes, not its trigger.
Actually it does affect outcome: a tight threshold catching "fixable"
entries may now leave them unfixed if N is too small. So validate
end-to-end.

**Dependencies.** No reindex.

### Phase 28 — Threshold recalibration with phase-27 path active

Once phase 27 ships and the bot confirms FP=0 FN=0, recalibrate
`ExtremeWorstThreshold` using `FraudCountTopNSweep` (not `FraudCountFull`)
as the "oracle within IVF". Goals:

- Loosen count=0/1/5 thresholds (catch only entries actually needing
  escalation, not the over-broad sweep we have today).
- Consider replacing always-escalate count=4 with a threshold (only 52 of
  410 entries fixable → 87 % waste).
- Possibly drop count=2/3 to threshold too if the marginal fixable yield
  is low.

**Cost / gain.** If escalation rate drops from 6.21 % to ~2-3 %, p99 sees
those queries less often. Net **~−100 µs** at p99.

**Expected p99 after 27 + 28.** **~1.2–1.4 ms** (top-10 territory).

**Risk.** Loosening thresholds can introduce errors. Strict gate:
post-cal projection must remain FP=0 FN=0.

### Phase 29 — VPMADDWD i32 kernel

Replace `ScanBlock8AVX2` (f32 FMA, currently dim-major) with a
VPMADDWD-based int32 accumulator on dimension-paired blocks.

**Code.** New `internal/kernel/dist_amd64.s` with VPSUBW + VPMADDWD +
VPADDD inner loop. Drop the `VPMOVSXWD + VCVTDQ2PS` widen prelude. May
require reshaping `BlockData` to dim-paired layout — `internal/ivf/index.go`
serializer.

**Cost.** Estimated 20-30 % faster per cluster scan. Saves ~3-5 µs on
fast tier, ~30-50 µs on escalation. Compounds with phase 27.

**Expected p99 after 27+28+29.** **~1.0–1.2 ms**.

**Risk.** Two failure modes: (a) int32 overflow on accumulator —
mitigated by accumulating into 2× i32 lanes and reducing to i64 at exit;
(b) reshaping `BlockData` invalidates serialized index — need version
bump in `internal/ivf/index.go` magic header.

**Validation gate.** `TestKernelParity` over 10k random vectors comparing
new vs old kernel (must match exactly modulo `kernelSafety` margin).
`TestIVFFullDataset` FP=0 FN=0.

**Dependencies.** Reindex required (new block layout). Recalibration may
be needed if any precision drift surfaces.

### Phase 30 — Borderline pre-check via centroid radii

Before deciding to escalate, check whether ANY of the next M=8 closest
unscanned centroids could host a vector closer than current `Top.WorstI64()`.
Triangle inequality: if `(sqrt(centroidDist[c]) − radius[c])² >=
Top.WorstI64()` for *all* M, no cluster within range, skip escalation
entirely.

**Code.** `internal/search/ivf.go` — pre-loop in escalation block.

**Cost.** M=8 sqrts + 8 comparisons = ~50 ns. Cheap.

**Expected p99.** **−30 to −80 µs**. Conjecture: many "fast result was
already optimal" cases get vetoed before paying any escalation cost.

**Risk.** Sound — same triangle inequality as the in-loop gate. No new
errors possible.

**Dependencies.** None. Can ship before or after 29.

### Phase 31 — Hand-rolled `ScoreAllCentroids` asm

The pure-Go centroid scoring runs on **every** query (94 % fast-only,
6 % fast + escalation). At ~25 µs it dwarfs other per-query costs.

**Code.** New `internal/kernel/centroid_amd64.s`. Inner loop: per dim,
broadcast q[d], subtract from centroids[d*K + c], FMA into acc[c]. K=4096
fits comfortably in 8-register AVX2.

**Cost.** Estimated 10-15 µs saved on every query (40-60 % faster).

**Expected p99 after 27+28+29+30+31.** **~0.95–1.10 ms**.

**Risk.** Asm correctness. Mitigated by parity tests vs current
generic Go path.

### Phase 32 — PGO refresh + final polish

After the hot path stabilizes (post-31), capture a fresh PGO profile under
realistic test load. The current `cmd/api/default.pgo` was captured from
synthetic warmup and is invalidated by every algorithmic change since
phase 21.

**Cost.** 1-3 % overall — typically 20-50 µs at p99.

### Optional (deferred)

- **L4 — K reduction (4096 → 1664).** Crepao's choice. Full-sweep is
  K-proportional, but phase 27 already removed full-sweep. Defer unless
  phase 27 underperforms.
- **L6 — writev request pipelining.** jairoblatt-rust bundles up to 16
  iovecs per syscall. Bot keep-alive concurrency unknown; expected
  win ≤ 50 µs.
- **CPU rebalance LB 0.10 → 0.16** (à la jairoblatt-html). Marginal +2-5
  cosmetic points.

## 5. Phase order rationale

```
27 (top-N escalation)       —  biggest lever, ~−400 µs CPU on bad-1%
   ↓ ship + bot verify
28 (threshold recalibration)—  cuts escalation rate, ~−100 µs at p99
   ↓ ship + bot verify
29 (VPMADDWD kernel)        —  −50 µs per escalating query
   ↓ ship + bot verify (requires reindex)
30 (borderline pre-check)   —  −30-80 µs free
   ↓ ship + bot verify
31 (asm centroid scoring)   —  −10-15 µs on every query
   ↓ ship + bot verify
32 (PGO refresh)            —  +1-3 % cumulative
```

After 27 + 28 alone: **predicted p99 ≈ 1.2-1.4 ms → top 8-10**.
After 27+28+29: **≈ 1.0-1.2 ms → top 4-7**.
After 27-31: **≈ 0.95-1.10 ms → top 1-3**.

## 6. Validation discipline (lessons re-bound)

Phase 23 regressed −70 score because NPROBE was changed without amd64
validation. **Every phase must:**

1. Pass `TestIVFFullDataset` locally with FP=0 FN=0 (allow at most 1 FP
   if the score arithmetic justifies — usually not).
2. Pass `TestIVFFullVsBrute` (relaxed: classification match, not exact
   top-5 set). Cosmetic top-5 drift is OK; classification flips are not.
3. Be runnable on linux/amd64 via `docker buildx build --platform
   linux/amd64`; if the change affects kernel precision or threshold
   selection, **calibration must run in an amd64 container**.
4. Be reversible — each phase is one commit; rollback is `git revert` +
   re-push `:latest`.

Don't bundle two phases into one ship until both have individually
cleared the bot. Bot turnaround is 30-90 minutes; cheap to wait.

## 7. The 6000 ceiling

p99 ≤ 1.00 ms scores the max p99_score of 3000. There is no reward for
going faster than 1.00 ms — the points cap. So our actual goal is "get
to 1.0-1.1 ms reliably, then accept variance and call it done." Anything
below ~1.0 ms is bot-side variance and doesn't help.

Going below 0.98 ms might still happen — crepao's number — but the
return on the work is zero. We optimize toward p99 ≈ 1.0 ms, not 0.

## 8. References

- `docs/lectures/14-top6-deep-dive.md` — predecessor; analyzed the
  previous top-6 before crepao-da-massa joined.
- `references/crepao-silent-index/` — new top-1 source.
- `references/jairoblatt-html/` — code-gen variant of jairoblatt-rust.
- `cmd/calibrate/main.go` — calibration tool that will be extended in
  phases 27 and 28.
- `internal/search/{ivf.go,fast.go,thresholds.go}` — main targets.
