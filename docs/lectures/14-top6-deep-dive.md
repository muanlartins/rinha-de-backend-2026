# Lecture 14 — Top-6 deep dive: closing the gap to 6000

A line-by-line forensic comparison of the six current top-of-leaderboard
solutions against our phase-24 implementation (5471 / p99 2.23ms / FN=1).

The headline:

- The 1 FN we accepted as "structural at int16 quantization" is **not
  structural**. It is a quantization-scale bug. `int16 × 32000` introduces
  rounding noise on **48.4% of reference dim values**; `int16 × 10000` is
  **lossless** for the round4 input.
- The p99 gap to ≤ 1ms is not the kernel, not the LB, not mmap. It is
  `FastNProbe = 16`. Top solutions run **`FastNProbe = 1`** with a
  class-conditional escalation gate.

Both findings are pure scale/threshold changes — no algorithmic rewrite.

## 1. The leaderboard as of 2026-05-13

Pulled from `arinhadebackend.github.io/2026-preview/results-preview.json`,
sorted by `final_score`:

| Rank | Submission              | Lang      | Score    | p99      | FN | FP | Err |
|-----:|-------------------------|-----------|---------:|---------:|---:|---:|----:|
| 1    | luanlouzada             | C++       | **6000.00** | 1.00 ms | 0 | 0 | 0 |
| 2    | viniciusdsandrade       | C         | 5989.78  | 1.02 ms  | 0 | 0 | 0 |
| 3    | jairoblatt-rust         | Rust      | 5978.38  | 1.05 ms  | 0 | 0 | 0 |
| 4    | daniloitagyba           | .NET AOT  | 5946.86  | 1.13 ms  | 0 | 0 | 0 |
| 5    | rafaelcoelhox (`eu-sou-o-ze-pamonha`) | C | 5944.30 | 1.14 ms | 0 | 0 | 0 |
| 6    | hvini                   | C         | 5937.40  | 1.16 ms  | 0 | 0 | 0 |
| ...  |                         |           |          |          |    |    |     |
| 25-ish | us (phase 24)         | Go        | 5471.55  | 2.23 ms  | 1 | 0 | 0 |

The #1 spot is a **perfect 6000.00** — both detection and p99 saturated.
The top 6 all have FN=0/FP=0/err=0; the only thing separating them is p99.

The **#1 is a fresh C++ submission** that didn't exist when we wrote
lecture 08. p99 = exactly 1.00 ms means they sit on the p99-score cap
(`p99 ≤ 1ms → score 3000`) — there is no further reward for going faster.

## 2. Architectural convergence across the top 6

Sources cloned into `references/{luanlouzada,viniciusdsandrade,jairoblatt-rust,daniloitagyba,rafaelcoelhox,hvini}/`.

| Dimension                    | Top-6 convergence (5/6 or more)            | Outlier                                 | Ours (phase 24)                          |
|------------------------------|---------------------------------------------|-----------------------------------------|------------------------------------------|
| Quantization                 | **int16 × 10000**                          | viniciusdsandrade: f32 columns           | **int16 × 32000**                         |
| Algorithm                    | IVF, K = 1024–4096                          | viniciusdsandrade: flat + 16-group bucket | IVF, K = 4096                            |
| Fast-tier probes             | **1–8** (luanlouzada=1, jairoblatt=5, hvini=16) | —                                  | **16**                                   |
| Full-tier probes             | 20–128                                      | —                                       | K-sweep (4096, AABB-gated)              |
| Escalation trigger           | `count ∈ {1..4}` **and/or** `worst > thr[count]` | hvini: only `count ∈ {1..4}`        | only `count ∈ {2,3}`                    |
| SIMD                         | AVX2: `VPMADDWD` (int16 pair-mul-add)       | viniciusdsandrade: `VFMADD231PS` (f32)  | f32 `VFMADD` + i64 rerank                |
| Kernel early-exit            | mask + movemask at 8/14 dims                | —                                       | single gate at dim 8 (phase 24)         |
| Per-block prefetch           | 1× T0, 1–8 blocks ahead                     | luanlouzada: none                       | 3× T0 (phase 21)                         |
| HTTP                         | Hand-rolled, epoll, keep-alive, prebuilt responses | —                                | Hand-rolled, no pipelining              |
| LB                           | **SCM_RIGHTS fd-passing** (4/6)             | luanlouzada: Nginx round-robin over UDS; daniloitagyba: custom epoll | so-no-forevis (phase-18 compose) |
| Index mmap                   | mmap + MADV_HUGEPAGE / MAP_POPULATE / mlock | jairoblatt: `include_bytes!` heap-aligned | mmap + MADV_RANDOM/POPULATE/HUGEPAGE  |
| Determinism                  | Fixed seed, pre-built index in image        | —                                       | Fixed PCG seed                           |

The point is: **on every other dimension we already match the consensus.**
The two remaining gaps are quantization scale and fast-tier strategy.

## 3. FN = 1 root cause: int16 × 32000 vs int16 × 10000

### 3.1. What the JOURNEY claimed

Phase 23 inspection wrote:

> The 1 FN is structural at this memory budget. Fixing would require
> float32 references (168 MB, exceeds budget) or int32 quantization.

That conclusion was wrong. It rested on the assumption that our int16 ×
32000 quantization was as precise as a 32000-grid allows. It isn't.

### 3.2. The dataset is pre-rounded to 4 decimals

```bash
$ gunzip -c references.json.gz | head -c 200
[{"vector":[0.01,0.0833,0.05,0.8261,0.1667,-1,-1,0.0432,0.25,0,1,0,0.2,0.0416],"label":"legit"},...
```

Every dim value is either a sentinel (-1, 0, 1) or a 4-decimal float
written by Python's `round(x, 4)`. The native grid is **1 / 10000**.

### 3.3. Empirical verification

Counted rounding loss across the first 244,156 dim values in the dataset:

```
scale × 10000: 0 / 244156 rounded-loss     (0.000%)
scale × 32000: 118059 / 244156 rounded-loss (48.354%)
```

`v × 10000` is **always an integer** for round4 input → `int16(v × 10000)`
is lossless. `v × 32000` is integer only when `v` lies on a 1/32000 grid,
which only ~52% of round4 values do.

### 3.4. How the rounding noise produces the FN

`internal/dataset/dataset.go:71` does:

```go
func Quantize(v float32) int16 {
    if v <= -1 { return -QuantScale }
    if v >=  1 { return  QuantScale }
    return int16(v * QuantScale + 0.5)
}
```

For a round4 reference value `v = k / 10000` (integer k ∈ [-10000, 10000]):

- At `QuantScale = 10000`: returns exactly `k`. Lossless.
- At `QuantScale = 32000`: returns `round(k × 3.2)`. Half the time off
  by ±1 i16 unit.

The error per dim is bounded by `0.5` i16 units. Over 14 dims, the worst-
case shift in squared distance between two references near the boundary
is `14 × 2 × 32000 × 0.5 ≈ 224000` i16² units — well above the **8939**
i64 unit gap separating `ref=703996` (legit) and `ref=1707961` (fraud)
at the 5th-nearest boundary for entry 5472.

So our 5th-nearest can flip from fraud (oracle) to legit (us). It does.
Once. Deterministically. On exactly the one entry where the gap is small
enough that 32000-grid rounding crosses it.

### 3.5. The fix

```diff
- QuantScale  = 32000
- SentinelInt = -32000
+ QuantScale  = 10000
+ SentinelInt = -10000
```

Plus regenerate `index.bin` (Docker build step picks this up automatically).
The kernel and AABB-LB code is scale-agnostic — squared diffs scale by
the square of `QuantScale`, the bbox in i16 stays valid, `kernelSafety`
scales by the same factor (so `65536 × (10000/32000)² ≈ 6400`, or just
keep at 65536 to be conservative).

The fix should produce **FN = 0, FP = 0**. Detection score saturates at
3000. That alone is +106.85 points (closing 2819.38 → 3000 on the rate
component, modulo `weighted_errors_E` reset).

## 4. p99 closing path: FastNProbe = 1 with class-conditional escalation

### 4.1. What luanlouzada does

`references/luanlouzada/docker-compose.yml`:

```yaml
- NPROBE=20
- FAST_NPROBE=1
- ADAPTIVE_MIN=2
- ADAPTIVE_MAX=4
- EXTREME0_WORST_THRESHOLD=3501932
- EXTREME1_WORST_THRESHOLD=3569273
- EXTREME2_WORST_THRESHOLD=2906420
- EXTREME3_WORST_THRESHOLD=2738652
- EXTREME4_WORST_THRESHOLD=3297753
- EXTREME5_WORST_THRESHOLD=4594089
- REPAIR_MIN=99
- REPAIR_MAX=0
```

The runtime logic (paraphrased from `references/luanlouzada/src/index.hpp`):

1. **Fast tier**: scan only the single closest centroid (`FAST_NPROBE=1`).
   On a query with K=1280 and avg cluster size ~2343 vectors, this is
   one tight SIMD loop over ~2343 vectors — call it **~30µs** of CPU.
2. **Adaptive gate**: if `fraud_count ∈ {ADAPTIVE_MIN, …, ADAPTIVE_MAX}`
   = `{2, 3, 4}`, the result is ambiguous. Escalate to `NPROBE=20`.
3. **Extreme gate**: even when `fraud_count` is "extreme" (0 or 5, or
   any unconditional outcome), if the **worst-of-top-5 distance** exceeds
   `EXTREME[count]_WORST_THRESHOLD`, the cluster is sparse around the
   query → the top-5 might be tail-cluster noise → escalate.
4. **Repair pass**: full sweep with bbox pruning if both gates trip.

The thresholds were class-tuned from a calibration run (see
`references/luanlouzada/src/profile.cpp` — the source of the magic
numbers). Each threshold is a per-class "the top-5 is suspect if too
far" check. They guarantee FN=0 without paying for `nprobe=20` on every
query.

### 4.2. Our current strategy (`internal/search/ivf.go:43-103`)

```go
const FastNProbe = 16
// ... fast tier: scan 16 clusters ...
if count == 2 || count == 3 {
    // sweep remaining K-16 clusters with AABB-LB
}
```

`FastNProbe = 16` means **16 cluster scans per query unconditionally**.
At ~10 µs per cluster scan (our measured per-cluster time on linux/amd64
under the bot), that's **~160µs of CPU baseline** before any escalation.

luanlouzada's baseline is ~30µs (single cluster) — **5× cheaper**.
That's exactly the gap from p99 2.23ms → p99 1.13ms (daniloitagyba,
rank 4) or → 1.00ms (luanlouzada, with their other tweaks).

### 4.3. Why phase 22 ("NPROBE=8 + always-sweep") regressed

Lecture 11 and the journey both record that phase 22 tried `FastNProbe=8`
+ unconditional K-sweep. It regressed −84 points because **always-sweep
costs ~500µs**. That experiment doesn't refute the FastNProbe=1 + gated
escalation hypothesis — it confirms it. The principle is:

- Cheap fast tier (1 cluster).
- Gate escalation **narrowly** — only when the result is genuinely
  borderline. Escalation should fire on << 5% of queries.

luanlouzada's `EXTREME[count]_WORST_THRESHOLD` is the gating mechanism we
lack. We currently escalate only on `count ∈ {2,3}`, which is ~15-20%
of queries. That's both:

- **Not enough** to catch FN=1 (because the FN is at `count=2`, which we
  do escalate, but the escalation reruns the same broken-quantization
  search and finds the same wrong 5th).
- **Too much** for `count ∈ {2,3}` to be the only gate — luanlouzada also
  escalates on far-worst-distance for `count ∈ {0,1,4,5}`.

### 4.4. The fix

Two-step:

```
// internal/search/ivf.go
const FastNProbe = 1

// Add class-conditional worst-distance thresholds (calibrated offline).
var extremeWorstThreshold = [6]float32{
    /* count=0 */ THR0,
    /* count=1 */ THR1,
    /* count=2 */ 0, // always escalate
    /* count=3 */ 0, // always escalate
    /* count=4 */ THR4,
    /* count=5 */ THR5,
}

// After fast tier:
worst := scratch.Top.WorstF32()
count := scratch.Top.FraudCount()
needsEscalate := count == 2 || count == 3 ||
                 worst > extremeWorstThreshold[count]
if needsEscalate {
    // ...current sweep code...
}
```

Calibrate thresholds with a one-time offline run over `test-data.json`:
the 95th percentile of worst-distance per outcome bucket. Or just clone
luanlouzada's published thresholds, rescaled by `(10000/32000)² ≈ 0.0977`
once we switch quantization scales. (Their thresholds × 0.0977 give our
expected i64 units after the scale change.)

Expected p99: ~1.0–1.4 ms (matching ranks 4-6). Expected score: 5950+
once FN=0 and p99 < 1.4ms.

## 5. Per-repo highlights worth lifting

Beyond the two big findings, a few discrete-good-ideas to bank for
follow-ups:

### luanlouzada (`index.hpp:638-652`)

SIMD-masked Top5 insertion:

```c++
uint64_t limit = top.worst_dist();
__m256i limit_v = _mm256_set1_epi32(int(limit));
__m256i lt = _mm256_cmpgt_epi32(limit_v, acc);
uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(lt));
if (mask == 0) return;
// iterate live lanes via __builtin_ctz(mask)
```

We do 8 separate `if (scratch.BlockSum[lane] >= worstF32) continue` —
both correct, but the SIMD path lets the compiler keep the lane mask in a
register. Minor (single-digit %), but ergonomic. Not a priority.

### jairoblatt-rust (`src/knn.rs:200-285`, `src/http.rs:225-260`)

Multi-request **writev pipelining**:

> "receive buffer up to 8KB. parse loop extracts up to 16 iovecs per
> iteration. writev (monoio-backed io_uring) sends all responses in
> single syscall."

We don't pipeline. Our raw HTTP loop reads one request, writes one
response. Under high concurrency this matters; under low concurrency it
doesn't. Worth measuring once the quantization + FastNProbe wins land.

Also: jairoblatt's index is `include_bytes!`-embedded into the binary
(not mmap). That's a deliberate trade — no page-fault latency on first
touch, at the cost of doubled cgroup memory accounting (binary + heap
copy). Doesn't apply to us.

### viniciusdsandrade

Distinct strategy: **flat search with 16-group hard-partitioning by
discrete dims (online/card_present/known_merchant + 1)**, plus dimension
ordering by variance for max early-exit pruning effect.

> *"References partitioned into 16 groups at build-time via 4-bit binary
> key. Groups pre-sorted by lower-bound distance estimate; skip
> high-bound groups early."*

This is essentially the **phase-9 grid revival idea**, reborn as a
top-3 solution. The 32-partition layout we abandoned in phase 14 *was* a
viable architecture — we abandoned it because we couldn't get the
SIMD/kernel layer fast enough. viniciusdsandrade got both right.

Not a recommendation to revisit — IVF is working. But it's a useful
reminder: **architecture choice doesn't matter much above a certain
implementation quality**.

### rafaelcoelhox (`api.c:74-109`)

`mmap(MAP_POPULATE) + mlock() + MADV_HUGEPAGE`. We have MAP_POPULATE +
MADV_HUGEPAGE but **no `mlock`**. mlock prevents the kernel from
swapping the index pages out under pressure. Under a 167MB cgroup with
84MB index + ~50MB heap, swap pressure is unlikely. Skipping.

### hvini (`/src/main.c:354-404`)

Cleanest implementation of two-tier escalation. Pattern:

```c
for (stage = 0; stage < 2; stage++) {
    nprobe = stage == 0 ? 16 : 128;
    // ...scan nprobe clusters...
    int frauds = top5_fraud_count();
    if (frauds == 0 || frauds == 5) break; // unanimous, done
}
```

Their **fast tier is 16** (matching ours) and escalation is `count ∈
{1..4}` (vs our `{2,3}`). They sit at 5937.40 / p99 1.16ms while we sit
at 5471 / p99 2.23ms. The 800-point gap to us is presumably the
quantization scale (they're at int16 × 10000, lossless) — they get FN=0
where we get FN=1. Their p99 is then a function of careful C tuning,
not strategy.

This is direct evidence that **escalating on `count ∈ {1..4}` instead
of `{2,3}` does not blow up p99** — hvini does it and finishes at
1.16ms. So we have headroom to widen the gate.

## 6. Recommended trajectory

Three changes, ordered by expected impact and decoupled enough that they
can ship independently:

### Phase 25 — quantization scale: 32000 → 10000

- **Files**: `internal/dataset/dataset.go:24-25` (constants), regenerate
  `index.bin` at build time.
- **Expected**: FN 1 → 0. Detection score 2819.38 → 3000 (+180).
- **Risk**: low. The kernel is scale-agnostic. The AABB-LB stays valid
  with int16 endpoints. `kernelSafety` can stay at 65536 (over-margin).
- **Validation gate**: run `TestIVFFullDataset` locally — expect FN=0.
  Then submit. If darwin/arm64 generic kernel disagrees with linux/amd64
  AVX2 at the new scale, we get warned (per phase-23 lesson).

### Phase 26 — FastNProbe 16 → 1 + class-conditional escalation

- **Files**: `internal/search/ivf.go:21,43-103` (FastNProbe const,
  escalation logic).
- **Calibration**: one-time `cmd/calibrate` tool that runs all 54100
  test entries and emits per-class p95 worst-distance. Check it into
  `internal/search/thresholds.go`.
- **Expected**: p99 2.23 → 1.0-1.4 ms. Score 5800-5950 (assuming Phase
  25 already shipped, so detection is saturated).
- **Risk**: medium. Cross-platform precision (lesson from phase 23)
  applies — calibration must run on linux/amd64.
- **Validation**: `TestIVFFullDataset` FN=0 must hold. Plus regression
  test: `TestEscalationCoverage` — verify all 54100 entries either pass
  the fast tier or trigger correct escalation.

### Phase 27 — writev request pipelining (optional)

Only if Phase 26 doesn't fully close the p99 gap. Hand-rolled HTTP loop
parses up to N requests off the read buffer per round and emits all
responses in one `writev`. Reference: `references/jairoblatt-rust/src/http.rs`.

Expected: 0–200µs depending on bot concurrency. Marginal. Defer.

## 7. What this teaches

A few lessons in priority order:

1. **Don't accept "structural" claims without verifying them.** The JOURNEY
   declared the FN=1 unfixable based on a clean-looking arithmetic argument
   (quantization granularity vs i64 gap). The argument was correct in
   isolation, but the premise — that our quantization was as fine as the
   scale allowed — was empirically false. *Confidence in a deduction is
   bounded by confidence in its premises; verify the premises.*
2. **The data's native grid wins.** All five int16 top solutions use
   scale × 10000 — not because that scale is special, but because it's
   the grid the rinha generator outputs. We picked 32000 to be "as fine
   as int16 allows" and traded lossless representation for false
   precision. *When data is pre-rounded, match the rounding grid exactly.*
3. **CPU per query is the latency lever once the architecture is right.**
   Our phase 14-21 work made the algorithm correct. Phase 22-24 tried
   marginal asm tweaks. The **5x cost reduction from FastNProbe = 16 →
   1** dwarfs all of those. The lesson isn't "do less work" — it's
   "do work only where it changes the answer." luanlouzada's adaptive
   thresholds are a clean expression of that principle: spend CPU only
   on queries that need it. *Optimize the trigger, not the body.*
4. **Cross-platform validation is binding.** Phase 23 cost us 70
   points by changing NPROBE without an amd64 test bed. Phase 26 must
   not repeat the error: calibrate on linux/amd64 (e.g., via Docker
   buildx + `docker run --platform linux/amd64`).

## 8. References

- Cloned repos under `references/`:
  - `luanlouzada/` (6000.00)
  - `viniciusdsandrade/` (5989.78)
  - `jairoblatt-rust/` (5978.38)
  - `daniloitagyba/` (5946.86)
  - `rafaelcoelhox/` (5944.30)
  - `hvini/` (5937.40)
- Leaderboard JSON: `arinhadebackend/arinhadebackend.github.io@2026-preview:results-preview.json`.
- Our code under examination: `internal/search/ivf.go`, `internal/dataset/dataset.go`, `internal/ivf/index.go`, `cmd/api/main.go`.
- Predecessor lectures: 08 (top-10 survey), 09 (k-means IVF), 11 (two-tier nprobe), 12 (SCM_RIGHTS), 13 (mmap madvise).
