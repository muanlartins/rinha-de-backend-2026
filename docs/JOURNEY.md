# Journey: from -2640 to (hopefully) top 30

A record of every iteration on the Rinha submission, what worked, what didn't, and why. The numbers in the final-score column are from the rinha bot's evaluation on the Mac Mini Late 2014 (not local Rosetta benches).

## Phase summary

| Phase | What | Final score | p99 | Notes |
|---|---|---|---|---|
| 0 | First submission (dataset missing) | **-2558** | 361ms | FN=23945 — we returned stub for every request because `/resources/references.json.gz` wasn't mounted by the rinha bot |
| 0b | After dataset fix, but image was cached | **-2640** | 437ms | Same failure — Docker compose without `pull_policy: always` reused the old image |
| 1 | Exact grid + scalar early-exit, dataset baked | **3566** | 140ms | First working result. FP=0, FN=1, Err=2 |
| 1b | Same image, repeated runs | 3231, 3020 | 387, 494ms | Mac Mini noise — same code, 3× p99 variance across runs |
| 2 | IVF approximate, K=128, nprobe=8 | **3223** | 218ms | Built at runtime, K-means took ~25s. FP=5, FN=6 — not worth the accuracy loss |
| 3 | Pre-built IVF index baked into image | 3223 | 218ms | Same accuracy, but startup is now fast — pre-built tool runs at Docker build time, persists to /resources/index.bin |
| 4 | Exact IVF (LB pruning) + block-SIMD (BlockScan8) | **3410** | 200ms | Per-vector SIMD → per-block 8-wide SIMD (8 vectors at once, dim-major layout, AVX2 VPSUBW + VPMULLD + VPADDQ). Local Rosetta hit p99 15ms — proves the kernel works |
| 5 | + ulimits + seccomp:unconfined | (test pending) | — | Compose-only change, no code |
| 6 | Custom HTTP server (drop net/http) | 3452 | 232ms | No meaningful change on Mac Mini. The HTTP layer wasn't the bottleneck. |
| 7 | Flat IVF K=4096 with Lloyd k-means (rewrite) | local: 2888 | local 456ms | **Regression.** Lloyd k-means produces unbalanced clusters with meaningless centroids on discrete dims (online/card/unknown sit at intermediate 16000 values). 65 FP + 61 FN crossing threshold on local. |
| 8 | Flat IVF K=8192 with balancedSplit + ambiguity expansion | TBD | TBD | Replaced Lloyd with josehenrique-dev-Go's recursive median-split on max-variance dim. Exactly balanced clusters. nprobe=32 fast + nprobe=128 on borderline. Local FP=4 FN=8. |
| 9 | **Grid revival** — exact partitioned grid + AABB LB pruning + scalar early-exit | Mac Mini: **3065** | Mac Mini 567ms | Reverted to the algorithm of phase 1, now properly engineered: 32 partitions × ≤1024 cells × percentile-binned grid dims (0, 12, 7) × per-cell AABB. Drops the SIMD kernel (early-exit doesn't compose with 8-wide block ops). Restores exact KNN: **FP=0, FN=1, Err=0**. Mac Mini measured on commit `c4d219b` via issue [#3709](https://github.com/zanfranceschi/rinha-de-backend-2026/issues/3709) — det_score = 2819 (max, modulo the 1 structural FN), p99_score = 246. All remaining gap is latency, not detection. See lecture 05. |
| 10 | + GOAMD64=v3 + PGO + sort cells by dist-to-centroid | Mac Mini: **3712** | Mac Mini 100.21ms | Three low-risk wins bundled, validated on Mac Mini via issue [#3727](https://github.com/zanfranceschi/rinha-de-backend-2026/issues/3727). (1) Haswell baseline (SSE4.2/AVX/AVX2/FMA3) — default v1 leaves SSE off entirely. (2) Profile-guided optimization with a profile captured from the local handler bench and checked in as `cmd/api/default.pgo`. (3) Per-cell sort by squared distance to centroid at index build time — Josiney's "Golden Optimization". Net **+647** over phase 9 (3065). FP=0, FN=1, Err=1. Picked up 1 Err (likely queue-timeout under burst). |
| 11 | int8 quantization | local: regression, NOT pushed | n/a | Tried per-dim linear int8 (scale 127, sentinel -127). `TestGridFullDataset` showed **FP=51, FN=62** — projected −1080 detection points. Local-bench latency win only 10 %. Net negative in every Haswell scenario. Reverted. Detailed in lecture 06. |
| 12 | float32 with mmap-shared-tmpfs | local: regression, NOT pushed | n/a | Tried float32 storage to eliminate the 1 FN. Empirically **the 1 FN survives float32** — it's not a quantization artifact, it's a structural mismatch between our search and rinha's label generator (probably parser-related, on test-data.json entry 5472). Plus latency regressed 18 % locally (cache pressure from 84 → 168 MB). Reverted. Built `LoadIndexMmap` + cgroup-friendly shared-tmpfs design for archive purposes — usable if some future change wants the memory headroom. Detailed in lecture 06. |
| 13 | Load shedder (`SHED_SLOTS=4`, `SHED_TIMEOUT_MS=3`) | Mac Mini: **3823** | Mac Mini 99.25ms | Net **+110** over phase 10 via issue [#3768](https://github.com/zanfranceschi/rinha-de-backend-2026/issues/3768), but not where expected. p99 was essentially unchanged (99 vs 100 ms) — shedder rarely fired. Win came from **eliminating the 1 Err** phase 10 picked up (Err weight 5 vs FN weight 3 → −106 detection penalty saved). Detection now saturated at **2819/3000 = max modulo the 1 structural FN**. p99_score still 1003/3000, all remaining headroom there. |

## What I learned

### The dataset doesn't get mounted

The single biggest mistake of the early phases: the rinha bot does NOT mount `references.json.gz` into the container. Every participant must ship the file inside the Docker image. The FAQ literally says *"pre-process the dataset during the container build"* — I read it as suggesting an optimization, when in fact it's a requirement.

Workaround: `COPY references/rinha-official/resources/references.json.gz /resources/` in the Dockerfile, and (importantly) `pull_policy: always` on the submission compose file so the rinha bot doesn't re-use a cached version of `:latest`.

### `pull_policy: always` matters

When Docker compose doesn't specify a pull policy, it uses whatever image is already cached locally on the bot's machine. The first time our image was pulled (broken version), it stuck around. Subsequent test runs found `:latest` locally and didn't re-pull. Adding `pull_policy: always` was the fix.

### IVF approximate isn't worth the accuracy hit on this scoring

K-means + probe-top-M-clusters gives a useful speedup vs scanning the whole partition. But the rinha scoring punishes FP/FN heavily: 1 FN = 3 weighted errors, each error costs ~30-50 points. Even a small (~5/24k) miss rate costs ~150-200 points. The exact LB-pruning version of IVF (probe all clusters but break when LB ≥ topD5) recovers full accuracy and is comparably fast since most clusters get pruned anyway.

### Block-major SIMD (8-wide) is much faster than per-vector

Our first AVX2 kernel computed one squared distance at a time (16 int16 lanes, VPMADDWL chunks). The block-major version computes 8 squared distances in parallel (16 dims × 8 vectors per block; one VPBROADCASTW + VPSUBW + VPMULLD + VPADDQ per dim). On Apple M4 Pro local bench, it dropped p99 from ~5ms to ~15ms — actually slower because Rosetta's 8-wide ops were less optimal than narrower. But on the real Mac Mini, the same kernel was the difference between p99 217ms (per-vector) and 200ms (block) — the workload is memory-bound there, and block-major has better cache locality (dim-by-dim sequential reads).

### Local Rosetta numbers don't transfer

Apple Silicon M4 Pro under Rosetta consistently gave us p99 4-15ms. The Mac Mini gave p99 140-494ms (3× variance across runs). The difference is:
1. M4 Pro is ~3× faster per cycle natively, plus Rosetta x86 emulation is good
2. Mac Mini has ~6 MB L3 cache (vs 48 MB on M4 Pro) — our 96 MB dataset thrashes
3. **The Mac Mini runs k6 on the same hardware** — load gen contends with SUT for CPU

The third point is the killer. k6 saturating one core while our 2 replicas have 0.45 CPU each means k6 itself queues, and the measured p99 is dominated by queue wait, not server latency.

### Top Go submissions beat us by avoiding net/http

Looking at josehenrique-dev-Go (#23, p99 1.76ms), they don't use net/http at all. The server is `net.Listen("unix", ...)` + manual accept loop + custom byte-level HTTP parser + pre-built response bytes (full HTTP/1.1 headers included). Each `handleConn` reads requests in a loop on a single connection (keep-alive), so they amortize the syscall cost across many requests.

Stdlib net/http allocates per request: `http.Request` struct (~500 bytes), header map (~100 bytes), routing, etc. Even with a sync.Pool body buffer, the per-request overhead in Go's HTTP stack is ~20-50 µs. At our target of 1 ms per request, that's 2-5% of budget — meaningful at the top of the leaderboard.

### Quantization costs us exactly 1 FN

The official labels were generated by **brute-force KNN over float32 vectors**. We quantize to int16 (scale 32000), which introduces ~0.003% per-dim error. On entry 5472 of test-data.json, this is enough to flip which vector is the 5th-nearest, producing fraud_count=2 (approve) instead of the labeled fraud_count=3 (deny). Verified by running over all 54,100 test entries locally — exactly 1 mismatch, deterministic.

Fix: switch dataset storage to float32. Cost: 168 MB for 3M × 14 × 4 bytes — over our 167 MB cgroup. Tight, doable if we drop padding and minimize other overhead.

## Architecture (current)

```
┌────────────────────┐
│ HAProxy (0.10 CPU, 16 MB)
│ port 9999 → UDS round-robin
└──────┬─────────────┘
       │
   ┌───┴───┐    ┌──────┐
   │ api-1 │    │ api-2│   each: 0.45 CPU, 167 MB
   │  Go   │    │  Go  │
   └───┬───┘    └──┬───┘
       │           │
       └─────┬─────┘
             │
       ┌─────▼──────┐
       │ index.bin  │ pre-built at image build time
       │  ~99 MB    │ 32 partitions × 128 k-means clusters per partition
       └────────────┘ Stride=16, block-major SIMD layout
```

**Search path (current):**
1. Parse POST /fraud-score body with custom byte-level parser (zero-alloc, ~430 ns/request)
2. Compute 5-bit partition key from dim 9/10/11/sentinel bits
3. Lookup cluster bbox lower bounds (LB) in that partition
4. Sort clusters by LB ascending
5. Walk clusters in order, scan blocks of 8 vectors with BlockScan8AVX2 SIMD
6. Break when cluster LB ≥ current top-5's worst distance
7. Return pre-built response bytes by fraud-count index

## What we learned (running tally)

### The bottleneck isn't where you think

We assumed for many phases that the HTTP layer was the gap to top Go submissions. Custom raw HTTP gave us **+0** on Mac Mini. The actual gap was algorithmic — partitioned K=128 IVF (4096 clusters with LB pruning, exact KNN) scans ~5,000 vectors per query and runs at 200 µs CPU. Top Go's flat K=8192 balancedSplit (366 vectors per cluster, approximate via nprobe=8) scans 2,900 vectors at 80 µs CPU. The 2.5× CPU reduction matters more than HTTP layer micro-optimizations.

### k-means is the wrong tool for mixed continuous/discrete data

The Rinha vector layout has 14 dims, of which 5 are effectively binary (is_online, card_present, unknown_merchant, sentinel bits) — values are either 0 or 32000 (or -32000 for sentinels). Lloyd k-means produces centroids at the MEAN of cluster members, which for a 50/50 binary split lands at 16000 — a value no real vector has. Searching by "nearest centroid" then bunches together vectors that don't actually share the binary characteristic. The result: 65 FP + 61 FN at K=4096 nprobe=8.

**Balanced KD-split** (recursive median-split on max-variance dim) sidesteps this by partitioning on discrete dims FIRST (they have max variance early in the recursion). Each leaf cluster is homogeneous on those bits, like our original 32-partition prefilter was — but without hardcoding which dims to split on.

### Mac Mini variance is real and not your code's fault

Identical images gave us p99 140ms → 387ms → 494ms across three runs of the original grid + net/http build. 3.5× variance with the same code. The Mac Mini runs k6 ON THE SAME HARDWARE as the SUT containers; depending on how the OS schedules them, k6 can grab CPU at the wrong moment and queue our requests. Below a certain server speed (~5 ms p99) you stop being CPU-bound and start being scheduler-bound, and there's nothing in our code that fixes that.

### `pull_policy: always` saved the submission

The single most impactful change of the session was a YAML one-liner. Without it, the rinha bot's Docker daemon cached our first (broken, no-dataset) image and never re-pulled — every subsequent test ran the same stale binary regardless of how many times we pushed `:latest`. The fix went from -2640 to +3566.

### `ulimits` + `seccomp:unconfined` eliminated tail-latency HTTP errors

Without these, ~1-3 requests per test would time out at 2001 ms during the k6 ramp. The fix in compose alone took detection score from 2713 → 2819 (+106 points).

## What's next (open questions, post-phase-13)

After three rounds of Mac Mini testing (#3709 = 3065, #3727 = 3712, #3768 = 3823), the score landscape is:

```
final_score 3823 / 6000  (top-cluster target: 5500-5900)
├── det_score 2819 / 3000  (SATURATED at max modulo the 1 structural FN)
└── p99_score 1003 / 3000  (all remaining headroom is here — 100ms → 1ms = +2000)
```

Things we've **already eliminated** (don't retry without new evidence):

- **int8 quantization** (lecture 06). Linear per-dim int8 → FP=51/FN=62 on full test set. Adaptive per-dim normalization changes geometry, deviates from labels. Spherical k-means is a centroid-only trick that doesn't fix per-vector noise. **Won't be a win in our exact-grid pipeline.**
- **float32 storage** (lecture 06). The 1 structural FN is not a quantization artifact; float32 doesn't eliminate it. Doubled cache pressure also makes latency 18 % worse locally.

Things still **on the table**:

1. **Load shedder tuning sweep** (lecture 07). After phase 13's first result, sweep `SHED_SLOTS` ∈ {1, 2, 4, 8} and `SHED_TIMEOUT_MS` ∈ {1, 3, 5} from the compose to find the operating point with the best score. No code changes required.
2. **shed-as-deny** instead of shed-as-approve. Expected weighted-E per shed is 0.56 (vs 1.32 for approve). One-line change if shed-as-approve doesn't carry its weight.
3. **Find and fix the 1 structural FN.** It's not quantization (confirmed in phase 12). Likely the parser produces a different float64 for one specific timestamp/amount combo on test-data.json entry 5472. Bisect: dump our parsed query vs `vectorizeSlow`'s output for that entry. If they differ, fix the fast parser. Free +106 detection points.
4. **SIMD asm distance kernel.** Phases 4-6 retired this. The early-exit kernel doesn't compose with 8-wide block ops cleanly, but a **VPDPBSSD** (AVX-VNNI) or **AVX-512** kernel can mask early-exit per lane — Mac Mini Haswell has neither, so this is a dead end without a hardware upgrade. Marking as "won't fit Mac Mini."
5. **Look at top contestants once more.** The journey doc references `josehenrique-dev-Go (#23, p99 1.76ms)` — we haven't surveyed his full repo. A focused dive into how he gets sub-2ms with similar memory budget could surface a structural trick we're missing (a different data layout, a faster JSON parser, kernel work).
6. **Profile the Mac Mini run.** Capture `pprof` from inside the container during a real bench run on the bot. We've been profile-blind since we can't replicate Mac Mini contention locally.

What we should **not** do without more data:

- Algorithm rewrites (we just did one with grid revival, returns are diminishing).
- Compose CPU split sweeps (top-Go submissions converged on 0.10 / 0.45 / 0.45; we shouldn't fight it).
- Switching from custom raw HTTP to fasthttp/etc. (phase 6 showed this isn't the bottleneck).

The fundamental insight from this journey: **the top of the leaderboard is achieved by getting many small things right, not by one heroic optimization**. The compose-level tweaks (pull_policy, ulimits, seccomp) gained us ~150 points. The algorithm change (k-means → balancedSplit → grid revival) was net +700 points end-to-end. The block-major SIMD kernel was a wash on the Mac Mini despite being 8× wider locally and was retired in phase 9. **Quantization-width changes (int8, float32) both empirically regressed** despite our theoretical expectations — see lecture 06 for why. Top Go submissions clustering between 5500-5900 final are doing **all** of these well, not picking one.

## Phase 9 — what changed and what we learned

- **Approximation does not pay under this scoring.** The flat-IVF detour cost us ~600 points on `score_det` (FP=4, FN=4 vs FP=0, FN=1). The supposed latency win never materialized because IVF without per-cell pruning scanned the *same* ~11700 vectors regardless of how close they were to the query, while grid+LB scans ~3000-6000 *and* most of them only the first 2-3 dims (early-exit).
- **The right SIMD-vs-scalar question is "what shape is the inner loop?", not "is the CPU faster at vector ops?".** SIMD wins when every candidate runs to completion. Early-exit means most candidates abort at dim 1-2 — and you can't abort half the lanes in a SIMD block. We retired `blockScan8AVX2`. The cost of dropping it locally is negligible; the maintenance cost was real.
- **The partition layout is sparser than the bit count suggests.** 32 partition keys exist; only 12 have data. `online + card_present` never coexist; sentinel-5-XOR-sentinel-6 never coexist (transactions either have full last-tx data or none). So we really have *6 non-sentinel + 6 double-sentinel = 12 working partitions*. The other 20 are dead code paths.
- **Cycle-sort direction trap is real.** The in-place permutation routine for the partition step was the bug-prone part of the rebuild. The right form: swap row `i` with row `target[i]`, *also swap their target entries*, repeat until `target[i] == i`. Walking the *inverse* permutation produces near-correct results that are wrong in subtle ways.
- **Bench harness noise is huge.** Same image, different runs of the local k6 bench produce p99 = 80 ms one minute and p99 = 800 ms the next, depending on which CPU cores k6 lands on. Single-run numbers are barely signal. The Mac Mini will be even noisier per the phase-1-1b note above.
