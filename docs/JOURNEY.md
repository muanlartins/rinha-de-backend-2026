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

## What's next (open questions)

1. **Float32 storage**: eliminates the 1 quantization FN, plus ~doubles kernel throughput via VFMADD231PS. Memory cost is 168 MB — just over our 167 MB cgroup. Doable if we eliminate other overhead.
2. **PGO with a real profile from the rinha bot**: probably 5-15% across the board. Need to figure out how to capture profiles from the bot.
3. **Pre-sort cluster vectors by some discriminating dim** so early-exit kicks in faster.
4. **Variance-ordered dim layout in the asm kernel** (top-Go does this — STEP(10), STEP(12), STEP(4) rather than STEP(0..15) in order — to make partial sums exceed the threshold faster).

The fundamental insight from this journey: **the top of the leaderboard is achieved by getting many small things right, not by one heroic optimization**. The compose-level tweaks (pull_policy, ulimits, seccomp) gained us ~150 points. The algorithm change (k-means → balancedSplit) gained us ~300 expected points. The block-major SIMD kernel was a wash on the Mac Mini despite being 8× wider locally. Top Go submissions clustering between 5500-5900 final are doing **all** of these well, not picking one.
