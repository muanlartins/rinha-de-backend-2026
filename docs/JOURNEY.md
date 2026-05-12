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
| 6 | Custom HTTP server (drop net/http) | TBD | TBD | The big win we were missing — top Go submissions write the response bytes directly to the socket, no http.Request alloc |

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

## What's next

1. **Custom HTTP server**: drop net/http entirely, port josehenrique-dev-Go's pattern. Biggest p99 win remaining.
2. **Float32 storage**: eliminates the 1 FN, ~doubles kernel throughput via VFMADD231PS. Tight on memory.
3. **Smaller cluster sizes (K=256+)**: less work per query but slower image build.
4. **Tighter LB pruning + bbox per block** (not just per cluster): cut more vectors from scan.

The fundamental insight from this journey: **the top of the leaderboard is achieved by eliminating Go-runtime overhead, not by inventing exotic algorithms**. HNSW (rank 3) and grid+SIMD (rank 23) cluster within 5× of each other; the rest is HTTP serving, memory layout, and reducing per-request allocations.
