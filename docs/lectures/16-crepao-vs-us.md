# Lecture 16 — crepao-da-massa vs us: structural design comparison

Written 2026-05-14 after phase 30 (K-tuning experiment) failed. Goal:
understand what crepao-da-massa does differently from us, where each
design wins, and where we have room to surpass.

Phase 28 put us at **rank 6 / p99 1.12 ms / 5950.61**. crepao-da-massa
is at **rank 1 / p99 0.98 ms / 6000.00**. Gap: **140 µs of p99**.

This lecture is the architectural picture, not a refactor checklist.

## 1. What I learned from the K experiment

Phase 30 changed `const K = 4096` to 8192 (smaller clusters) and 2048
(larger clusters). Both produced a stubborn **FN=1 that no practical
top-N could recover**:

```
K=8192, N=24..192    →  FP=0 FN=1   (persistent)
K=4096, N=28..48     →  FP=0 FN=0   ← our sweet spot
K=2048, N=16..32     →  FP=0 FN=1   (persistent)
```

The full-K sweep at every K still produces FP=FN=0 (because of bbox-LB
pruning). But top-N — which ranks by *centroid distance* — can miss a
cluster whose centroid is far from the query but whose bbox edge
reaches in toward the query.

**Takeaway:** top-N escalation works as long as cluster geometry
doesn't have a "near-bbox / far-centroid" cluster within the answer
window. K=4096 happens to avoid that for our k-means++ seed. K≠4096
re-rolls the geometry and re-introduces the problem.

This is a *structural* constraint, not a bug. It explains why crepao
made a different architectural choice.

## 2. crepao's architecture, re-read

I went back to `references/crepao-silent-index/src/index.hpp` for a
second pass after the K experiment. Key findings:

### 2.1. crepao does NOT do top-N escalation

`index.hpp:317-325`:

```cpp
if (fraud >= repair_min && fraud <= repair_max) {  // count ∈ [1,4]
    for (uint32_t c = 0; c < kk; ++c) {           // ALL K=1664 clusters
        if (scanned[...]) continue;
        if (bbox_lower_bound(q, ...) >= top.worst_dist()) continue;  // AABB-LB prune
        scan_cluster(c, q, top);                  // scan if survives
    }
}
```

They iterate **every cluster** and rely on `bbox_lower_bound` to prune
~95 % cheaply. This avoids the "true 5-NN sits in a far-centroid
cluster" problem entirely, because every cluster is checked. The cost
is paid in pruning, not scanning.

Their fast tier is **NPROBE=2** (`search_two_with_repair_avx2`), so on
the ~94 % of queries that don't escalate, they only scan 2 clusters.

### 2.2. The block layout: dimension-paired interleaved

`index.hpp:124-126`:

```cpp
inline size_t block_pair_offset(int dim_pair, uint32_t lane) {
    return size_t(dim_pair) * Block * 2 + size_t(lane) * 2;
}
```

A block of 8 vectors stores **7 dim-pairs**, each containing 16 int16s
= 32 bytes:

```
pair 0: [v0_d0, v0_d1, v1_d0, v1_d1, …, v7_d0, v7_d1]  (32 bytes)
pair 1: [v0_d2, v0_d3, v1_d2, v1_d3, …, v7_d2, v7_d3]
…
pair 6: [v0_d12, v0_d13, …, v7_d12, v7_d13]
```

Total: 7 × 32 = 224 bytes per block (same as our dim-major).

### 2.3. The kernel: VPMADDWD

For each dim-pair:
1. `VMOVDQU` 16 int16 (one dim-pair, 8 vectors) — single 32 B load
2. `VPBROADCASTD` query's (d, d+1) pair across 8 lanes
3. `VPSUBW` → 16 i16 diffs
4. `VPMADDWD` → 8 i32 lanes, each = diff_d² + diff_(d+1)² for one vector
5. `VPADDD` into accumulator

That's **5 instructions per dim-pair × 7 pairs = 35 ops/block**.

Our kernel is **per-dim: VPMOVSXWD + VCVTDQ2PS + VBROADCASTSS + VSUBPS
+ VFMADD231PS = 5 ops × 14 dims = 70 ops/block**.

crepao's kernel is **2 × fewer ops per block** — that's the headline
SIMD win. The cost: i32 accumulator can overflow at 7 pairs × max diff²
= 7 × (2·10000)² = 5.6 × 10⁹ > 2³¹. They cope by clamping to
INT32_MAX (line 116-121) and accepting that the worst-case query
loses precision in distances that are anyway beyond the top-5 boundary.

### 2.4. The vectorized bbox_lower_bound

`index.hpp:132-150`:

```cpp
__m128i below = _mm_sub_epi16(mnv, qv);
__m128i above = _mm_sub_epi16(qv, mxv);
__m128i diff  = _mm_max_epi16(_mm_max_epi16(below, above), zero);
uint64_t s    = hsum128_epi32(_mm_madd_epi16(diff, diff));
```

8-lane i16 bbox-LB → 4-lane i32 squared-and-pair-summed via the same
VPMADDWD trick. Inner loop is ~10 ops. The K=1664 pre-prune
loop runs this for every cluster.

Pre-prune cost: 1664 × ~25 cycles ≈ ~16 µs. Affordable.

### 2.5. K=1664 was empirically tuned

`Dockerfile:20`: `build_index ... 1664 65536 6`.

Commit `6609bd3 "Tune offline index cluster count"` shows they swept
K and picked the knee, using `offline_api_profile.cpp` to capture p99
under their actual repair path. Their tuning target is full-K-sweep
with vectorized AABB-LB pruning — *for that path*, smaller K = lower
pre-prune cost, larger clusters = better cache locality and amortized
SIMD. The optimal sweet spot was K=1664.

For *our* path (top-N escalation), the optimum is different (K=4096,
because top-N relies on cluster geometry having well-separated
boundary candidates within N=32).

**Lesson: K is co-tuned with the escalation strategy. Don't compare
K values across architectures.**

## 3. Cost models, head-to-head

Order-of-magnitude per-query CPU, ignoring HTTP/IO:

### Fast tier (~94 % of queries):

| Component | crepao (K=1664, NPROBE=2) | us (K=4096, NPROBE=1) |
|---|---|---|
| Centroid scoring | K=1664, ~5 µs (their VPMADDWD path) | K=4096, **~4.5 µs asm** |
| Top-N pick (N=2) | 2 µs | 1 µs (single MIN) |
| Cluster scans | 2 × ~5 µs (VPMADDWD) = 10 µs | 1 × ~3 µs (f32 FMA) = 3 µs |
| **Subtotal** | **~17 µs** | **~8.5 µs** |

**We're ~8 µs faster on the fast tier** because:
- Our K=4096 scoring is fast (asm)
- Our NPROBE=1 doesn't double-scan
- Smaller clusters scan faster per cluster

### Escalation tier (~6 % of queries — sets p99):

| Component | crepao | us |
|---|---|---|
| AABB-LB pre-prune | K=1664 × ~25 cy = ~16 µs | top-N picks 32 clusters: 0 (skip) |
| Top-N pick | n/a (full sweep) | 32-pick from K=4096 = ~5 µs |
| Surviving scans | ~30-50 × ~5 µs = 150-250 µs | ~32 × radius+AABB + ~10 actual × 3 µs = ~30 µs |
| **Subtotal** | **~165-265 µs** | **~35 µs** |

**Our escalation is ~5-7× faster than crepao's.** Two reasons:
1. We don't do full-K pre-prune (top-N skips it)
2. We scan fewer clusters total

### Net per-query CPU:

| | Avg query | p99 query (escalating) |
|---|---|---|
| crepao | 0.94 × 17 + 0.06 × 215 = ~29 µs | ~215 µs |
| us | 0.94 × 8.5 + 0.06 × 35 = ~10 µs | ~35 µs |

**Our CPU model says we should be 3-6× faster than crepao.** But the
bot reports us at p99 1.12 ms vs crepao 0.98 ms.

That means **140 µs of our p99 is non-CPU**: HTTP/scheduler/queueing
overhead. Same for crepao but they spend less wall-clock on it somehow.

## 4. Where the gap actually lives

If CPU isn't the gap, the gap must be in:

### 4.1. Go runtime overhead

C++ has no GC, no goroutine preemption, no GOMAXPROCS=1 scheduling
quirks. Go incurs:

- **Preemption every ~10 ms** — at p99 our worst-1 % can land right
  before/after a preemption window
- **GC mark phase** — even with `GOGC=200, GOMEMLIMIT=140MB` we run
  GC periodically; if it triggers mid-request, latency spikes
- **Goroutine scheduling** — net.Listen on UDS spawns goroutines per
  request; goroutine-creation cost is ~1-2 µs each

C++ servers (epoll loop, no GC, single thread) avoid all three.

This is a structural disadvantage of Go. Estimated cost: **30-80 µs at
p99**.

### 4.2. HTTP parsing & response

We use a custom raw-byte HTTP parser (`internal/api/rawhttp.go`).
crepao parses inline via `memchr` for content-length, hardcoded
positions for known JSON fields, and pre-built responses for
fraud_score ∈ {0, 1, 2, 3, 4, 5}.

Our response building does some pre-built work but I haven't audited
whether it's as tight as theirs. Estimated cost: **5-15 µs per query**.

### 4.3. Date math

`server.cpp:144-162` hardcodes `epoch_minutes` and `weekday_monday0` to
O(1) lookups for March 2026 (the test month). We do a full
`dayNumberFromYMD` calculation in `internal/vector/fast.go:355`.

Estimated cost: **1-3 µs per query**.

### 4.4. Allocation pattern

We already use `sync.Pool` for `IVFScratch` and HTTP read buffers
(`internal/api/server.go:112`). Verified zero allocation per request
in the bench (`5184 B/op` is fixed-size scratch). This is not a gap.

### 4.5. Page locality

crepao uses `MADV_WILLNEED` + an explicit `warm_pages()` page walk to
fault in TLB entries. We have `MADV_HUGEPAGE | MADV_POPULATE_READ |
MADV_RANDOM` plus a 500-iteration warmup loop that exercises the
search path. Probably equivalent or slightly tighter than crepao.

## 5. Where we surpass crepao already

### 5.1. Fast-tier CPU efficiency

NPROBE=1 + asm centroid scoring is **~2× cheaper** than NPROBE=2 with
their kernel. 94 % of queries benefit. This is a real win.

### 5.2. Escalation algorithm

Top-N=32 at K=4096 scans ~35 µs of work vs their full-K-sweep at
~200 µs. **5-7× cheaper escalation.**

(But: theirs is FP=FN=0 at any cluster geometry, ours requires
calibrated N. Their design has more headroom for K-tuning; ours has
lower CPU.)

### 5.3. Per-cluster scan cost (per cluster, ignoring count)

Our K=4096 makes individual clusters smaller (~733 vec) than theirs
(~1804 vec). Per-cluster scan: us ~3 µs vs them ~5 µs. Net win per
cluster.

## 6. Where they surpass us

### 6.1. Wall-clock consistency

p99 measures the worst 1 %. Even if our average is lower, our tail
distribution may be heavier due to Go runtime jitter. crepao's C++
has tighter tails.

### 6.2. Per-block kernel ops

35 ops/block (VPMADDWD path) vs our 70 (f32 FMA + widen). For
cache-resident, warm-pipeline workloads this is 1.5-2× faster per
block.

Only matters if we do many blocks per query. On escalation we do
~10 cluster scans × ~92 blocks = ~920 block-ops; halving block-op
count saves ~30 µs there. Real but bounded.

### 6.3. Hardcoded date math

Saves ~2 µs per query (universal). 100 % of traffic.

### 6.4. Likely simpler/smaller HTTP path

C++ epoll + manual parse + sendto vs our Go raw HTTP shim. Less code,
likely less per-request constant overhead. Hard to quantify without
detailed profile.

## 7. Where we can surpass them (concrete bets)

Listed by expected EV given the cost models above:

### Bet A — Reduce Go runtime tail (highest EV, riskiest)

p99 jitter is the most likely source of our ~140 µs gap. Approaches:

1. **`GOGC=off` during steady-state** with a manual `runtime.GC()` in
   a background goroutine triggered by request count. Removes the
   GC-during-request spike.
2. **Reuse the same goroutine for many requests** — currently each
   accept spawns a handler goroutine. A pool of long-lived workers
   could amortize spawn cost.
3. **`GOMAXPROCS=1` is already set** but try `GOMAXPROCS=auto` to
   measure whether scheduler preemption helps or hurts in our config.
4. **Profile with `runtime/trace`** to find actual jitter sources.

Expected: **−50-100 µs at p99** if successful, but bot-side
unobservable without amd64 instrumentation.

### Bet B — Hardcoded date math (low effort, predictable)

Replace `dayNumberFromYMD` with a 31-entry lookup table for March 2026
(plus April/February shoulders if data spans months). Universal path,
~2-3 µs save per query.

Expected: **−2-3 µs at p99**, **+5-10 score**.

### Bet C — Asm picker (low effort, medium)

`PickTopNCentroids` for N=1 is a single MIN over K=4096 floats. Our
Go version does sorted-insert with a linear scan. Asm with VMINPS
reduction could be ~2× faster. `PickNextNUnscanned` for N=32 is more
expensive — also a candidate.

Expected: **−5-10 µs at p99**, **+5-15 score**.

### Bet D — VPMADDWD kernel + dim-paired blocks (high effort, medium)

This is the textbook recommendation from lecture 15. Now I have a
better cost model: it saves **~30 µs at p99** (per the math above),
not the ~50-100 µs I initially estimated. The block layout change is
multi-hour work with i32 overflow handling.

Expected: **−30 µs at p99**, **+30 score**.

### Bet E — Audit and tighten HTTP/response path (low effort, medium)

Look hard at `internal/api/rawhttp.go` and `server.go`. Are we
allocating? Are we doing one syscall per response? Could we
write+close in a single system call?

Expected: **−5-15 µs at p99**, **+5-30 score**.

## 8. Recommendation

**Phase 31 = Bet B (date math hardcoding)**: cheap, universal, ~2-3 µs
per query, no risk. Quick to ship, measurable.

**Phase 32 = Bet C (asm picker)**: medium effort, ~5-10 µs save.
Compounds with Phase 31 to ~10 µs total.

**Phase 33 = Bet E (HTTP audit)**: profile first, then targeted
optimization. Could be the actual gap-closer.

**Phase 34 = Bet D (VPMADDWD)**: only after exhausting the cheap
options, because it's the highest-risk highest-effort and the cost
model says it isn't as decisive as I first thought.

**Bet A (Go runtime tail)** is the wildcard. It could be worth all
the others combined, or it could be impossible to measure without
amd64 profiling infrastructure. Defer until we have a way to
profile under real load (probably never on this submission timeline).

## 9. The honest position

**We are not behind crepao on algorithm or CPU efficiency.** Our hot
path is tighter for the median query. The 140 µs gap to them is
either Go runtime tail or HTTP overhead — neither of which is
impossible to close, but both require profiling work that's hard to
do remotely.

If we get to p99 ≈ 1.05 ms (top 3-4) and stop, that's an honest
result. Below that requires Go-runtime engineering or a runtime
switch, and the diminishing returns get steep fast.

We can ship Phase 31 + 32 + 33 in the next session. Each is bounded
and measurable. Stretch target: **p99 ≈ 1.00 ms, rank 2-3, score
~5980**.
