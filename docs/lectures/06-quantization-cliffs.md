# Lecture 06 — Quantization Cliffs

> Goal: capture the empirical results of trying int8 and float32 storage so future sessions don't repeat the same experiments. **Both were dead ends in this codebase's algorithmic shape**, and the reasons are non-obvious from theory. Read this before proposing any quantization change.

## 1. What we tried, in two sentences

We tried **int8 quantization** (halve the dataset to 42 MB, hoping for cache-pressure wins) and **float32 storage** (kill the structural FN, accept the 168 MB memory cost via shared-tmpfs mmap). Both were net negative on the rinha scoring formula, for reasons specific to our partition-grid pipeline rather than fundamental properties of the techniques.

## 2. int8 attempt — FN explosion, FP=51 / FN=62

### What we changed

- `dataset.Vectors` from `[]int16` to `[]int8`.
- `QuantScale = 127`, `SentinelInt = -127`.
- Distance accumulator `int64` → `int32` (max sum across 14 dims of (255)² ≈ 916k fits easily).
- All bbox/bounds/centroid types tracked: `int16` → `int8` (with `int32` accumulators where summing).

### What it cost

On the full 54,100-entry test set, the int8 grid algorithm produced **FP=51, FN=62** compared with int16's **FP=0, FN=1**. Scored:

```
int16 (phase 10): E = 1 + 1·5 = 8 (with the 1 Err from queue contention)
                  ε = 8/54100 = 0.000148
                  score_det = 1000·log10(1/0.001) - 300·log10(9)
                            = 3000 - 286 = 2714

int8 (projected): E = 51 + 62·3 + 1·5 = 242 (with same 1 Err)
                  ε = 242/54100 = 0.00447
                  score_det = 1000·log10(1/0.00447) - 300·log10(243)
                            = 2350 - 716 = 1634
```

**A −1080-point detection hit.** The local-bench latency win was only ~10 % (320 → 287 µs/op on Apple Silicon), and even projecting a 2× Haswell win on the smaller L3 (100 → 50 ms p99 ≈ +300 score), the net was −778. Reverted.

### Why our int8 looked worse than the survey predicted

The survey said top-Go submission `JosineyJr/rdb-26` runs int8 with detection penalty roughly 4× ours. We expected ~5 FNs, got 62. The discrepancy comes from **how we quantize**, not from int8 vs int16:

- **Rinha normalization is `MaxAmount = 10000`** for dim 0. The actual transaction distribution is heavy-tailed below $200, so after dividing by 10000 most amounts cluster in `[0, 0.02]` — that's `[0, 2.5]` in int8 levels. We're using 2 distinct int8 values for 90 % of the data.
- **Per-dim linear int8 collapses the long tail.** Two transactions at $50 and $200 land on int8 values 0 and 2 → distance² = 4. The same two in int16 land on 159 and 638 → distance² = 229,441. The relative ranking of "similar amount" vs "very different amount" gets squashed flat in int8.
- **Josiney avoids this with IVF k-means clustering, not per-dim adaptation.** Their `spherical k-means with magnitude=127 renormalization` is a centroid-only trick — it keeps centroids representable in int8 during k-means iteration. The reference *vectors* still suffer the same per-dim linear quantization noise we have. But Josiney's pipeline is **approximate** (nprobe=8 of 1024 clusters), and the FN damage from int8 happens to land roughly in the same range as the approximation noise from nprobe — so the int8 vs float ranking divergence shows up as a constant cost they were already paying.

In our **exact** grid-LB pipeline, int8 noise becomes the *only* source of error, and it dominates.

### Approaches we considered and rejected

| Idea | Why we rejected |
|---|---|
| Per-dim adaptive normalization (e.g., `amount/200` instead of `amount/10000`, clip the tail) | Changes the relative weighting between dims in L2 distance. Rinha labels were generated from the ORIGINAL float-linear geometry; any deviation produces ranking divergences vs labels = new FNs. |
| Non-linear remap (sqrt, log) per dim | Same issue: distorts L2 geometry, deviates from label ground truth. |
| Spherical normalization of the dataset (`v / ||v||`) | Sentinel dims (-1) break this. Binary dims (`is_online`) become tiny after normalization. |
| Hybrid int16 (storage) + int8 (SIMD inner loop) | Adds complexity. Storage cost is unchanged. Pre-empts no real bottleneck. |

### Takeaway

**int8 doesn't fit our pipeline.** It only works for approximate algorithms where the int8 noise is dominated by the approximation noise. We are exact; we have no headroom to absorb int8 noise.

## 3. float32 attempt — same detection, worse latency

### What we changed

- `dataset.Vectors` from `[]int16` to `[]float32`.
- `Quantize()` became a clamp + sentinel passthrough — no actual quantization, just `[0, 1] ∪ {-1}`.
- All math from `int*` to `float32`. Threshold `1<<62` → `1e30`.
- Added `LoadIndexMmap()` using `golang.org/x/sys/unix.Mmap` with `MAP_SHARED|MAP_POPULATE` (the latter only on Linux — `mmap_linux.go` vs `mmap_other.go` build-tagged stubs).
- Added a `data-init` style execution path: write index to a tmpfs volume from one container, exit, let API containers mmap the file from shared page cache.
- Index format bumped to v6: 16-B header (magic+version+count) + partition starts/counts + **vectors slab at a fixed offset** so mmap can locate it cleanly + labels + per-partition grid records.

The cgroup-accounting trick that makes this viable: under cgroup v2, **tmpfs pages are charged to the cgroup that wrote them**. If the init container writes the 168 MB index then exits, its cgroup is destroyed and the pages get reparented to root — they no longer count against any API container's `memory.current`. API containers reading via mmap see the shared page cache but pay zero against their cgroup limit.

### What it cost

Tested `TestGridFullDataset` with the float32 build expecting the structural FN to vanish:

```
54100 entries: parse_fails=0 FP=0 FN=1
```

**Identical to int16.** The 1 FN is NOT a quantization artifact. The journey doc's claim ("entry 5472 flips because int16 loses precision near a quantization boundary") was wrong. The actual root cause is either:

1. Our `parseFloat` (in `internal/vector/fast.go`) produces a slightly different float64 from rinha's reference parser for one specific input, or
2. There's a 5th-NN tie on entry 5472 broken differently by our top-K cascade vs rinha's algorithm.

We didn't bisect to find out — whichever it is, it survives the int16 → float32 transition because both parsers and both algorithms have the same divergence.

Plus latency:

```
M4 Pro local handler bench: 378 µs/op (float32) vs 320 µs/op (int16+PGO+centroid)
                          = 18 % slower locally
```

Doubling the dataset from 84 to 168 MB doubles the effective cache pressure on a workload that's already memory-bound. On Haswell with 6 MB L3, the slowdown is probably worse.

### Approaches we considered

- **mmap-shared without float32 (keep int16, share the 84 MB)**: would save 70 MB per replica that we don't actually need (we're not memory-bound at 137 MB heap). No benefit.
- **Investigate the 1 FN's actual root cause and fix it in the parser**: worth doing but tangential to quantization; cheaper than float32 if the bug is in `parseFloat`. *Open follow-up.*
- **Use float32 only for borderline queries (rerank)**: complex two-stage search. Significant engineering.

### Takeaway

**Float32 doesn't fix the 1 FN.** Going to float32 is also pure regression on latency. The next attempt to eliminate the FN should investigate the parser before throwing memory at storage width.

## 4. The shape of "what's left" after these two dead ends

The empirical landscape now:

```
       ┌─────────────────────────────────────┐
       │ Detection (max 3000):               │
       │   Phase 10 = 2714                   │
       │   Saturated at 2820 modulo the 1 FN │
       │   ─── float32 / int8 won't move this│
       └─────────────────────────────────────┘
       ┌─────────────────────────────────────┐
       │ Latency p99 (max 3000):             │
       │   Phase 10 = 999 (p99 = 100 ms)     │
       │   ↑↑↑ all remaining headroom here   │
       └─────────────────────────────────────┘
```

So future work targets **p99 latency on the Mac Mini**, not detection accuracy. And on the Mac Mini, p99 ≈ queue wait + CPU work. Our local M4-Pro CPU work is 320 µs; Mac Mini Haswell CPU work is probably 2-5× that = 1-2 ms. So **80-95 % of our 100 ms p99 is queue wait, not algorithm time**.

This is why the **load shedder** (lecture 07) is the next-priority experiment — it directly attacks queue wait, which storage-width optimizations can't touch.

## 5. The fundamental tension

The rinha scoring rewards exactness *and* latency. The top-cluster contestants navigate this by:

1. Using fast-enough algorithms that they don't *need* approximation tricks (their CPU work is < 1 ms, so queue wait is naturally tiny).
2. Adding a small load shedder to clamp the worst tail spikes.
3. Accepting a small constant detection penalty from quantization, offset by the latency win that goes WAY beyond what we can reach with exactness.

Our current shape is the opposite: we're exact and slow. To match the top, we either need to:

- **Become much faster on the inner algorithm** (SIMD asm, smaller dataset, fewer cells to visit) — this is a structural code change.
- **Accept the same detection penalty everyone else does** by going approximate (IVF nprobe, HNSW, etc.) and use the freed latency budget — this is the "join the herd" strategy.

Neither path is what we've been on. Both are open for the next session.
