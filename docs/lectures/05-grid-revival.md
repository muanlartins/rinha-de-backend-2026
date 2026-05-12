# Lecture 05 — The Grid Revival

> Goal: explain *why* we are tearing out the flat-IVF code and going back to the partitioned grid algorithm described in lectures 01–02. Concept-level, not line-by-line — the next session should be able to read this and understand both what changed and the engineering reasoning behind it.

## 1. Where this lecture sits in the journey

Up until this point we tried, in order:

1. **Phase 1 — Exact partitioned grid + scalar early-exit.** Mac Mini score **3566**, p99 ~140 ms. The first thing that worked. FP=0, FN=1, Err=2. The 1 FN is structural (quantization), not algorithmic.
2. **Phase 2–4 — IVF with per-cluster LB pruning + block-SIMD kernel.** Score **3410–3452** on the Mac Mini. Faster *per scan* but each scan was longer because the algorithm visited more vectors. We added a fancier kernel and lost ground.
3. **Phase 7 — Flat-IVF with Lloyd k-means, K=4096, nprobe-only.** Local regression to score **2888**. K-means produces meaningless centroids on the binary dims, and "probe top 32" silently misses neighbors in adjacent clusters. **Approximate**.
4. **Phase 8 — Flat-IVF with balancedSplit (KD-tree median split), K=8192, nprobe=32+128.** Still approximate. Still FP=4 / FN=4 locally.

**The pattern:** every move *away* from the partition+grid algorithm hurt us, either on accuracy (IVF approximation eats 4+ detection errors) or on speed (IVF without per-cell pruning scans more vectors than necessary).

This lecture is about reverting that drift, with the lessons from QRust's final form (`search-s3b.ts`) baked in: their 0.855 ms p99 on Apple-Silicon-via-Rosetta uses pure scalar TypeScript over the same data structures we already half-built and threw away. The Go port of that approach is what this lecture describes.

## 2. The big idea, restated

```
Reference dataset
   │
   ├── partition by 5 discrete bits (dims 9, 10, 11, sentinel-5, sentinel-6)
   │      ↓
   │   32 disjoint partitions; each query touches exactly one
   │
   └── within each partition, lay out a percentile-binned grid
          on 3 high-variance continuous dims
              ↓
          up to 1024 cells; each cell knows its AABB

Query path (the one ~1 ms we have):
   ├── compute partition key  (constant time)
   ├── for each cell in that partition: compute lower-bound dist (AABB) 
   ├── sort cells by lower bound  (insertion sort, sub-µs)
   ├── walk cells in LB order; break when LB ≥ current-top-5 worst
   │     ↳ inside each cell, scan vectors with early-exit kernel
   └── tally fraud labels among top-5
```

The whole thing is **exact**. There is no approximation anywhere: a cell can be pruned only when we have **mathematical proof** (via the AABB bound) that no vector inside it can beat the current 5th nearest. The same holds for the early-exit kernel — once partial distance exceeds the threshold, the full distance can only be larger, so the vector can be skipped without changing the answer.

## 3. Why this beats flat-IVF on *both* axes

Two scoring components matter: **detection** and **latency**. Flat-IVF gives up accuracy for speed. Partition+grid gives up neither.

### 3a. Accuracy: where the IVF errors came from

Flat-IVF with `nprobe=32` says: "search the 32 clusters whose centroid is closest to the query, ignore the rest." That works *most* of the time. It fails when:

- The query sits near a cluster boundary, so its true 5th nearest is in cluster 33 (or 100, or 500). We never look there.
- The query is in a region where the 8192 balanced clusters were split with somewhat arbitrary borders (recursive median split on max-variance dim). One vector in the "wrong" cluster, and a top-5 slot gets the wrong label.

The "ambiguous expansion" (`nprobe=128` if fraud_count ∈ {2,3,4}) catches some but not all of these — by design, because the threshold to trigger expansion is itself heuristic.

The grid algorithm has none of this. Every cell in the partition gets a *lower bound*. We either:
- Visit it (LB < topD4) and find the exact distances of its members, possibly updating top-5, or
- Skip it (LB ≥ topD4) with **proof** that nothing in it could win.

There is no fence-sitter cluster that gets unfairly pruned. The output is bit-identical to brute-force-over-the-whole-partition, which is itself bit-identical to brute-force-over-the-whole-dataset (because partitions are correctness-preserving — see lecture 01 §3, dims 9–11 + sentinels contribute ≥1 to squared distance, larger than any reasonable 5th-nearest).

The **only** detection error left after this is the quantization FN (entry 5472 of test-data.json). That one is inherent to int16 storage and can't be fixed without going to float32, which doesn't fit the memory budget.

### 3b. Latency: how the cell pruning beats the cluster pruning

Flat-IVF visits 32 clusters × 366 vectors = **11,712 vectors** per query, *every* query, computing full 16-dim distance via the SIMD kernel.

Grid + LB pruning visits, on average, **3,000–6,000 vectors** per query. The savings come from two layers compounding:

1. **Cells are smaller than clusters.** A typical partition has 30k–300k vectors split into ~600–800 non-empty cells (out of 1024). Average cell holds 100–300 vectors. So the granularity of "decide to scan or not" is finer.
2. **Cells are tighter spatial regions.** An AABB over 200 vectors is far tighter than an AABB over 366 — and the LB grows quadratically with axis offset, so a tighter box prunes much more aggressively.

After the partition prefilter (32× reduction off 3M), the cell LB pass typically prunes 80–95% of the remaining cells *before* any vector is touched. The early-exit kernel inside the surviving cells then bails most candidates at dim 1 or 2 because by that point the top-5 is already tight.

QRust's numbers, transferred: 60 µs for LB sweep, 30 µs for cell sort, 500 µs for early-exit scan. ~600 µs of search out of an 855 µs budget. The rest is JSON parse + HTTP framing.

### 3c. The "free wins" we kept anyway

Three things from the IVF/SIMD experiments are valuable in *any* algorithm:

1. **Block-major contiguous storage.** We had the right idea about cache locality — within a cluster, vectors are contiguous and dim-major-blocked so SIMD reads are sequential. We keep contiguous storage, just per-cell instead of per-cluster, and switch dim-major → row-major (because early-exit reads dims in order from each vector, then jumps to the next vector).
2. **Pre-built index baked into the Docker image.** No k-means at runtime is the only way to pass the bot's health check. The grid build is even faster than k-means (no iteration), so this stays trivially.
3. **`pull_policy: always` + `ulimits` + `seccomp:unconfined`.** Nothing to do with the algorithm. Keep all of them.

## 4. Why we are *not* keeping the SIMD distance kernel

This is the optimization that looked great in micro-benchmarks and hurt us in practice. The asm kernel `blockScan8AVX2` computes 8 squared distances over a full 16-dim block per call, no branches, fixed cost.

In the **grid+LB+early-exit** algorithm, the property "fixed cost per vector" is **bad**:
- For ~50% of candidate vectors, partial distance exceeds topD4 by dim 1 or 2. Scalar early-exit ends the work right there — call it ~4 ns. SIMD pays for all 14 dims — ~10 ns minimum even on a hot cache.
- Vector-level early-exit and 8-wide SIMD do not compose. You can't bail out of one of the 8 lanes; you compute all 8 to completion or none.
- "Block-level early-exit" (skip an entire block of 8 if its best lane exceeds topD4) is possible but adds branches that hurt throughput in the common case.

So we go scalar. **In the inner loop, we have BCE hints, an unrolled-and-flat dim list, and one branch per dim.** The Go compiler turns that into something like 11 `MOVSWL` + 11 `IMUL` + 11 `ADD` + 11 cmp/conditional-jump, of which most candidates execute only the first 2-3 before bailing. That's ~5 ns per *full* vector, much less for the pruned majority.

If we ever want a SIMD kernel back, the right shape is **AVX-512 with masked early-exit on 8 lanes** — and the Mac Mini's Haswell doesn't have AVX-512. Don't optimize for hardware we don't have.

## 5. What the new index file holds

The persisted artifact (built at Docker build time) needs to carry:

| field | size | purpose |
|---|---|---|
| magic + version | 8 B | sanity check |
| count | 8 B | total vectors |
| partitionStarts[32], partitionCounts[32] | 256 B | locate each partition's slab |
| vectors (flat row-major int16) | count × 14 × 2 B = ~80 MB | the data |
| labels (flat uint8) | count × 1 B = ~3 MB | per-vector fraud bit |
| per-partition grid metadata (×32): | | |
| · numCells | 4 B | cells with ≥1 vector |
| · cellStarts[numCells], cellCounts[numCells] | 8 × numCells B | locate each cell within partition |
| · bboxMin / bboxMax (each: numCells × 14 × int16) | 56 × numCells B per partition | AABB for LB |
| · binBoundaries (29 × int16 = 58 B) | 58 B | percentile thresholds for the 3 grid dims |

Total: **~95 MB** on disk = **~95 MB** in process RSS after `LoadIndex`. Fits comfortably inside the 167 MB cgroup with the Go runtime overhead (~15 MB).

The vectors are laid out in **cell order within partition, partition order within file**. So the vectors of cell `c` in partition `p` are at `vectors[(partitionStarts[p] + cellStarts[c]) * 14 + d]` for dim `d` of vector index `(0 .. cellCounts[c]-1)`.

## 6. The build pipeline, end to end

Once at Docker build (`cmd/build-index/main.go`):

1. Stream `references.json.gz`, quantize floats → int16, fill `vectors[]` and `labels[]` in JSON order.
2. **Partition pass**: compute the 5-bit key per vector, count per partition, prefix-sum into starts. Permute (cycle-sort) so partitions are contiguous in `vectors`/`labels`.
3. **Grid pass, per partition**: compute the bin boundaries for dims 0/12/7 via percentile of the partition's actual values. Assign each vector a cell key (16 × 8 × 8 = 1024 possible). Cycle-sort within partition so cells are contiguous. Walk through and compute `numCells, cellStarts, cellCounts, bboxMin, bboxMax` for non-empty cells only.
4. Write `index.bin`. Drop the intermediate JSON parse buffer.

Total build time: ~30 seconds on local dev hardware, dominated by gzip-JSON streaming, not the partition/grid passes (those are O(N) with small constants).

Runtime (`cmd/api/main.go`):

1. `mmap` or `read` `/resources/index.bin`. We use plain `read` into the heap because mmap-on-startup is no faster (the OS will fault the same pages either way under load), and a single heap allocation is easier on the Go runtime's GC model.
2. After load, `runtime.GC()` once, then `debug.SetGCPercent(200)` and `debug.SetMemoryLimit(140 << 20)`. The dataset never becomes garbage; the hot path allocates zero.
3. Start the raw HTTP server on UDS. Mark `/ready` healthy.

## 7. The query path, end to end

For a single `POST /fraud-score`:

1. **Raw HTTP read** — pooled 4 KB read buffer in `internal/api/rawhttp.go`. Body parsed out of the buffer with no copy.
2. **Parse + quantize** — `vector.VectorizeFast(body, &query)` writes 14 int16s into a stack-allocated buffer. No allocations.
3. **Partition key** — 5 ifs on 5 query dims. Result is a uint8 in [0, 31].
4. **LB sweep** — iterate cells in the partition. For each cell, sum `clamp(q[d] - bbox[d])²` over the variable dims of the partition (skip constants 9/10/11 always; skip 5 if partition is sentinel-5; skip 6 if sentinel-6). Result is `cellLB[ci]`, written into a per-handler scratch buffer.
5. **Sort cells** by LB — insertion sort for small (~600 elements) is fastest. Output is a `[]uint16` of cell indices.
6. **Cell scan loop**, in LB order:
    - If `cellLB[ci] >= topD4`: break (the magic line; this is what gives us the sub-1 ms).
    - For each vector in the cell, run the early-exit kernel:
      - Skip constant dims.
      - For each variable dim, add the squared diff. If running total ≥ `topD4`, `continue`.
      - If we reach the last dim and `dist < topD4`: cascade the new value into the top-5 slots.
7. **Fraud tally** — sum the 5 stored labels.
8. **Response write** — `w.Write(rawhttpResponses[fraudCount])`. Pre-built status line + headers + body. Zero allocation.

## 8. The trade-offs we are accepting

- **No SIMD.** Our absolute speed ceiling drops vs the asm kernel, but the algorithm dominates by a wider margin. If we ever want to push under 500 µs we revisit (likely after the leaderboard freezes).
- **Stride 14, not 16.** We lose the ability to round-trip through `[16]int16` arrays for the asm kernel (since the kernel is gone), but save 13% memory bandwidth on every vector read. The early-exit kernel's effective stride is even smaller (most candidates only read 2–3 dims).
- **Index file is rebuilt at Docker build, never mmap'd from `/resources` at runtime.** The single-replica-mmap-shared-page-cache trick from lecture 03 is theoretical anyway — under cgroup v2 accounting, the second replica's faults still count against its memory budget, so we don't lose much by giving each replica its own heap copy.
- **Tests need to be updated.** The `blockScan8` and `sqdistAVX2` tests are dead — they test code we're deleting. The `TestIVFMatchesBruteForce` is renamed to `TestGridMatchesBruteForce`. The handler bench fixes the `ds.Partition()` / `ds.BuildGrid()` calls that are currently dangling references to non-existent methods (a clear signal the test was written for an earlier code state that no longer exists).

## 9. What this lecture *isn't* solving

- The 1 FN from quantization (entry 5472). Fixing it needs float32 storage. We'd need to drop 10 MB of overhead from elsewhere to fit. Out of scope here.
- Latency on the Mac Mini specifically. Our local Rosetta measurement is at best a ratio indicator; the absolute number on Haswell will be 3–10× higher. We optimize for the algorithm being right; the Mac Mini will tell us the real number.
- HAProxy / cgroup tuning. The compose-level settings (`pull_policy: always`, `ulimits`, `seccomp:unconfined`, 0.10 / 0.45 / 0.45 CPU split) are inherited from the IVF era and don't need to change.

## 10. The shape of the next debugging session

If the rebuild brings p99 back down but detection errors go up, look at:
- The 32 partition counts at index-build time — log them. A partition with <100 vectors is a smell. If we see one, the fallback is "scan the partition + its two nearest neighbors" — not implemented now, but easy to add.
- The cell counts within a small partition. If many cells have 1–2 vectors, the LB sweep wastes time and the early-exit kernel never warms its branch predictor. Lower the bin count (16/8/8 → 8/4/4) for partitions below some size threshold.
- The bin boundaries themselves. If percentile boundaries collapse (many bins with identical boundary value) the cell assignment becomes degenerate. Log unique boundary count per partition.

If p99 stays high (>5 ms locally), profile with `go test -cpuprofile`. Likely culprits in order: (1) sort algorithm — try `sort.Slice` vs hand-rolled insertion sort; (2) LB sweep — eliminate the branch on `q < bboxMin / q > bboxMax` with branchless clamps; (3) Go's escape analysis — make sure the cell-scan local variables don't spill to heap.

## 11. The single sentence that summarizes this lecture

> **The grid algorithm is exact, the IVF is not, and the rinha scoring formula punishes approximation harder than it rewards extra speed — so we go back to the grid, with the IVF era's only durable wins (zero-alloc HTTP, pre-built index) carried forward.**
