# Lecture 02 — Algorithm & Data Structures

> Goal: understand *exactly* how the reference state-of-the-art finds 5 nearest neighbors in a few hundred microseconds, and what that translates to in Go.

## 1. The KNN-5 problem, distilled

Given:
- a fixed reference set `R` of 3M vectors in `[0,1]¹⁴ ∪ {-1 on dims 5,6}`,
- per-vector labels `L: R → {legit, fraud}`,
- a query vector `q` of the same shape,

compute:

```
NN_5(q) = the 5 vectors r ∈ R minimizing ||q − r||²        (Euclidean²; sqrt unnecessary)
score   = (count of fraud among NN_5(q)) / 5
```

Everything else — partitioning, grids, pruning, quantization — is engineering to make this fast. The output must match brute-force KNN-5 on real-valued vectors exactly, because every detection error costs ≥ 180 points and zero-error is required for top placement.

## 2. Exact vs approximate — why exact wins here

| approach | recall | typical query cost (3M, 14D, 1 CPU) | risk |
|---|---|---|---|
| Brute-force `O(N·D)` | 100% | ~6 ms (TS JIT'd TypedArray) / ~3 ms (Go SIMD-asm) | OK accuracy, but too slow |
| KD-tree / Ball-tree (exact) | 100% | degrades with D; ~1–3 ms at D=14 | branchy, hard to vectorize |
| Grid + LB pruning (exact) | 100% | 0.3–1.0 ms (reference QRust hits 650 μs) | tunable, vectorizable inner loop |
| HNSW (approximate) | 95–99% | 0.05–0.3 ms | **kills the score** — even 1% recall miss → 30+ FN |
| IVF / LSH (approximate) | 90–98% | even faster | same problem |

The competition rewards 100% recall and rewards lower latency *only if* recall is perfect. So we use an **exact** method built on three independent search-space reductions, all of which preserve correctness:

1. **Partition by discrete-valued dimensions** (5 bits → 32 partitions).
2. **Spatial grid with axis-aligned bounding boxes** inside each partition.
3. **Lower-bound pruning + early-exit distance** on the candidates.

Each layer cuts the candidate set further without ever losing the true top-5.

## 3. Layer A — partition by the discrete dims (the "free" 30× cut)

Five of the 14 dimensions are effectively binary, in the sense that the query and reference are either *identical* on that dimension or *very far apart* (squared contribution ≥ 1):

- bits 9, 10, 11: `is_online`, `card_present`, `unknown_merchant` ∈ `{0, 1}`
- "bit" 5: `is_sentinel_5` ∈ `{0, 1}` (does dim 5 equal `-1`?)
- "bit" 6: `is_sentinel_6` ∈ `{0, 1}` (does dim 6 equal `-1`?)

Build a 5-bit **partition key** from those bits. Group all references by key:

```
                                 partitionStarts[k]    partitionCounts[k]
                                       │                     │
   refs ─sort by key─►   [ key=0...    │     key=1...        │      ...     ]
                          ▲           ▲                     ▲
                          │           │                     │
                          contiguous block per key
```

At query time, compute the same key from the query, and **search only inside that one block**. The pigeonhole counts are uneven (e.g., `is_online=true, card_present=true` is common; `card_present=false` with no sentinels less so), but partitions land in the 30k–500k range — a 6×–100× reduction off 3M for free.

**Why it preserves correctness.** Any reference in a different partition has at least one bit difference from the query. That bit contributes ≥ 1 to the squared distance. Inside the query's partition we will *almost certainly* find a reference closer than 1, so no out-of-partition vector can possibly be in the top-5.

> One exception: the only case where this could fail is if the query's partition is extremely sparse and we get neighbors with squared distance > 1. We can guard at startup: if any partition is below some threshold (say, 200 vectors), we fall back to scanning the union of that partition with the next-closest one. In practice, with a 3M-vector dataset and 32 partitions, no partition is that small.

**Go layout** for this layer:

```go
type Dataset struct {
    Vectors          []int16   // length = N * 14, flat, row-major
    Labels           []uint8   // length = N (1 = fraud, 0 = legit)
    PartitionStarts  [32]uint32
    PartitionCounts  [32]uint32
}
```

A single contiguous `[]int16` for all vectors is preferable to `[][]int16` (one slice per partition): same total bytes, but **one** allocation, one indirection per access, and cache-friendly scans. The partition metadata (`Starts`, `Counts`) is a tiny fixed array, kept as values not pointers.

## 4. Layer B — the spatial grid V2

Inside one partition (say, the 200k vectors with `is_online=true, card_present=true, unknown_merchant=false, no sentinels`), the 9–11 "variable" dimensions are still continuous in `[0,1]`. We need to find 5 nearest in 200k. Brute-force is fine — `200_000 × ~14 ops ≈ 3M ops`, ~3 ms in Go — but we can do much better with a coarse spatial grid:

### 4a. Select grid dimensions

Pick **3 dimensions with high variance** (in QRust: `amount` (dim 0), `mcc_risk` (dim 12), `km_from_current` (dim 6 — but only in non-sentinel partitions; for sentinel partitions, substitute another like `km_from_home` (dim 7))). Bin each:

- dim 0 (amount): **16 bins** at percentiles 1/16, 2/16, ..., 15/16 of the partition's amount distribution.
- dim 12 (mcc_risk): **8 bins** at percentiles 1/8, ..., 7/8.
- dim 6 or 7 (distance-ish): **8 bins**.

Total cells per partition: up to 16 × 8 × 8 = **1024**. Often less because many cells end up empty (correlations exist).

> The bin boundaries are *percentile-based*, not equally spaced. This produces approximately balanced cells. The reason: real transaction data is heavily skewed (most amounts are small, most MCCs cluster around a few high-volume categories). Equal-width bins would put 95% of vectors in 5% of cells and starve LB pruning.

### 4b. Cell key encoding

```
cell_key = bin(d0) * 8 * 8  +  bin(d12) * 8  +  bin(d6)
         = bin(d0) * 64     +  bin(d12) * 8  +  bin(d6)
```

This is a positional encoding. The query gets the same encoding. **Cells are stored contiguous** — at build time, we permute the partition's vectors so cell 0 comes first, then cell 1, etc. A `cellStarts[c]` and `cellCounts[c]` pair locates each cell's slice:

```
              cellStarts[c]   cellCounts[c]
                    │              │
   partition ─►  [ cell 0    │    cell 1    │    ...    │   cell 1023 ]
                   ▲                                            ▲
                  contiguous within partition, no holes
```

### 4c. Axis-aligned bounding boxes (AABBs)

For each cell, store the min and max of each of the 14 dimensions across the cell's vectors:

```
bboxMinFlat[cell * 14 + d] = min(v[d] for v in cell)
bboxMaxFlat[cell * 14 + d] = max(v[d] for v in cell)
```

These are tight bounds on what the cell "contains" in space. They power the lower-bound pruning of Layer C.

### 4d. The in-place permutation trick

Building the grid means **reordering the partition's vectors** so that vectors in the same cell sit consecutive. Naively you'd allocate a parallel `vectorsNew` array and copy. That doubles startup memory.

Instead, decompose the permutation into cycles and rotate each cycle in place — O(N) extra memory is just an `[]uint8` visited bitmap. See the `// Apply permutation (correct cycle sort)` block in [references/qrust-luanmonteiro/src/grid-v2.ts:113](references/qrust-luanmonteiro/src/grid-v2.ts:113). In Go, a clean version:

```go
// reorder vectors so that they appear sorted by cellKey
visited := make([]bool, pCount)
buf := make([]int16, 14)
for i := 0; i < pCount; i++ {
    if visited[i] { continue }
    j := i
    for {
        visited[j] = true
        next := sortIndices[j]
        if next == i || next == j { break }
        // swap vectors[j] ↔ vectors[next]
        srcJ := vectors[(pStart+j)*14:(pStart+j)*14+14]
        srcN := vectors[(pStart+next)*14:(pStart+next)*14+14]
        copy(buf, srcJ); copy(srcJ, srcN); copy(srcN, buf)
        labels[pStart+j], labels[pStart+next] = labels[pStart+next], labels[pStart+j]
        cellKeys[j], cellKeys[next] = cellKeys[next], cellKeys[j]
        j = next
    }
}
```

## 5. Layer C — lower-bound pruning + early-exit distance

Now the search. For a query `q` in partition `P` with C cells:

### 5a. Compute the lower-bound distance to each cell

For an AABB `[min_d, max_d]` and a query coordinate `q_d`, the **minimum squared distance** that *any* point in the box can be from `q` along that axis is:

```
lb_d = 0                              if min_d ≤ q_d ≤ max_d  (q is inside the box on that axis)
     = (q_d - min_d)²                 if q_d < min_d         (closest point is on the min face)
     = (q_d - max_d)²                 if q_d > max_d         (closest point is on the max face)
```

Summed over the variable dimensions of the partition, that's `lb(cell)` — a strict lower bound on the distance from `q` to any vector inside that cell.

```
         ┌─────────┐                  q is here (above the cell)
         │  cell   │
         │ bbox    │   lb_d on Y axis = (q_y − max_y)²
         │         │   lb_d on X axis = 0 (q_x is in [min_x, max_x])
         └─────────┘
                          ●  q
```

### 5b. Visit cells in increasing LB order

Sort the cell indices by their LB. Then iterate:

```
for ci in cellsSortedByLB:
    if lb(ci) >= currentTopD5²:
        break                         # no remaining cell can beat the current 5th-NN
    scan all vectors in cell ci with the early-exit kernel
```

The moment a cell's LB is ≥ the current worst-of-top-5 squared distance, we know **no future cell** (since they're sorted) can contribute. We stop. With well-shaped grids, this typically prunes 80–95% of cells before even looking inside.

### 5c. Early-exit per-vector distance

Once we're inside a cell, the kernel walks the variable dimensions accumulating squared distance. After **each** dim:

```
dist += (q_d - v_d)²
if dist >= topD5: break               # this vector is already worse than the 5th-NN
```

Three reasons this is much faster than computing all 14 (or 11, or 9) squared diffs unconditionally:

1. Most candidate vectors in a cell are *not* in the top-5. The earlier we abandon them, the better.
2. The first few dims (often the high-variance ones picked for gridding) dominate the distance signal — they discriminate fast.
3. The branch is highly predictable: once topD5 has converged, almost all candidates exit at dim 2 or 3.

### 5d. Constant-dim skipping

Dims 9, 10, 11 are constant within a partition (it's *the partition key*). Dims 5 and 6 are constant in sentinel partitions. The kernel skips them entirely — no diff, no multiply, no add.

So the effective dim count per partition is:

| partition type | variable dims | kernel ops/vector (max) |
|---|---|---|
| Non-sentinel (16 partitions) | 11 (skip 9, 10, 11) | 11 |
| Single-sentinel (8 + 8 = 16 partitions, but in practice many empty) | 10 (skip 9, 10, 11 and one of 5/6) | 10 |
| Double-sentinel (mostly the 4 fully-sentinel keys) | 9 (skip 9, 10, 11, 5, 6) | 9 |

### 5e. Top-5 maintenance — no heap, just 5 variables

Instead of a heap (allocates, branches) or a sort-on-update (O(K log K)), maintain 5 paired variables `(topD0, topI0), ..., (topD4, topI4)` always kept in ascending distance order. On insert, a cascading 5-slot shift:

```
if dist < topD0:     shift all up, place at 0
elif dist < topD1:   shift [1..3] up, place at 1
elif dist < topD2:   shift [2..3] up, place at 2
elif dist < topD3:   shift [3] up,    place at 3
elif dist < topD4:                    place at 4
```

It's literally an unrolled insertion sort over 5 elements. In Go, write it as a single function with no slice, no map, all named locals — the compiler will keep it in registers.

## 6. Scalar quantization — the int16 width choice

We want a 14-dim distance kernel on 3M vectors to be fast. Three reasons int16 beats float32:

1. **Half the memory.** 3M × 14 × 2 B = 80 MB vs 160 MB. Critical inside a 350 MB total budget.
2. **Half the cache footprint.** A single L1d miss costs ~10 ns; quantizing halves the misses per inner loop.
3. **Integer arithmetic.** On x86, `IMUL` between int16 → int32 results is single-cycle with throughput 1; SIMD `PMADDWD` packs **8 multiply-and-add** into a single instruction (per 128-bit lane); AVX2 does 16.

### 6a. The quantization scheme

```
QUANT_SCALE = 32000
SENTINEL    = -32000

for each float v in [0, 1]:           int16  = round(v * 32000)
for sentinel -1.0:                    int16  = -32000
```

After quantization, `int16` values land in `[-32000, +32000]`. The maximum per-dim squared difference is `(32000 − (−32000))² = 4.1 × 10⁹` — already overflows `int32` (`2.1 × 10⁹`). Summed across 14 dims, the squared distance can reach `5.7 × 10¹⁰`. So **accumulate in `int64`**, not `int32`.

Within a single partition (where the sentinel bits are pinned), the max per-dim diff is `32000` (a real `[0, 32000]` value vs another), squared = `1.024 × 10⁹`. Across the 11 variable dims, max squared sum = `1.13 × 10¹⁰` — still int64 territory. Don't fall for the int32 trap.

### 6b. Does this affect detection accuracy?

The dataset itself uses 4-decimal floats. The quantization error per dim is at most `1 / 32000 ≈ 3.1 × 10⁻⁵`. Squared and summed: max contribution to distance `≈ 14 × 9.6 × 10⁻¹⁰ ≈ 1.3 × 10⁻⁸` (in original float² units, or `~14` in int16² units after the `32000²` scale). For points where the 5th-NN distance is on the order of `0.001` in float² (= `10⁶` in int16²), the quantization noise is 5 orders of magnitude smaller. **In practice, zero detection divergences vs brute-force float on the QRust sample tests.**

### 6c. Why not int8?

Save another 40 MB, but:
- Max per-dim diff = 255, squared = ~65k; max sum across 14 dims = ~900k — fits in int32 trivially. *Tempting.*
- But quantization error per dim is now `1 / 255 ≈ 4 × 10⁻³` — about **130×** worse than int16. Squared and summed: ~2 × 10⁻⁵ in float² units. Still small, but starts colliding with the 5th-NN gap in dense partitions.
- Risk: a few detection divergences (FP/FN) at the very edges. At -181 points per FN, this risk likely doesn't pay off.

If memory pressure becomes the binding constraint in the Go port, revisit int8 with paired-vector PMADDUBSW + careful unbiased rounding. Default: stick with int16.

## 7. The full algorithm, top to bottom

```
Build time (once, at container startup):

  load references.json.gz from disk
  for each reference:
    quantize 14 floats to int16
    compute partition key (5 bits)
  sort references by partition key
  for each non-empty partition:
    select 3 grid dims based on variance
    compute per-dim percentile bin boundaries
    assign each reference to a cell
    permute references so cells are contiguous (cycle sort)
    compute per-cell AABB (min/max per dim)
  write the whole layout to dataset.bin
  free the original JSON / float arrays

Query time (per request, hot path):

  parse JSON payload → 14 float values  (and one lookup: mcc → risk)
  quantize floats → int16 query[14]
  compute partition key from query[9], [10], [11], [5], [6]
  fetch partition P (start, count)
  if P.count == 0: return frauds=0

  for each cell c in P:
    compute lb(c) using AABB and query (skip constant dims)
  sort cells ascending by lb
  topD5 = +∞,  5 top-distance/top-index slots = (+∞, -1) × 5

  for each cell c in sorted order:
    if lb(c) >= topD5: break
    for each vector v in c:
      dist = 0
      for each variable dim d:
        dist += (query[d] - v[d])²
        if dist >= topD5: skip to next v
      insert (dist, idx) into top-5 (cascading 5-slot shift)
      topD5 = top-5 worst distance

  frauds = sum of labels[topI_i] for i in 0..4
  return frauds                  # 0..5
  → fraud_score = frauds / 5
  → approved   = fraud_score < 0.6
```

## 8. Performance budget per stage (reference numbers from QRust)

Measured on QRust at ~650μs total search time (the algorithm alone, no HTTP/JSON):

| stage | time | notes |
|---|---|---|
| Partition select | < 50 ns | 5 if-statements |
| AABB LB sweep across ~600 cells | ~60 μs | ~100 ns/cell, mostly L1d hits |
| Cell sort by LB (~600 cells) | ~30 μs | quicksort over Uint16 indices |
| Cell scan with early-exit | ~500 μs | dominated by ~180k vector visits average |
| Top-5 final read + label lookup | < 1 μs | 5 random accesses to labels[] |
| **total search** | **~650 μs** | not counting JSON parse / HTTP |

To beat 1 ms total p99, the JSON parse + HTTP + LB hop has to fit in the remaining ~300 μs. (See Lecture 03 for that.)

## 9. Go-specific implementation considerations

Things that change between TS/Bun and Go for this algorithm:

### Data layout

Stick to **`[]int16` flat**. Indexing is `vectors[i*14 + d]`. Go's compiler does bounds-check elimination (BCE) when the indices are obviously in bounds. To help BCE on the hot path:

```go
// Hint to the compiler that all 14 reads are in-bounds:
v := vectors[base : base+14 : base+14]
d0 := q[0] - v[0]
d1 := q[1] - v[1]
// ... no further bounds check
```

The `:base+14:base+14` triple-slice forces a single bounds check at the slice expression, and subsequent accesses to `v[0..13]` are unchecked.

### Accumulator type

```go
var dist int64 = 0
diff := int32(q[d]) - int32(v[d])      // int16 → int32 widening before subtract; subtract can't overflow
dist += int64(diff) * int64(diff)
```

The double widening looks ugly but is what gives the compiler the most freedom. On amd64 it'll fuse into `MOVSWL` + `IMUL`.

### Top-5 update

Inline-write the cascading shifts as named locals (do *not* use a `[5]int64` array — the compiler will spill it to stack). Five if/else branches, no loop:

```go
if dist < topD0 {
    topD4, topI4 = topD3, topI3
    topD3, topI3 = topD2, topI2
    // ... etc
    topD0, topI0 = dist, idx
} else if dist < topD1 {
    // ...
}
```

### Early-exit kernel — the hot loop

Write it as a function that takes `(q *[14]int16, v *[14]int16, threshold int64) int64`. The pointer-to-array form gives the compiler more guarantees than `[]int16` (length is known constant 14). Inside, an unrolled sequence of 11 or 9 diff/sqr/accumulate/branch.

Or — and this is the key escape valve — drop to **Plan 9 assembly** for `vec_dist_sq_int16(q, v, mask) int64`:

- AVX2 `VPMADDWD` can multiply-add 8 int16 pairs → 4 int32 results per ymm register in one instruction.
- We only have 11–14 dims, so one ymm vector (16 lanes) covers everything with masking.
- Horizontal sum at the end (`VPHADDD` + extract).

Go's autovectorizer is **not** going to emit this. Hand-rolled asm in `internal/vec/dist_amd64.s` is the right move once the algorithm is otherwise stable. Pure-Go fallback in `dist_amd64_fallback.go` for builds without AVX2.

### Allocation discipline

Pre-allocate at startup:
- `cellLBs []int64` of length = max cells per partition (~1024)
- `sortedCells []uint16` same length
- A per-handler `*[14]int16` query buffer (or `sync.Pool` if we want true GMP-scaling, though with 0.45 cores we'll have ~2 OS threads, so a small pool suffices)

Never allocate in the hot path. `pprof -alloc_objects` should show zero allocations for `/fraud-score` once we're done.

### GOGC and GOMEMLIMIT

After the dataset is loaded and the grid is built, do an explicit `runtime.GC()`, then set `debug.SetGCPercent(-1)` to disable the GC altogether. The dataset never becomes garbage; the only allocator activity is from any per-request slop (which we'll drive to zero). If we miss a few allocations, set `GOGC=200` instead of disabling, and `GOMEMLIMIT=160MiB` so the GC reacts to memory pressure rather than the default 100% growth heuristic.

## 10. Where this lecture ends and the next begins

We now understand:
- The three-layer search-space reduction: partition → grid → LB-pruned cell scan.
- Why it preserves correctness (exact) and why each layer is correctness-preserving.
- The data layouts: flat `[]int16` for vectors, AABBs per cell, `[]uint8` for labels.
- The quantization choice (int16, scale 32000, sentinel -32000) and its safety budget.
- The Go-flavored implementation pattern for each piece.

What we have **not** yet covered:
- The HTTP server + JSON parsing hot path (also ~300 μs of budget).
- HAProxy round-robin + UDS in tmpfs vs TCP loopback.
- Docker cgroup tuning rationale (why 0.10/0.45/0.45 for CPU; why 16/167/167 for RAM).
- The dataset build pipeline (downloading `references.json.gz`, streaming gzip, building `dataset.bin` deterministically).
- How to measure all of this rigorously and chart it.

Those are Lectures 03 and 04.
