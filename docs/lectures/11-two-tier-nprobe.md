# Lecture 11 — Two-Tier IVF: Adaptive nprobe and the Borderline-Only Escalation

This is the algorithm-level glue between phases 14 and 15. Given a k-means IVF index (phase 14) and an AVX2 inner kernel (phase 15), we have to decide *which clusters to scan* and *how to keep the answer exact*. The answer is "two tiers": a cheap fast pass that gets the right answer on the easy 95% of queries, and an expensive slow pass that catches the borderline cases.

## The framing: what makes a query "easy" vs "hard"?

The contest's decision rule is `count_frauds_in_top_5 >= 0.6 → deny`, which is the same as `count >= 3 → deny`. Treat the count as the random variable: the binary classification (approve / deny) flips between `count == 2` and `count == 3`.

- `count == 0`: 5 nearest are all legit. Approve. Very confident — even if a different 5th-nearest had been chosen, the count would still be 0 or 1, both still approve.
- `count == 5`: 5 nearest are all fraud. Deny. Very confident — symmetric.
- `count == 4`: 4 of 5 are fraud. Deny. The 5th would need to flip from legit to fraud for this to change to count=5 (still deny) or from fraud to legit (count=3, still deny).
- `count == 1`: 1 of 5 is fraud. Approve. Symmetric to count=4.
- **`count == 2` or `count == 3`: 2 or 3 of 5 are fraud.** A single nearer neighbor whose label differs could flip the decision. **These are the hard queries.**

The top 10 leaders all use this observation: escalate the search if and only if `count in {2, 3}`. Some escalate on `{1, 2, 3, 4}` to be conservative; the best (rank #6, #10) escalate only on `{2, 3}`.

We pick `{2, 3}` — strictly less work for the same correctness.

## The fast tier

For every query, do this work:

1. **Compute centroid distances**. The query vector q has 14 f32 dims. Each of the K=4096 centroids has 14 f32 dims. We compute K squared distances `||q - c_k||^2`, one per cluster. With the centroids stored SoA (`centroids[d*K + c]`), this is a tight 14-dim loop over 8-wide AVX2 chunks: ~57k FMAs, vectorizable to ~7 200 instructions, ~2-3 µs on Haswell.

2. **Pick top nprobe=8 clusters by centroid distance**. We're selecting 8 from 4096, so a partial sort is overkill; a single linear pass with insertion into an 8-element sorted small array is faster than any heap-based partial sort at this size.

3. **Scan each picked cluster** with AABB-LB pruning. For cluster c:
   - Compute `lb = aabb_lower_bound(q, bbox[c])`.
   - If `lb >= worst_top5_distance`, skip the cluster.
   - Otherwise, for each block in the cluster:
     - Call `ScanBlock8AVX2(q, block, worst_top5, sum)`.
     - For each lane reported alive: insert into top-5.

4. **Count frauds in top-5**. Each top-5 entry has a label; sum them.

5. **If count in {0, 1, 4, 5}, return the count**. We're done.

The fast tier touches ~8 × 732 = ~5 800 candidate vectors on average, of which 95% terminate inside dim 4 of the kernel because their squared distance after 4 dims already exceeds the running worst-of-top-5. Cumulative compute: ~5 800 × ~4 dim-FMAs / 8 lanes = ~2 900 vector ops ≈ ~4 µs. Plus the AABB-LB sum of 14 fast int16 max-and-square checks per cluster, ×4096-but-cluster-pruned-to-maybe-20. Negligible compared to the kernel work.

## The slow tier

Trigger when `count in {2, 3}`. We have two options, both used by top submissions:

- **Option A: expand to top-24 clusters by centroid distance** (e.g. `andrade-cpp-ivf` rank #1 with `FULL_NPROBE=24`). The next 16 clusters we hadn't yet scanned are now scanned. Same kernel, same AABB-LB filter.
- **Option B: scan ALL remaining clusters but with strict AABB-LB pruning** (`whereisanzi` rank #10). Of the ~4 088 remaining clusters, ~95% will be pruned by the LB check, leaving ~200 actually scanned.

Both are exact under the AABB-LB soundness argument (next section).

We pick **Option A**, the simpler one, with `FULL_NPROBE=24`. It's deterministically bounded in work — 16 more clusters × 732 vectors = ~12 000 more candidates, of which most will still terminate early — keeping p99 predictable. Option B has higher variance (the LB pruning rate depends on the query's distance distribution; worst-case all clusters are scanned).

The top-5 carried into the slow tier is the top-5 from the fast tier. The new candidates only need to beat the worst-of-top-5 from the fast tier — i.e., the worst gets tighter and the kernel's early-exit fires harder.

## Why AABB-LB is sound

The AABB lower bound formula:

```
lb(q, cluster_c) = sum_d  L_d
where L_d = max(0, q_d - bmax_c[d])^2     if q_d > bmax_c[d]
            max(0, bmin_c[d] - q_d)^2     if q_d < bmin_c[d]
            0                              otherwise
```

**Claim:** for any vector v in cluster c, `||q - v||^2 >= lb(q, c)`.

**Proof:** for each dim d, the contribution to `||q - v||^2` is `(q_d - v_d)^2`. Since `bmin_c[d] <= v_d <= bmax_c[d]`:
- If `q_d > bmax_c[d]`: the closest possible v_d is `bmax_c[d]`, giving `(q_d - bmax_c[d])^2 = L_d`. Any other v_d in the bbox is further away. So `(q_d - v_d)^2 >= L_d`.
- If `q_d < bmin_c[d]`: symmetric, `L_d = (bmin_c[d] - q_d)^2`.
- If `bmin_c[d] <= q_d <= bmax_c[d]`: the closest possible v_d is `q_d` itself (allowed inside the bbox), giving `L_d = 0`. So `(q_d - v_d)^2 >= 0 = L_d`.

In all cases the per-dim contribution is bounded below by `L_d`. Summing over dims: `||q - v||^2 >= sum_d L_d = lb(q, c)`. ∎

Consequence: if `lb(q, c) >= worst_top5`, no vector in cluster c can beat the worst of our current top-5. The cluster is safe to skip.

## The cascading top-5 insert

The kernel returns 8 partial sums after each block. For each lane, we may need to insert it into the top-5. Since the heap is only 5 elements, a heap is gratuitous overhead — we use a 5-slot sorted array and shift insertions.

```
type top5 struct {
    dist  [5]float32
    label [5]uint8
}
worst := top5.dist[4]  // largest among the current top-5
```

To insert a new candidate `(d, l)`:
- If `d >= worst`, do nothing.
- Else find the position `p` such that `top5.dist[p-1] <= d < top5.dist[p]` (linear scan; 5 elements).
- Shift `top5.dist[p..]` and `top5.label[p..]` right by one.
- Write `top5.dist[p] = d, top5.label[p] = l`.
- Update `worst = top5.dist[4]`.

This is ~10-15 ns per insertion in practice. With ~5-50 insertions per query (most candidates fail the early-exit before they could land in top-5), the total cost is well under 1 µs.

## The data we need to assemble

Phase 14 gave us the index format. The search function needs:

- `centroids []float32` — the 14*K SoA array.
- `offsets []uint32` — block ranges per cluster.
- `bboxMin []int16, bboxMax []int16` — bounding boxes per cluster, 16-wide padded.
- `blockData []int16` — the per-block 14×8 int16 panels.
- `labels []uint8` — per-block-lane fraud bit.

And we'll add one helper:

- `Centroid8AVX2(q *float32, centroids *float32, dimStride int) [8]float32` — vector-of-8 squared distances from q to 8 consecutive centroids in SoA layout. Returns a `[8]float32` of distances. Used to score K/8 = 512 chunks per query in the centroid pass.

Actually, since we want all K=4096 centroid distances scored in one pass and *then* a top-N selection, a simpler design: write a single `ScoreAllCentroids` function that writes the K f32 distances into a caller-provided buffer. This is roughly 8 ymm-wide loads per dim × 14 dims × 512 chunks ≈ 57 k ops, all in tight asm or Go-friendly auto-vec.

For phase 16 we'll write the centroid scorer in pure Go and let the compiler auto-vectorize. AVX2-explicit centroid scoring can be added later if it shows up in pprof.

## The struct types

The phase-16 search code introduces:

```go
type IVFSearch struct {
    idx *ivf.IVFIndex
}

// FraudCountIVF runs the two-tier IVF search and returns the fraud count
// (0..5) among the top-5 nearest reference vectors to q.
// q is the dequantized query (14 f32). qi is the quantized query (14 int16)
// — used by the AABB-LB check, which is int16 native.
func (s *IVFSearch) FraudCountIVF(q *[14]float32, qi *[14]int16) uint8

// aabbLowerBound (private): integer-arithmetic LB in q16 units.
// Returns the lower-bound squared distance as int32 (fits given bmax-bmin <= 2^16-1).
func aabbLowerBound(qi *[14]int16, bmin, bmax *[16]int16) int32
```

Both `q` (f32) and `qi` (int16) are passed because the kernel needs f32 (for the AVX2 cvt path) and the AABB-LB is cheaper in int16. The caller already has both — the parser produces int16, and we widen to f32 once per query.

## Code layout

The implementation lives in:

- `internal/search/ivf.go` — `FraudCountIVF` and helpers.
- `internal/search/bbox.go` — `aabbLowerBound`, generic implementation.
- `internal/search/centroid.go` — `ScoreAllCentroids` (pure Go, auto-vec).
- `internal/search/top5.go` — `Top5` cascading insert.
- `internal/search/ivf_test.go` — `TestIVFMatchesBrute`, `TestIVFFullDataset`.

The grid path (`internal/search/grid.go`) stays through phase 16 for A/B testing. Phase 17 deletes it.

## What we are *not* doing

- **Quantized AABB-LB in pure int32.** We could keep the LB in `int32` units (q-bmax is int16, squared is int32, sum-of-14 is int32 if we're careful about overflow at `32000^2 * 14 ≈ 1.4e10` which doesn't fit in int32). Float32 LB is safer. We can revisit if profiling demands it.
- **Refined ordering of cluster scan.** Some submissions scan clusters in order of centroid distance (so the tightest worst-of-top-5 is established earliest). The order matters slightly — but with K=8 fast clusters, the variance from ordering is small. We sort fast.

## References

- Faiss `IndexIVFFlat` source — the canonical exact-IVF implementation.
- `andrade-cpp-ivf` rank #1, file `cpp/src/ivf.cpp:837-843` — the `boundary_full` cascade trigger.
- `whereisanzi-rinha-backend-2026` rank #10, file `src/search.rs` — Option B (all-clusters with LB).
- Lecture 09 § AABB rationale, lecture 10 § the kernel signature.
