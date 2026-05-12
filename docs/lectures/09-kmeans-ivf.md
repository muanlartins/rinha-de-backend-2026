# Lecture 09 — k-means IVF: Why Data-Aware Cells Crush Axis-Aligned Grids

The single most important architectural change from phase 13 to phase 14 is replacing the **grid** with **k-means IVF**. All 10 top submissions use IVF. None use a grid. This lecture explains why, in detail, before we touch any code.

## The framing: where does a query's time go?

For exact KNN-5 over 3 million reference vectors, the *unavoidable* work per query is whatever it takes to prove no candidate outside your "kept" set could be in the true top-5. Everything else is wasted CPU.

A good index gives you a small "kept" set of candidates that *provably* contains the top-5. The smaller the kept set, the less work. The strength of an index is measured by **how few candidates it asks you to score** to be sure of recall.

For a grid (what we have):

- Partitioning is **axis-aligned** along fixed feature boundaries. Two vectors that are extremely close in Euclidean space but happen to straddle a bin boundary on dim 0 (amount) end up in different cells.
- Cells must be **conservatively scanned** when their bounding box can't be ruled out by the AABB lower bound. Because many bins are sparse, this rules-in lots of half-empty cells.
- A single query in our phase-13 implementation scans **~3000-6000 candidates** on the hot path. Most of those scans terminate early after dim 1 or 2 — but we still pay the cache-line load.

For k-means IVF:

- Partitioning **follows the data**. Cluster centroids drift toward dense regions. A vector and its nearest neighbor almost always land in the same cluster (the empirical figure across the top 10 is ~95%).
- Each cluster gets a centroid stored as f32 (used for picking nearest clusters) **and** a per-cluster bounding box (used for pruning).
- A single query scores **all K centroid distances** (cheap: 4096 dist comps over 14 dims = ~57k mul-add ops, vectorizable) and **scans only ~`nprobe` clusters** of average size 700 vectors. With `nprobe=8`, that's ~5600 candidates scored — comparable to grid in candidate *count*, but the candidates are radically more relevant.
- Crucially, with the AABB lower bound + early-exit per block, most of those ~5600 don't get fully scored either.

The net effect on the contest hardware (Haswell, 1 CPU, 350 MB):

| Metric | Grid (us) | IVF (top 10) |
|---|---:|---:|
| Candidates scanned | 3000-6000 | 5000-8000 |
| % scanned to dim ≥ 8 | ~10% | ~5% (centroids already gated relevance) |
| % full 14-dim distances | <1% | <0.5% |
| p99 measured (ms) | 99.25 | 1.04-1.40 |

The candidate count is *similar* — but IVF's candidates are concentrated around the query, so the early-exit kernel fires more often and the cache footprint is smaller. The other big swing is the SIMD layout (lecture 10), which the grid never benefited from. We attempt both at once.

## How k-means works (Lloyd's algorithm)

The algorithm is older than its applications. Coined by Stuart Lloyd in 1957 for pulse-code modulation. The version we'll implement:

1. **Init**: pick K initial centroids `c_1 .. c_K` from the dataset (more on init below).
2. **Assign**: for each of the N reference vectors `v_i`, find the centroid `c_{a(i)}` closest in Euclidean distance.
3. **Update**: for each cluster, recompute its centroid as the mean of all vectors assigned to it.
4. **Repeat** assign + update until convergence (or fixed iteration count).

It converges to a *local* minimum of the within-cluster sum-of-squares — not the global minimum. For exact-KNN downstream we don't care; any reasonable local minimum is fine. The contest hardware permits ~6-10 iterations comfortably; the top submissions converge on 6.

**The objective function** being minimized:

```
J = sum over all i:  ||v_i - c_{a(i)}||^2
```

Iterating the two steps strictly decreases `J` (or holds it constant, in which case we've converged). This is provable: the assign step reduces `J` by definition (each `v_i` picks its closest centroid); the update step reduces `J` because the mean minimizes squared deviations to a set of points.

## Initialization: k-means++

Random init is terrible for k-means: most random K-tuples have at least one pair of nearby centroids, and the local minima reached from such inits leave a few mega-clusters and many tiny ones. **k-means++** is the standard fix:

1. Pick `c_1` uniformly at random from the data.
2. For `k = 2..K`: pick `c_k` from the data with probability proportional to `D(x)^2`, where `D(x)` is `x`'s distance to its *currently nearest* centroid.

This biases each next centroid toward the unexplored regions of feature space. The result: well-separated initial centroids, much faster convergence, and the bonus of a provable bound on `J` (Arthur and Vassilvitskii 2007: k-means++ inits within an `O(log K)` factor of optimal).

We implement it sample-trained: pick a random 65 536-row sample from the 3M references, run k-means++ + Lloyd's on the sample. Why a sample? Because Lloyd's is `O(N * K * D)` per iteration; for N=3M, K=4096, D=14 that's ~172 billion ops per iteration, dominated by the assign step. On a sample of 65k it's ~3.8 billion — runs in seconds. After convergence on the sample, we do a single full-data assign (`O(N * K * D)` once) to bucket all 3M vectors.

Implementation detail: the choice of N for the sample is a tradeoff. Too small (< 4K) and rare clusters get missed entirely. Too large (> 200K) and Lloyd's becomes the bottleneck. Empirically `5-20 * K` works well; we use ~16 * K = 65 536.

## How big should K be?

`K` = number of clusters. The top 10 picked values between 256 and 4096:

| Repo | K | Avg cluster size (over 3M refs) | Notes |
|---|---:|---:|---|
| pedrosakuma (#8) | 256 | 11 720 | Smallest K. Rerank with Q16 closes the gap. |
| whereisanzi (#10) | 512 | 5 859 | |
| luanlouzada (#5) | 1 280 | 2 343 | |
| lemesdaniel (#7) | 2 048 | 1 465 | |
| jairoblatt (#2), rafaelcoelhox (#3), steixeira93 (#4, #6), joycegodinho (#9) | 4 096 | 732 | The mode. |

K=4096 is the consensus. The tradeoff:

- **Small K** → big clusters → more candidates per probed cluster → AABB-LB pruning has to work harder → more time per query.
- **Large K** → tiny clusters → many of them get pruned cheaply → but the centroid-distance pass itself becomes the bottleneck (need to score all K centroids per query).

At K=4096, the centroid pass costs ~57 thousand ops per query (`K * 14` widened FMAs over 8-wide SIMD = ~7 200 vector ops ≈ 2 µs on Haswell). The cluster pass costs ~5 600 candidates × 14 dims = ~78 thousand ops = ~10 µs. Total: ~12 µs of compute per query, plus ~3-5 µs of memory traffic. Matches the observed p99 of ~1 ms once the HTTP overhead is layered on.

We pick **K=4096**.

## What we store per cluster

Three things, plus the per-vector payload:

1. **Centroid**: 14 × f32 = 56 B. Used to compute centroid distances for nprobe selection. Stored **SoA across all clusters**: `centroids[d*K + c]` — so one f32 ymm broadcast of the query's dim-d, then 8 lane-loads of `centroids[d*K + c..c+7]`, then one VFMADD231PS, gives 8 centroids' dim-d contribution to the running squared distance in two instructions.

2. **Bounding box**: 14 × int16 min + 14 × int16 max = 56 B. Used by the AABB lower-bound check. We store as `bboxMin[c][d]` and `bboxMax[c][d]` row-major in cluster order (cluster-major), with the dim-14/15 lanes zero-padded to 16, so one 32-byte `VMOVDQU` per cluster pulls the whole bbox into registers (lecture 11 covers the LB formula).

3. **Block stream**: 8 vectors × 14 dims = 112 int16 values = 224 B per **block**, written **dim-major within the block**: `block[d*8 + lane]`. The 14×8 layout is the secret sauce — see lecture 10.

Plus per-vector:

4. **Label** (1 byte: fraud or not). Stored as a flat `[]u8` indexed by global reference id. We dereference labels *after* the top-5 are computed, so labels don't sit on the SIMD-hot path.

The cluster offsets `[K+1]u32` tell us where each cluster's blocks start in the block stream. The last cluster's block can be partial; we pad it with `INT16_MAX` sentinels in unused lanes so phantom lanes never beat any real candidate.

## Why bounding boxes (not just centroids)?

A common mistake: assume that if a query lies outside a cluster's "Voronoi cell" (i.e., the centroid isn't in the top-nprobe), then no neighbor in that cluster could be in the top-5.

**That's false.** A cluster's centroid can be far from the query while some of the cluster's vectors are close. Voronoi-based pruning is *not* sound for exact KNN. It's what makes vanilla IVF *approximate*.

The bounding box fixes this. The cluster's AABB gives us a **provable lower bound** on the smallest possible squared distance from the query to any vector in the cluster:

```
lb(q, cluster) = sum_d  max(0, q_d - bmax_d)^2 + max(0, bmin_d - q_d)^2
```

If `lb(q, cluster) >= worst_top5_distance`, no vector in that cluster can be in the top-5 (since the smallest possible distance is already too big). Skip the cluster entirely. This is the *only* way to make IVF exact, and it's what every top submission uses.

The grid framework also computes per-cell bounding boxes, but it does so on cells defined by axis-aligned bin boundaries that don't track the data. K-means clusters' AABBs are much tighter because the cluster's vectors are tightly clustered (by construction), so most clusters' AABBs are far from any given query and get pruned cheaply.

## Sentinel handling

Two dimensions can be missing in the data:
- `minutes_since_last_tx` (dim 5)
- `km_from_current` (dim 6)

When the customer has no prior transaction, both are missing. We quantize "missing" to `-32000` (the `SentinelInt`) and treat it as an extreme out-of-range value. In `bbox_min[d]` and `bbox_max[d]` for such clusters, the sentinel value participates exactly like any other value — if a cluster contains a mix of "had-prior-tx" and "no-prior-tx" vectors, its dim-5 bbox spans `[-32000, +30000]` and the AABB-LB contribution from dim 5 is zero (because the query's dim-5 will fall inside that span no matter what).

This is fine. The only thing to watch out for in k-means training: a centroid for a sentinel-containing cluster will end up at some weird value like -10000 (the average of -32000 and +30000). That's also fine — the centroid distance is just a cluster picker, not the final scorer.

## On-disk format `IVF8` v5

We commit to a single binary file produced at Docker build time. The layout:

```
+--------------------------------------+
| Header (32 bytes):                   |
|   magic   = "IVF8"  (4 bytes)        |
|   version = 5       (u32 LE)         |
|   N       (u32 LE)  total vectors    |
|   K       (u32 LE)  clusters         |
|   blocks  (u32 LE)  total blocks     |
|   scale   (f32 LE)  quantization scale (32 000) |
|   reserved (8 bytes)                 |
+--------------------------------------+
| Centroids: [14][K] f32 (SoA)         |
+--------------------------------------+
| Cluster offsets: [K+1] u32           |
|   block_start[c] = offsets[c]        |
|   block_count[c] = offsets[c+1] - offsets[c] |
+--------------------------------------+
| BboxMin: [K][16] int16 (cluster-major, 16-wide padded with zeros) |
+--------------------------------------+
| BboxMax: [K][16] int16 (cluster-major, 16-wide padded with zeros) |
+--------------------------------------+
| Labels: [N] u8 (fraud bit per ref, original order) |
+--------------------------------------+
| Vector IDs: [N] u32 (original ref id per block-lane; needed only for label lookup, see below) |
+--------------------------------------+
| Blocks: [blocks][14][8] int16 (dim-major within block) |
+--------------------------------------+
```

Two design choices worth noting:

- **Labels stored in original ref order, not cluster order.** When the kernel finds the top-5 nearest candidates, each candidate is `(blockIndex, laneIndex)`. We map back to a global ref id via the `vectorIDs` table, then fetch the label. Storing labels in cluster order would save the indirection but require renumbering on load. The original-order layout makes the index trivially regeneratable: the offline builder doesn't need to know what the test set looks like.

  Actually, we should reconsider. The top submissions store the *fraud bit per block-lane* directly, so the top-5 candidates produce labels immediately. We'll do that: store `Labels: [blocks*8] u8` in block-lane order, padded with zero in phantom lanes. Saves the indirection.

- **Bbox padded to 16-wide.** The natural dim count is 14, but Haswell's AVX2 register width is 16 × int16 = 256 bits. Padding lanes 14-15 with zeros lets one `VMOVDQU` per cluster load the whole bbox vector. The zero pad doesn't contribute to the LB (`max(0, q-bmax) = 0` because `q=0` against `bmax=0`).

## Determinism

K-means converges to *a* local minimum, not *the* local minimum. The same input data + different RNG seed → different centroids → different bbox extents → different score (~100 points of variance, the steixeira93 author documented this).

Mitigation:
- Fix the RNG seed: `math/rand/v2.NewPCG(0xCAFE, 0xBABE)`.
- Bake the resulting `index.bin` into the Docker image at build time.
- Confirm that consecutive `docker build`s produce byte-identical `index.bin`.

If we ever swap to QEMU/Rosetta builders, this gets shakier (FP rounding order can shift centroid boundaries). The cure: pin the index across builds (`COPY --from=<previous-tag> /index /index`), which is the steixeira93 trick. We'll do this in phase 19.

## Code layout (forward reference)

The implementation will live in:

- `internal/ivf/kmeans.go` — Lloyd's + k-means++ + sample-train + full-data assign.
- `internal/ivf/index.go` — `IVFIndex` struct, serialize, deserialize, getters used by the search path.
- `internal/ivf/bbox.go` — AABB computation per cluster.
- `cmd/build-index/main.go` — the offline driver: read references → train k-means → emit `index.bin`.

And tests:

- `internal/ivf/kmeans_test.go` — `TestKMeansConverges`, `TestKMeansDeterministic`.
- `internal/ivf/index_test.go` — `TestIndexRoundTrip`, `TestIndexExactRecallVsBrute` (on 1000 sampled queries, IVF with `nprobe=K` matches brute force).

## What we are *not* doing in this phase

Phase 14 stops before search. We build the index, validate the data structures, and stop. The grid path still runs on requests. The next two phases (15: SIMD kernel; 16: search) plug the new index into the hot path.

This isolation is deliberate: it lets us test the k-means output in a controlled way before the rest of the system depends on it. We'll dump the centroids and a few cluster bboxes by hand and sanity-check them against the data distribution.

## References

- Lloyd, *Least Squares Quantization in PCM*, 1957 (published 1982).
- Arthur and Vassilvitskii, *k-means++: The Advantages of Careful Seeding*, SODA 2007.
- Faiss documentation, `IndexIVFFlat` — the C++/Python reference implementation of this exact algorithm: https://github.com/facebookresearch/faiss
- Lecture 02 of this repo for the grid we're replacing.
