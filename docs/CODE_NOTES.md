# Code Notes

Design and implementation context for the Go submission. The source files are kept comment-light on purpose; everything that would otherwise have lived in a paragraph-long docstring is here.

For the higher-level problem statement and algorithm walkthroughs, see `docs/lectures/`.

## Quantization

Real-valued dims arrive in `[0, 1]` (or `-1` as a sentinel meaning "no previous transaction"). Both the reference dataset and the per-request query go through the same int16 quantization:

- `[0, 1]` → `[0, 32000]`, half-up rounding via `int16(v*QuantScale + 0.5)`.
- `-1` (sentinel) → `-32000`.

Within a single partition, the maximum per-dim squared difference is `32000² ≈ 1e9`. Summed across 14 dims, the squared distance fits comfortably in `int64`, so the inner loop accumulates with no overflow risk.

The intermediate `float64 → float32` conversion in `quantClamp01` is deliberate. The reference dataset's quantization step runs through float32; staying in float64 in the query path produces a 1-unit divergence on values near a quantization boundary. We mirror the float32 step exactly so the same input always quantizes to the same int16 — important because the LB-pruning code compares quantized distances and a 1-unit shift can flip which cell is "closest."

## Partition key

Each vector belongs to one of 32 partitions, identified by a 5-bit key:

| bit | dim                          | meaning when set |
|-----|------------------------------|------------------|
| 0   | `is_online`                  | online           |
| 1   | `card_present`               | card present     |
| 2   | `unknown_merchant`           | merchant not in known list |
| 3   | `last_tx.minutes` sentinel   | no previous tx (minutes dim is sentinel) |
| 4   | `last_tx.km` sentinel        | no previous tx (km dim is sentinel) |

Two vectors with different partition keys disagree on at least one of these dims, each disagreement contributing at least `QuantScale²` (or `(2 · QuantScale)²` for the sentinel bits — sentinel value is `-32000`, real values live in `[0, 32000]`, so the gap is `≥ 32000`) to the squared distance. That swamps any within-partition distance, so the true KNN-5 always lives inside the query's own partition. The grid index then only ever needs to scan one partition.

## Grid V2 — cell index inside a partition

Each non-empty partition gets a 3-axis grid built on percentile boundaries:

- dim 0 (`amount`): 16 bins
- dim 12 (`mcc_risk`): 8 bins
- dim 6 (`km_from_last_tx`) for non-sentinel partitions, or dim 7 (`km_from_home`) for sentinel partitions (where dim 6 is pinned to `-32000`): 8 bins

Boundaries are at percentiles of the partition's values along that dim, not on a uniform grid — important because the `amount` dim is heavily right-skewed.

Cell sizes vary; the worst partition lands around 1024 cells. Per cell we keep an axis-aligned bounding box over all 14 dims (not just the grid dims) so the LB sweep can prune cells purely from bbox math:

```
LB(query, cell) = Σᵈ axis_lb(query[d], cell.min[d], cell.max[d])

where axis_lb(q, lo, hi) =
    (lo - q)² if q < lo
    (q - hi)² if q > hi
    0         otherwise
```

A cell can only contain a top-5 candidate if `LB ≥ topD5`. We sort cells by `LB` ascending and break out of the sweep once `LB ≥ topD5²`. Dims that are partition-constant (9, 10, 11, and 5/6 when sentinel) are skipped from the kernel entirely.

## Cycle-sort permutation

Both `Partition()` and `BuildGrid()` reorder the dataset in place via cycle decomposition. The direction is non-obvious and easy to get wrong: the routine walks the **inverse / source** map (`src[p] = original index of the vector that should end up at position p`), not the forward dest map.

If you mix them up the result looks "almost right" — the count of vectors per partition is still correct (a multiset preservation), but individual vectors are placed at the wrong positions. The verification step in `cmd/load-test` catches this by recomputing the partition key for every vector after the sort and asserting it matches the partition it ended up in.

## Hot-path zero-allocation

The HTTP handler is engineered so a steady-state request allocates nothing on the Go heap:

1. **Body read** — `bodyBufPool` is a `sync.Pool` of `[]byte` slices with 2 KB cap. Spec bodies are 500–700 bytes; we read into the pooled slice with a hand-rolled `readBodyInto` (not `io.ReadAll`, which grows the slice and allocates).
2. **Parse** — `VectorizeFast` writes into a stack-allocated `[14]int16` (`out *[14]int16`). It uses `bytes.Index` (SIMD-accelerated on amd64) to find fixed key sequences, then parses numbers and dates by reading bytes at known offsets. No reflection, no map lookups in the hot path (`mccRiskTable` is a fixed `[10000]int16`), no `time.Parse`.
3. **Search** — `FraudCountGrid` uses a `sync.Pool` (`scratchPool`) of pre-sized scratch buffers (`cellLBs`, `sortedCells`) sized for the worst-case partition. Steady state: zero allocs.
4. **Response** — pre-built `fraudResponses[0..5]` byte slices keyed by fraud count. No JSON encoding at request time.

Per-request micro-bench on Apple M4 Pro (production amd64 build): **429 ns/op, 0 allocs/op**.

## Why `GOMAXPROCS(1)` and the `GOGC`/`GOMEMLIMIT` pair

Each API container has a cgroup limit of 0.45 CPU and 167 MB. The Go runtime reads `NumCPU` and sees the host's full CPU count, not the cgroup share — without intervention it spins up 8+ schedulers and oversubscribes. `GOMAXPROCS(1)` aligns the runtime with the actual share (we have two replicas + an LB, total = 1.00 CPU per the spec).

For memory: the resident dataset is ~100 MB after partition + grid build. Default GOGC=100 doubles the heap before triggering, which would land at ~200 MB — past the 167 MB cgroup, OOM-kill. So:

- `debug.SetGCPercent(200)` — let the heap grow 3× before a GC, since per-request garbage is essentially zero.
- `debug.SetMemoryLimit(140 << 20)` — cap the heap at 140 MB. If anything ever does leak, the GC starts running early instead of letting the OOM killer take over.

Together: throughput-friendly under normal load, OOM-safe under any anomaly.

## Why a hand-rolled `references.json.gz` parser

The reference dataset is a 3M-entry JSON array (~70 MB uncompressed). Stdlib `encoding/json` allocates ~300 MB of transient garbage parsing it — that's a 3× heap overshoot under the 167 MB cgroup, OOM before we serve a request.

`dataset.scan` streams gzip+bufio and decodes by hand: scan for `"vector":[`, parse 14 floats, scan for `,"label":"`, parse `fraud`/`legit`. Zero per-entry allocation; we fill the `Vectors` and `Labels` slabs directly. Two passes (count, then fill) so the slabs are sized exactly.

## Dataset distribution — the gotcha

The rinha test infra does **not** mount `references.json.gz` into the container. Submissions are expected to bake it in at build time (the FAQ: *"Pre-process the dataset during the container build"*). Our `deploy/Dockerfile` does this with `COPY references/rinha-official/resources/references.json.gz /resources/`. The file is ~48 MB compressed, adding ~48 MB to the image size — still small overall.

Local benches must mirror this: do **not** mount `/resources` as a volume override (that would hide the in-image copy and pass on Rosetta while the rinha env fails). The local override now only bumps HAProxy memory (Rosetta JIT overhead).

## 16-lane vector layout

Each vector is stored as 16 int16s (`Stride = 16`) instead of 14. The two trailing lanes are always zero in both reference vectors and queries. Padding to 16 makes the AVX2 / SSE4.1 kernel layout-clean — `PSUBW` over 8 int16s × 2 chunks, then `PMADDWL` on the diffs to get squared pairs, no masking. Real dims (0-13) are unaffected.

## AVX2 / SSE4.1 squared-distance kernel

`internal/search/sqdist_amd64.s` computes the 16-lane squared Euclidean distance via:

1. `PSUBW` — 8 int16 diffs per chunk, 2 chunks total.
2. `PMADDWL` — pairs each chunk's diffs into 4 int32 squared-sums.
3. `PMOVSXDQ` — sign-extend each chunk's 4 int32s to two pairs of int64s before summing. This step is load-bearing: without it, two int32 chunks summed in int32 (`PADDD`) can overflow on extreme inputs (peak per-int32 sum reaches ~4.1e9, beyond int32 max).
4. `PEXTRQ` + scalar `ADDQ` — final reduction to one int64.

Selected at request entry via `cpu.X86.HasAVX2` (set once at init). Non-amd64 builds (`//go:build !amd64`) compile out the asm entirely and force the scalar fallback. Mac Mini Late 2014 (Haswell) supports AVX2/SSE4.1 natively.

### Why per-dim early-exit was traded away

The scalar kernel bails out of the inner squared-distance loop as soon as the accumulated distance exceeds the current 5th-best. SIMD computes all 14 (+2 padding) dims at once — no way to bail mid-vector. The trade is worth it: even with no early-exit, the SIMD instruction does the equivalent of 8 int16 subtracts + 8 squarings + 4 int32 adds in ~3 instruction slots, several × faster than the scalar inner loop's best case.

## Submission-branch shape

The `submission` branch contains only what the Rinha Engine runs:

- `docker-compose.yml` — references `muanlartins/rinha-de-backend-2026:latest` on Docker Hub (must be public + linux/amd64).
- `haproxy.cfg` — round-robin LB across the two Go replicas via Unix Domain Sockets in a shared tmpfs volume.
- `info.json` — participant metadata.

No source. The `main` branch is the source-of-truth for everything in `cmd/`, `internal/`, `docs/`, `deploy/`, `benchmarks/`, `references/`.
