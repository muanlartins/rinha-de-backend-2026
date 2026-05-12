# Lecture 08 — Top-10 Survey: How the Winners Beat 5800

We are at rank #86 with score 3823 and p99 99.25 ms. The cluster between rank #1 and rank #10 ranges from 5983.13 to 5853.33 with p99 1.04–1.40 ms — all with FP=FN=Err=0. The gap is almost entirely on the latency axis.

This lecture surveys the top 10 deeply. The encouraging finding: **the top 10 are not 10 different ideas**, they are 10 implementations of the same idea. The unencouraging finding: that one idea is *not* what we currently do.

## The Top 10 at a Glance

| # | Submitter | Score | p99 | Lang | Repo |
|---|---|---|---|---|---|
| 1 | viniciusdsandrade | 5983.13 | 1.04 ms | C++20 | `andrade-cpp-ivf` |
| 2 | jairoblatt | 5978.38 | 1.05 ms | Rust | `rinha-2026-rust` |
| 3 | rafaelcoelhox | 5947.37 | 1.13 ms | C | `eu-sou-o-ze-pamonha` |
| 4 | steixeira93 (v2) | 5929.88 | 1.18 ms | Zig | `rinha-backend-26-v2` |
| 5 | luanlouzada | 5910.43 | 1.23 ms | C++20 | `rinha_backend2026` |
| 6 | steixeira93 (v1) | 5905.15 | 1.24 ms | **Go** | `rinha-backend-26` |
| 7 | lemesdaniel | 5883.65 | 1.31 ms | Zig | `floating-finch-zig` |
| 8 | pedrosakuma | 5859.53 | 1.38 ms | C# (NativeAOT) | `rinha-backend-2026` |
| 9 | joycegodinho | 5854.11 | 1.40 ms | **Go** | `rinha-2026` |
| 10 | whereisanzi | 5853.33 | 1.40 ms | Rust | `rinha-backend-2026` |

Two of the top 10 are Go (rank #6 and #9). They beat us by **80×** on p99 with the same compiler and the same hardware. That is the headline.

## The Consensus Stack

Across all 10 submissions, the architecture is uncannily uniform. Independent contestants in five different languages converged on the same shape. Where one would expect "10 different tricks", the reality is "one trick with 10 different fillings."

### 1. Index: k-means IVF (not grid, not HNSW)

Every single one of the top 10 uses **Inverted File Index** with **k-means clustering**. Not grid. Not HNSW. Not flat brute-force. Not LSH. Not Annoy.

Cluster count varies (`K=256, 512, 1280, 2048, 4096`), but the algorithm is identical:
- Train k-means offline at Docker build time (sklearn, custom Lloyd, k-means++ seed).
- At query time, compute distance to all `K` centroids, pick the top `nprobe` closest.
- Scan only those `nprobe` clusters.

This is the **single biggest gap** between us and the leaders. Our 32-partition + 1024-cell grid is a fixed axis-aligned tessellation of feature space — the cells follow the bin boundaries, not the data. K-means cells follow the data. The bins that matter end up small and dense; the bins that don't end up sparse and far away.

**Practical consequence:** With k-means, the top-8 centroids in the dataset cover ~95% of true-nearest neighbors. With a regular grid, you scan all 12 non-empty partitions plus their bbox-LB cells. The candidate count difference is one to two orders of magnitude.

### 2. Quantization: int16 × 10 000

Everyone, without exception, quantizes references to `int16` with scale `10 000`. This gives 4 decimal digits of precision and **is lossless** because the producer's reference vectors are already `round4`-precision. The conversion `f32 → int16` is exact for any value in `[-1, +1]` that has at most 4 decimal digits, and the entire reference set satisfies this.

We are also at int16 × 32 000. That's fine — but our scale choice (`32 000` instead of `10 000`) saves no precision and costs us nothing, it's just gratuitously different from the consensus. Either scale is exact for the dataset.

What does matter: distance is computed in **f32** (`cvtepi16_epi32 → cvtepi32_ps → fmadd`), not in int32. The reason is that even though distance accumulators in int32 would never overflow for 14 dims of int16, **fmadd produces fewer fused operations** in int land than in fp land on Haswell — `vfmadd231ps` is one cycle of throughput while integer multiply-add `vpmaddwd` is also one cycle but requires the lanes to be int16-paired in a specific way that you can only achieve by paying for the SoA layout (see below). The fp path is simpler to write and identical in throughput once the data is in registers.

### 3. Layout: 8-vector dim-major blocks

The universal physical layout for an IVF cluster is:

```
block[b * 14 * 8 + d * 8 + lane]
```

That is, **8 vectors per block, 14 dims per vector, dim-major within a block**. So the 8 lanes of dim 0 are contiguous; then 8 lanes of dim 1; etc. 14 × 8 × 2 B = 224 B per block.

Why this layout is the magic:

- One `_mm_loadu_si128` (128-bit = 8 × int16) pulls the dim-d value for 8 vectors in one instruction.
- After `_mm256_cvtepi16_epi32 + cvtdq2ps`, you have eight f32 dim-d values in one ymm register.
- Broadcast the query's dim-d into another ymm.
- `_mm256_sub_ps + _mm256_fmadd_ps`: 8 squared differences accumulate in one ymm in **two cycles of throughput**.
- 14 dims = 28 cycles per block of 8 vectors = **3.5 cycles per vector distance, scalar-equivalent**.

This is the inner loop that everybody runs. The Rust and C and C++ submissions write the AVX2 intrinsics by hand. The Zig submissions use `@Vector(8, ...)`. The Go submissions write **Plan 9 amd64 assembly** in `.s` files (both `joycegodinho` and `steixeira93` do this). NativeAOT C# uses `System.Runtime.Intrinsics.X86.Avx2` directly.

### 4. Pruning: AABB lower-bound per cluster

After picking the top `nprobe` clusters by centroid distance, the inner scan does *not* compute distance for every vector in every cluster. Instead it maintains a current worst-of-top-5 distance and uses a **per-cluster axis-aligned bounding box** as a cheap lower bound on the best distance that cluster could produce.

For each cluster:
```
lb = sum_d  max(0, q_d - bmax_d)^2 + max(0, bmin_d - q_d)^2
if lb >= worst_top5: skip this entire cluster
```

If a cluster's bbox-LB exceeds the current worst-of-top-5, the cluster cannot possibly contain a top-5 neighbor and is skipped without touching its vectors. This is the same idea we already use, but applied to k-means cells instead of grid cells — and k-means cells are *much* more selective because they actually concentrate similar vectors together.

Inside the cluster, the same idea is applied at the block level: every 8-vector block has its own implicit lower bound formed by tracking the partial-distance accumulator at dim 4 / dim 6 / dim 8 and comparing all 8 lanes against the worst-of-top-5 with `_mm256_cmpgt_ps + _mm256_movemask_ps`. If the mask is zero (all 8 lanes already exceed worst), the block is abandoned and the remaining dims are not computed.

This is the **3-stage early-exit at dims 4/6/8** that Repo A (cpp-ivf) and Repo #9 (joycegodinho-go) document. The pattern is fundamental.

### 5. Adaptive nprobe: two-tier search

A fixed `nprobe` either over-scans (slow on easy cases) or under-scans (wrong on hard cases). The universal solution is **two tiers**:

- Fast pass: `nprobe = 4-8` (`FAST_NPROBE`)
- Slow pass: `nprobe = 20-48` (`FULL_NPROBE`)

Decide whether to escalate based on the **fraud count** from the fast pass:
- If count is `0` (clearly clean) or `5` (clearly fraud) → return immediately.
- If count is `2` or `3` → ambiguous, escalate.
- Some implementations also escalate on `1` and `4` to be safe.

The Repo B (`whereisanzi`, rank 10) variant is particularly elegant: instead of escalating to `FULL_NPROBE`, it scans **all remaining clusters** with strict AABB-LB pruning — so the escalation is exact (cannot miss a neighbor) but skips ~95% of clusters whose bbox makes them impossible candidates.

This adaptive policy is what holds FP=FN=0. A fixed nprobe is either wrong on hard cases or always-slow.

### 6. Server: hand-rolled, UDS, pipelined

No top submission uses a framework HTTP server on the hot path.

The Rust submissions use **monoio** (single-threaded io_uring runtime) or roll their own raw io_uring loops. The Zig and C++ submissions use raw epoll or raw io_uring. The .NET submission uses minimal-API Kestrel but with the io_uring transport plugin and Date-header suppressed. The Go submissions either use **fasthttp on a UDS** (joycegodinho) or write a **custom HTTP/1.1 parser on UDS** (steixeira93).

The common shape:
- Listen on a **Unix Domain Socket**, not TCP.
- Single thread per process (the kernel does the multiplexing via epoll/io_uring).
- Two API processes in compose, each bound to its own UDS.
- **HTTP pipelining is supported**: the response is appended to a vectored write buffer and emitted with `writev` so multiple responses share one syscall.
- **6 pre-rendered response byte strings** (one per fraud count 0..5) with `Content-Length` and `Connection: keep-alive` already baked in. Hot path is one byte-slice copy.

### 7. Parser: hand-rolled positional

Nobody uses `encoding/json`, `serde_json`, `System.Text.Json`'s `JsonSerializer`, or any general JSON parser on the hot path. Everybody hand-rolls a **positional scanner**.

The trick: the spec defines field order, so the parser doesn't compare key names — it walks through colons and quotes, extracting values by position. Date parsing is byte-index arithmetic on the fixed `YYYY-MM-DDTHH:MM:SS` prefix (no `strptime`, no `time.Parse`). MCC lookup is a compile-time inlined switch or a 10000-entry `int16` table (we already do this).

Repo A (`andrade-cpp-ivf`): "Hand-rolled, structural-positional. Walks `to_next_value()` between colons/quotes — no key matching."

Repo B (`jairoblatt-rust`): "Custom `parse_f32` (integer*10 + fractional*10^-d lookup via `FRAC_POWERS`), custom ISO-8601 parser that indexes into the string at fixed offsets."

The parser cost is ~50-200 ns per request. Encoding/json on a ~500-byte payload is ~10 µs. That's the difference between a 1 ms p99 and a 10 ms p99 on parse alone.

### 8. Load balancer: `SCM_RIGHTS` fd-passing (top 4 only)

This is the most differentiating trick in the top 4. Instead of HAProxy or nginx — which read every request byte to round-robin it — the top 4 ship a tiny LB binary that:

1. Listens on TCP `:9999`.
2. `accept4()`s the client connection.
3. Sends the **accepted fd** over a Unix `SOCK_SEQPACKET` control socket to one of the API processes using `sendmsg + SCM_RIGHTS`.
4. Closes its copy of the fd.

The API process `recvmsg`s the fd, registers it with its own epoll, and speaks HTTP directly to the client. **The LB never sees a byte of the request or the response.** The kernel routes packets straight from client to API.

This trick is what gets the top 4 from ~1.4 ms p99 to ~1.05 ms p99. The author of Repo #1 explicitly attributes the breakthrough in his daily report to swapping nginx for SCM_RIGHTS:

> *"Before: nginx stream proxy, p99 ≈ 2.83 ms, score 5548.91. After FD-passing: p99 1.06 ms, score 5976.27. +427 score points from this one change."*

Two participants ship the LB as a **public reusable image**:
- `jrblatt/so-no-forevis:v1.0.0` (used by Repo #1, #5)
- `ghcr.io/steixeira93/rinha-lb:preview-v20` (Go-based, uses `splice(2)` instead of fd-pass — kernel zero-copy proxy)

So we can adopt the trick without writing our own LB. We just `image: jrblatt/so-no-forevis:v1.0.0` in compose and implement `recvmsg(SCM_RIGHTS)` in our Go server.

### 9. Compose tricks

The pattern across all submissions:

```yaml
api1:
  cpus: "0.40-0.45"
  memory: 145-160M
api2:
  cpus: "0.40-0.45"
  memory: 145-160M
lb:
  cpus: "0.10-0.20"
  memory: 30-60M
```

Total: exactly 1.00 CPU + 350 MB.

Specific tricks:
- `security_opt: seccomp:unconfined` — required so io_uring syscalls (`io_uring_register`, `io_uring_enter`) are not blocked by the default seccomp filter. Multiple submissions use this even without io_uring just for `accept4`/`splice` headroom.
- `ulimits.nofile: 65535/65535`.
- A shared named volume for UDS files (e.g. `sockets:/run/sock`).
- Same `image:` URI for `api1` and `api2`: the kernel page-cache is keyed by inode, so **one image → both replicas share the page cache for the mmap'd index** (effectively halving the resident memory cost of the index).
- The index file is either **baked into the image at build time** (one Dockerfile stage runs the k-means + serialization, then the final stage copies the resulting `index.bin`) or **produced by a one-shot builder container** that writes to a shared volume.
- `logging: driver: none` (no stdout/stderr to disk).

The .NET submission (`pedrosakuma`, rank #8) adds `cpuset: "0,1"` and `cpuset: "2,3"` to pin each replica to specific logical CPUs and prevent CFS migration jitter. This is more relevant in CI than on the contest hardware (which has 1 physical CPU), but the pattern is worth knowing.

### 10. Build flags

The consensus:
- **Haswell baseline**: `-march=haswell` / `-Dcpu=haswell` / `GOAMD64=v3` / `IlcInstructionSet=x86-64-v3` / `RUSTFLAGS=-C target-cpu=haswell`.
- **LTO fat / link-time optimization**: turned on everywhere except Go (which uses PGO).
- **Strip symbols**: `-ldflags="-s -w"` / `--strip` / `<StripSymbols>true</StripSymbols>`.
- **Panic = abort / no-exceptions**: `panic="abort"` (Rust), `-fno-exceptions -fno-rtti` (C++), unwind tables disabled.
- **One codegen unit**: `codegen-units=1` (Rust), single TU after LTO (C/C++/Zig).
- **PGO (Go-specific)**: a `default.pgo` is committed and Go auto-PGO inlines hot paths.

We're already at `GOAMD64=v3` and PGO. We're at parity here.

### 11. Runtime tuning

- **GC off after warmup** (Go): `runtime.GC(); debug.SetGCPercent(-1); debug.SetMemoryLimit(MaxInt64)`. After the index is loaded and warm, no more allocations happen on the hot path, so GC is pure overhead — turn it off. Both Go top submissions do this.
- **GOMAXPROCS(1)** explicitly (we already do this).
- **mimalloc** (Rust): global allocator set to mimalloc to avoid glibc malloc's per-thread arenas.
- **Workstation GC** (.NET): `DOTNET_gcServer=0`, `DOTNET_GCHeapCount=1`, `GCSettings.LatencyMode=SustainedLowLatency`.
- **In-process warmup**: ~200-500 random fraud-score iterations before opening the listener, so the page cache, branch predictor, and i-cache are primed. Some submissions use an external `curl` sidecar that hits `/fraud-score` 48 times after `/ready` succeeds.

### 12. Index loading

- mmap with `MAP_PRIVATE | PROT_READ`.
- `madvise(MADV_RANDOM)` because IVF probes hit non-sequential clusters — disable kernel readahead.
- `madvise(MADV_POPULATE_READ)` or a manual page-walk to fault every page in at startup, so first request doesn't pay page-fault latency.
- `madvise(MADV_HUGEPAGE)` to encourage transparent hugepages and shrink the TLB footprint (a 192 MB index uses 48 000 4 KB pages but only 96 hugepages of 2 MB).
- `mlock` (where `ulimits.memlock: -1` permits) to pin the index in RAM.

## The Go-specific Wins (Repo #6 and #9)

Two Go submissions hit the top 10. Their distinguishing techniques over a "default Go service":

| Trick | steixeira93 (#6) | joycegodinho (#9) |
|---|---|---|
| Plan 9 amd64 assembly | Yes (`internal/blockasm/*.s`) | Yes (`scan_blocks_amd64.s`) |
| `GOAMD64=v3` | Yes | Yes |
| GC permanently off after warmup | Yes (`SetGCPercent(-1)`) | Yes (`SetGCPercent(-1)`, mem limit 120 MB) |
| UDS between LB and API | Yes | Yes |
| Custom HTTP parser | Yes (raw byte loop) | Uses fasthttp on UDS |
| Hand-rolled positional JSON | Yes | Yes |
| Pre-rendered responses | 7 byte slices | 6 byte slices |
| LB | Custom Go w/ `splice(2)` zero-copy | HAProxy TCP-mode → UDS, nbthread 1 |
| Index storage | mmap'd flat `index.bin`, page-warm | gzip-decoded into heap |
| Committed `default.pgo` | No | **Yes** |
| Block-of-8 dim-major SIMD layout | Yes | Yes |
| Centroids SoA in RAM | Yes (`bboxMinCM[c*16+d]`) | Yes |
| Borderline-only escalation | `count ∈ {2,3}` only | `count ∈ {2,3}` only |
| Pin index across Docker builds | **Yes** (copies `/index` from previous tagged image to avoid k-means non-determinism) | No |

The interesting takeaways for us:

1. **Both Go top submissions write Plan 9 assembly for the inner SIMD kernel.** There is no high-level Go SIMD API that matches AVX2 + FMA intrinsics on Haswell. We have to write `.s` files. The standard pattern is one `_amd64.s` file with the AVX2 kernel and one `_generic.go` file with a scalar fallback, gated by `//go:build` tags.
2. **HAProxy can work** (joycegodinho uses it) but only in `mode tcp` with `nbthread 1` and UDS backends. We currently use HAProxy too — but if we're still doing HTTP-mode round-robin, that's an obvious switch.
3. **Pin the index across builds.** k-means is non-deterministic under QEMU emulation; the centroid boundaries shift, and the score moves by ~100 points per build for no algorithmic reason. The steixeira93 Dockerfile copies `/index` from a previous tagged image of itself so the index is frozen across builds. This eliminates measurement noise and makes A/B testing meaningful.

## Where We Stand (Phase 13)

| Aspect | Top 10 consensus | Us (phase 13) | Gap |
|---|---|---|---|
| Index | k-means IVF, K=512-4096 | Grid 16×8×8 with 32 partitions | **Critical** |
| Quantization | int16 × 10 000 | int16 × 32 000 | Cosmetic |
| Layout | 8-vec dim-major blocks | Row-major per cell | **Critical** |
| Pruning | AABB-LB per cluster + 3-stage early-exit at dims 4/6/8 | AABB-LB per cell, no inner early-exit | Significant |
| Adaptive nprobe | fast (4-8) → full (20-48), borderline-only | fixed grid sweep | Significant |
| HTTP server | Raw UDS, hand-rolled, pipelined `writev` | Raw HTTP/1.1 on UDS (we already do this) | None |
| Parser | Hand-rolled positional | Hand-rolled (we already do this) | None |
| Responses | 6 pre-rendered byte strings | 6 pre-rendered byte strings (we already do this) | None |
| LB | SCM_RIGHTS fd-passing or splice(2) | HAProxy TCP-mode | **Significant** |
| Compose | 2 API @ ~0.42 + LB @ ~0.16, same image, shared inode | 2 API @ ? + LB @ ? | Need to verify |
| GC | Off after warmup | Off (we already do this) | None |
| Build flags | `GOAMD64=v3` + PGO | `GOAMD64=v3` + PGO (we already do this) | None |
| SIMD | Hand-written AVX2 (asm or intrinsics) | Compiler auto-vec only | **Critical** |
| Index mmap | mmap + MADV_RANDOM + MADV_POPULATE | Heap load via gzip | Significant |
| Warmup | 200-500 in-process iterations | Unknown — verify | Verify |

The four "Critical" rows are where the 80× p99 gap lives. None of them are about CPU or memory limits — they're all algorithmic and layout decisions we made and then iterated within. We need to step out of the grid framework entirely.

## The Roadmap

Ranked by expected delta to score, smallest implementation effort first:

### Tier 1 — port the consensus algorithm (3-7 days, expected p99 5-15 ms, score ~5400-5700)

1. **Replace grid with k-means IVF.** K=4096 (most top submissions use this number; balances fast-pass coverage vs centroid-distance cost). Train with Lloyd's algorithm at Docker build time on a 50-65k deterministic sample. K-means++ init. 6-25 iterations is enough. Centroids stored f32, SoA (`centroids[d*K + c]`). Output a single `index.bin` consumed by the runtime.
2. **Block-of-8 dim-major layout.** Per cluster, materialize a stream of 224-byte blocks: `block[d*8 + lane]` as int16. Pad with `INT16_MAX` so phantom lanes never beat any real candidate. Centroids and bounding boxes stored alongside.
3. **AVX2 distance kernel in Plan 9 assembly.** One `.s` file with `scanBlock8AVX2(q *float32, block *int16, worst float32) (acc [8]float32, mask uint8)`. Distance accumulates in 8 f32 lanes via `VPMOVSXWD + VCVTDQ2PS + VBROADCASTSS + VSUBPS + VFMADD231PS`. 3-stage early-exit at dims 4/6/8 via `VCMPPS + VMOVMSKPS`. Generic fallback in `_generic.go`.
4. **AABB-LB pruning per cluster.** Use the bounding boxes you already compute, but in IVF cluster space rather than grid space. Skip clusters whose `lb ≥ worst_top5`.
5. **Two-tier nprobe.** `FAST_NPROBE=8`, `FULL_NPROBE=24` (or "all-remaining with bbox-LB" — whichever benches faster). Escalate only on `fraud_count ∈ {2,3}`.

Outcome: same FP=FN=0 we already have, plus a 20-50× p99 reduction. Score should jump to the 5400-5700 range.

### Tier 2 — eliminate the LB-as-proxy (1-2 days, expected p99 1.5-3 ms, score ~5800-5900)

6. **Switch from HAProxy to SCM_RIGHTS fd-passing.** Two options:
   - **Option A (zero code, free):** Drop in `jrblatt/so-no-forevis:v1.0.0` as the LB image. Implement `recvmsg(SCM_RIGHTS)` in our Go server to consume passed fds. ~100 LOC for the recvmsg path using `golang.org/x/sys/unix.Recvmsg` and `unix.ParseSocketControlMessage`.
   - **Option B (more code, more control):** Use the `steixeira93/rinha-lb:preview-v20` splice-based image, or write our own splice(2) zero-copy proxy in Go (also ~200 LOC).

Outcome: p99 drops to ~1.5-3 ms. Score should be in the 5800-5900 range.

### Tier 3 — micro-tuning for the last 100 points (2-5 days, expected score 5900-5970)

7. **Index mmap + MADV_RANDOM + MADV_POPULATE_READ.** Move from gzip-decode-to-heap to mmap a flat `index.bin`. Page-walk at startup. Halves the heap pressure on the Go side and lets two API replicas share the kernel page cache via shared inode.
8. **Pre-rendered HTTP responses, vectored writev for pipelining.** Probably already done; verify and ensure the response slices are `[]byte` literals (not built per request).
9. **In-process warmup loop.** 500 random fraud-score iterations before opening the listener.
10. **Pin the index across Docker builds.** Either bake it into the image (so the SHA freezes the cluster boundaries) or COPY it from a previous tagged image.
11. **Compose tweaks**: `seccomp:unconfined` on all services, `ulimits.nofile: 65535`, `logging: driver: none`, identical `image:` URI on api1 and api2 for shared page cache.

Outcome: p99 in the 1.1-1.5 ms range. Score 5900-5970. We're now indistinguishable from rank #1-5 on technique.

### Tier 4 — chase the last 30 points to #1 (not worth it for learning, but documented)

12. **Hand-tuned "extreme repair" zones** (Repo #1's trick). After Tier 3, look at the small handful of remaining mispredictions on the test set; if they cluster around specific feature ranges, add hard-coded `if`-zones that force a full-scan. Repo #1 documents this explicitly in `ivf.cpp:610-638` — it's how they hold FN=FP=0 across the entire test set. Buys maybe 5-15 score points; not algorithmically interesting but worth knowing exists.
13. **HTTP pipelining with `writev` iovec batching** (Repo #2's trick). When multiple requests arrive on the same connection in one `read`, queue all responses into an `iovec[]` and emit them with one `writev`. Reduces syscall count under load.
14. **CPU pinning via `cpuset:`** (Repo #8's trick). Probably no effect on the contest hardware (1 physical CPU) but useful for stable benchmarking.

## What We Don't Need to Do

A few rabbit-holes we can definitively skip based on the survey:

- **int8 quantization**: nobody in the top 10 uses pure int8 for the reference store. Repo #8 (.NET) uses int8 only for the IVF inner scan with a Q16 re-rank — that's an optimization on top of IVF, not a replacement for int16. We already empirically eliminated int8 (FP=51, FN=62 in our phase 11), and the survey confirms it's not a winning representation.
- **Pure float32**: nobody uses pure f32 references. The combination of int16 storage + f32 distance computation is universal. We already empirically eliminated f32 (FP=0, FN=1, same as int16, plus 18% slower — phase 12).
- **HNSW**: despite Repo #6's name "go-hnsw", the actual code uses IVF — the HNSW path is dead. No top submission uses HNSW on the hot path. The graph-traversal cost dominates at this scale for low `efSearch` and there is no precision win over IVF + AABB-LB for exact-on-the-tail-end-classification.
- **Annoy / LSH**: nobody uses these. They're approximate and the contest punishes detection errors at weight 3× the FP weight.
- **GPU**: not in any submission. Single CPU, single core. Stay scalar/SIMD.

## Closing Thought

The most important takeaway from the survey is that **there isn't a secret algorithm**. The top 10 all do the same thing. The difference between rank #1 (5983) and rank #10 (5853) is which third of the four "Critical" rows above each implements perfectly. The difference between rank #10 and us is which third of the four "Critical" rows each implements *at all*.

We have the engineering chops to implement all four. The architectural reroute is mostly mechanical: build a k-means index, write one `.s` file, change the LB image, and run the bench. Our existing scaffolding — UDS server, hand-rolled parser, pre-rendered responses, build flags, PGO, load shedder, GC-off — all stays.

Next phase target: 5800. After Tier 1 alone we should clear that.
