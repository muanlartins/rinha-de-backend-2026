# Implementation Plan — Tiers 1 through 4

Based on lecture 08 (top-10 survey). The plan is laid out **layer by layer**, with a **concept lecture before each layer** so the underlying idea is documented before we touch code.

Every layer has:
- **Lecture**: the concept-level write-up that goes into `docs/lectures/` before code is written.
- **Implementation**: the code changes.
- **Local tests**: correctness check on the reference dataset, then a smoke benchmark on Apple Silicon (knowing the absolute numbers won't transfer, but the relative shape will).
- **Validation**: build a Docker image, open a rinha test issue, wait for the Mac Mini result.
- **Commit/push**: each layer ships as its own commit.

The plan is sequential by necessity — each layer depends on the previous. We don't parallelize implementation, but we *do* test each layer in isolation before stacking the next.

## Current state (Phase 13)

- Grid 16×8×8 partitioned by dims (0, 12, 7) into 32 partitions × ≤1024 cells.
- int16 quantization × 32 000 with `-32000` sentinel.
- AABB-LB pruning over grid cells.
- Per-vector scalar early-exit (no SIMD).
- Custom raw HTTP/1.1 on UDS.
- HAProxy → 2 API containers.
- `GOAMD64=v3` + PGO + load shedder.
- Score 3823, p99 99.25 ms. Detection saturated at 2819/3000.

## Target state (after Tier 3)

- k-means IVF, K=4096, two-tier nprobe.
- int16 × 10 000 quantization in 8-vector dim-major blocks.
- Plan 9 AVX2 distance kernel with 3-stage early-exit at dims 4/6/8.
- Per-cluster AABB-LB pruning.
- mmap'd index with `MADV_RANDOM + MADV_POPULATE_READ + MADV_HUGEPAGE`.
- SCM_RIGHTS fd-passing LB (`jrblatt/so-no-forevis:v1.0.0` consumed by our recvmsg loop).
- Same image for api1+api2 (shared kernel page cache).
- In-process warmup before listener opens.
- Score target 5900-5970, p99 1.1-1.5 ms.

## Phase 14 — k-means IVF index foundation

**Lecture (09):** *"k-means IVF — why data-aware cells crush axis-aligned grids"*. Covers Lloyd's algorithm, k-means++ initialization, the choice of K, sample-then-assign vs full-batch training, why centroids must be stored SoA, why bounding boxes per cluster (not just centroids) are essential for exact recall, the on-disk format design.

**Implementation:**
- New package `internal/ivf/`:
  - `kmeans.go`: k-means++ init on a sample (size 65 536), Lloyd's algorithm (6-25 iterations), full-batch assignment of all 3M vectors to nearest cluster.
  - `index.go`: in-memory representation (centroids, blocks, bboxes, offsets, labels). Serialization to v5 on-disk format.
- New `cmd/build-index/`: standalone binary. Reads `references.json.gz`, builds k-means index, writes `index.bin`. Already exists as a stub — replace with the real builder.
- New on-disk format `IVF8` v5:
  - Header: magic, version, n, K, dim, quant_scale.
  - Centroids: `[K][14]f32` stored SoA (`centroids[d*K + c]`).
  - Cluster offsets: `[K+1]u32` (start of each cluster in blocks).
  - Bounding boxes: `[K][14]int16` for both min and max, padded with zeros for unused lanes.
  - Labels: `[N]u8` (we keep this — fraud bit per reference).
  - Blocks: stream of 8-vector dim-major panels. Each panel is `[14][8]int16` = 224 bytes. Padded with `INT16_MAX` sentinels for partial last block per cluster.

**Tests:**
- `TestKMeansConverges` — train on the actual references, assert inertia decreases monotonically.
- `TestIVFIndexRoundTrip` — write index to bytes, read back, byte-equal.
- `TestIVFIndexExactCount` — for 1000 sampled queries, brute force returns the same top-5 set (modulo permutation of ties) as IVF with `nprobe=K` (i.e., scan-all).

**Local check:** time the build. Target: under 90s on the local machine (M-series can train 4096 centroids over a 65k sample in ~2-5s, then assign 3M vectors in ~30-60s).

**No deploy yet** — this layer is silent. The grid path still runs.

## Phase 15 — Plan 9 AVX2 distance kernel

**Lecture (10):** *"SIMD in Go without cgo — Plan 9 assembly for AVX2 + FMA"*. Covers Go's Plan 9 assembler dialect, register naming (`Y0..Y15`, `X0..X15`), 4-operand AVX vs FMA encoding, why `MOVOU` doesn't exist (use raw bytes for `VMOVDQU`), how to call Go functions from `.s` files, how to gate with `//go:build amd64`, the FuncData/ArgsSize/StackMap requirements, NOSPLIT, and how the generic fallback file is structured.

**Implementation:**
- New package `internal/kernel/`:
  - `dist_amd64.s`: the AVX2 + FMA distance kernel.
    - Signature: `ScanBlock8AVX2(q *float32, block *int16, worst float32) (sum [8]float32, anyAlive bool)`.
    - 14 dims × 8 lanes. After dim 4, `VCMPPS` against `VBROADCASTSS worst` and `VMOVMSKPS`; if zero, return. Same checks at dim 6 and dim 8.
    - Returns the 8 partial-sum f32 lanes (caller does the top-5 insertion).
  - `dist_generic.go`: the scalar fallback. Identical signature, runs on any arch. Used by tests on Apple Silicon and as the verification oracle.
  - `dist_amd64_test.go`: cross-check the amd64 kernel against the generic kernel on 1000 random inputs.

- Also add `Centroid8AVX2(q *float32, centroids *float32, dimStride int) [8]float32` — used to score 8 centroids per call when computing nearest cells.

**Tests:**
- `TestScanBlock8MatchesGeneric` — 1000 random `(q, block, worst)` tuples; the amd64 and generic kernels must agree to within `1e-3` (f32 fma rounding).
- `TestEarlyExitCorrect` — when all 8 lanes are guaranteed dead at dim 4, `anyAlive` must be false and the partial sums after dim 4 must be accurate.
- `BenchmarkScanBlock8AVX2` vs `BenchmarkScanBlock8Generic` — for context. Expect 4-8× speedup locally.

**Local check:** on Apple Silicon under Rosetta the AVX2 path will not run — the `_test.go` file should fall back to the generic path. Run the test under `linux/amd64` via `GOARCH=amd64 GOOS=linux go test -c` and execute under `colima` or via the Docker dev image we already have.

**No deploy yet** — this layer is silent.

## Phase 16 — IVF search with two-tier nprobe + AABB-LB

**Lecture (11):** *"Two-tier IVF — adaptive `nprobe` and the borderline-only escalation policy"*. Covers `nprobe` as the recall knob, the empirical observation that count-aware escalation captures hard cases at a fraction of the cost, the AABB-LB bound formula (`sum_d max(0, q-bmax)² + max(0, bmin-q)²`) and why it's a sound under-estimate of best-possible distance, why escalating on `count ∈ {2,3}` is sufficient (the only fraud_count values whose binary classification depends on getting the 5th neighbor exactly), and the cascading top-5 insert pattern.

**Implementation:**
- New `internal/search/ivf.go`:
  - `FraudCountIVF(q *[14]int16, qf *[14]f32, ds *IVFIndex) uint8` — the new hot path.
  - Step 1: compute `centroidDists[K]` via `Centroid8AVX2` over 8 centroids at a time.
  - Step 2: pick top `nprobe=8` closest centroids via a small insertion sort (we only need top-N for small N).
  - Step 3: scan each picked cluster. For each cluster, first check `bboxLowerBound(qf, bbox) >= top5.worst`; if so, skip. Otherwise iterate blocks, call `ScanBlock8AVX2`, insert results into top5.
  - Step 4: count frauds among top-5 labels.
  - Step 5: if `count in {2, 3}`, escalate. Two options to bench against each other:
    - **Option A**: scan top-24 closest centroids (more probes).
    - **Option B**: scan all remaining clusters with strict bbox-LB pruning.
  - Step 6: return the count.

- Modify `internal/search/grid.go` → mark for deletion after Phase 17. Keep around until then for A/B comparison in tests.

**Tests:**
- `TestIVFFullDataset` — load IVF index, run all 54 100 entries of `test-data.json`, assert `FP=0, FN ≤ 1, Err=0`. Same metric as the current `TestGridFullDataset`. We expect to match the current FN=1 (the structural one) or beat it; we will NOT accept any regression.
- `TestIVFMatchesBruteForce` — on 1000 sampled queries, the IVF's top-5 fraud count must match the brute-force oracle exactly.
- `BenchmarkFraudCountIVF` vs `BenchmarkFraudCountGrid` — local relative perf.

**Local check:** the test suite is the gate. Must pass before any deploy.

**Deploy:** swap `cmd/api/main.go` to load IVF instead of grid, build the image, push to Docker Hub as a new tag (e.g., `phase-14-15-16-ivf`), open a rinha test issue.

**Expected score:** 5400-5700. This is Tier 1 complete in one shipping step.

## Phase 17 — drop the grid, lock in Tier 1

**Lecture (none — refactor only.)**

**Implementation:**
- Delete `internal/dataset/grid.go`, `internal/search/grid.go`, `internal/search/grid_test.go`.
- Delete the partition-key code in `internal/dataset/dataset.go`.
- Simplify `Dataset` struct → `IVFIndex`.
- Update `cmd/api/main.go` to load `index.bin` v5 format only.
- Delete v4 format reader (we don't need backward compatibility).
- Update `Dockerfile` to invoke `build-index` with new flags.
- Update lectures 05-07 with a final-state pointer to lecture 11.

**Tests:** every test that was passing must still pass.

**Commit & push.** Tag the commit as `tier1-complete`.

## Phase 18 — SCM_RIGHTS fd-passing LB (Tier 2)

**Lecture (12):** *"SCM_RIGHTS file-descriptor passing — how the top 4 skip a userspace hop"*. Covers Unix domain sockets in `SOCK_SEQPACKET` mode, `sendmsg(2)` with `SCM_RIGHTS` ancillary data, how the kernel duplicates the fd into the receiving process's fd table, why this eliminates byte copies in the LB, the security and reliability properties, the `so-no-forevis` protocol (binary fd + 4-byte header), how to consume passed fds in a Go epoll loop using `unix.Recvmsg` + `unix.ParseSocketControlMessage` + `os.NewFile`, and the implications for keep-alive (the API process now owns the entire connection lifecycle including TCP teardown).

**Implementation:**
- New package `internal/fdpass/`:
  - `recv_linux.go`: Linux-only file. `Listen(ctrlPath string) <-chan int` — opens the `.ctrl` UDS, runs a goroutine that `recvmsg`s fds and yields them on the channel.
  - `recv_other.go`: stub on darwin/win for local tests (channel is closed immediately).
- Modify `internal/api/server.go`:
  - On Linux, start the fd-passing listener.
  - In the main accept loop, prefer fds from the fdpass channel; fall through to local `accept` if the channel is empty (so local dev still works).
- Modify `deploy/docker-compose.yml`:
  - Swap HAProxy for `jrblatt/so-no-forevis:v1.0.0`.
  - Configure `UPSTREAMS=/sockets/api-1.sock,/sockets/api-2.sock`.
  - Add the `.ctrl` UDS to each api container's expected sockets dir.
  - Add `security_opt: seccomp:unconfined` to LB (already there) and all API services.

**Tests:**
- `TestFdPassRecvLoopback` — Linux-only. Bring up a pair of UDS endpoints, send a self-`accept`ed fd via `SCM_RIGHTS`, verify the receiver gets a usable fd that can `Read`.
- Local integration: `docker compose -f deploy/docker-compose.yml up`, hit `:9999` with `curl`, verify it goes through. (Locally HAProxy was already in compose; this is a like-for-like swap with `so-no-forevis`.)

**Deploy:** build, push, open rinha issue.

**Expected score:** 5800-5900.

## Phase 19 — mmap + warmup + compose hygiene (Tier 3)

**Lecture (13):** *"mmap, MADV_RANDOM, MADV_POPULATE — making 200MB indexes feel like 0MB"*. Covers `mmap(2)` vs `read(2)`, the page-cache lifecycle and why two replicas with the same image inode share pages, `madvise(MADV_RANDOM)` to disable readahead for non-sequential access patterns, `madvise(MADV_POPULATE_READ)` to pre-fault pages at startup, `madvise(MADV_HUGEPAGE)` to encourage transparent hugepages and reduce TLB pressure, why `mlock` is sometimes overkill, and the implications for the Go GC (mmap'd memory doesn't count against the heap).

**Implementation:**
- Modify `internal/ivf/index.go`:
  - Change loader from `read`-into-heap to `mmap`-as-read-only.
  - Apply `madvise(MADV_RANDOM)` + `madvise(MADV_POPULATE_READ)` + `madvise(MADV_HUGEPAGE)`.
  - Index struct fields become `unsafe.Slice` views into the mmap'd buffer.
- New `internal/api/warmup.go`:
  - Runs 500 deterministic fraud-score iterations on a representative subset of `test-data.json` before opening the listener.
  - Goal: prime CPU caches, branch predictor, page cache.
- Modify `deploy/docker-compose.yml`:
  - Add `security_opt: seccomp:unconfined` on every service.
  - Add `ulimits.nofile: 65535:65535`.
  - Add `logging.driver: none`.
  - Confirm api1 + api2 use the same `image:` URI.
  - Drop any per-replica build context.
- Modify `Dockerfile`:
  - Pin the index inside the image (`COPY /index /index`) so the SHA freezes cluster boundaries across builds.

**Tests:**
- `TestIVFMmapReadOnly` — `mmap` the index, verify `bboxLowerBound` and `ScanBlock8AVX2` work over the mmap'd slices without copy.
- `TestWarmupCompletesBeforeListen` — start the api in a subprocess, hit `/ready` immediately, expect it to return 503 until warmup completes (or, configurable, simply log the warmup duration).

**Deploy:** build, push, open rinha issue.

**Expected score:** 5900-5970.

## Phase 20 — polish (Tier 4)

Only pursue these if Phase 19's score is below 5900. Each is a small delta; we'd rather stop here than over-engineer.

- HTTP pipelining with `writev` iovec batching (~15 points typical).
- Hand-tuned "extreme repair" zones for residual mispredictions (~5-15 points).
- CPU pinning via `cpuset:` if Mac Mini has multiple logical CPUs.

## Risk register

| Risk | Mitigation |
|---|---|
| K-means under QEMU emulation produces non-deterministic centroids → score variance | Run k-means with a fixed RNG seed (`math/rand/v2.NewPCG(0xCAFE, 0xBABE)`). Bake the index into the image. |
| Plan 9 assembly has subtle bugs | Cross-check every kernel against the generic fallback in tests; fail tests on any mismatch. |
| `so-no-forevis:v1.0.0` protocol changes break compatibility | Pin the exact tag. Document the protocol in lecture 12. |
| mmap'd index pages get evicted under memory pressure | Use `MADV_POPULATE_READ` + manual page-walk at startup. If Mac Mini score still degrades over time, add `mlock` (requires `ulimits.memlock: -1`). |
| FN=1 persists across the algorithm change | Expected — it's structural. After Phase 16, dump the parsed query for test-data entry 5472 and bisect. Likely fix is a parser float rounding bug. |

## Schedule

This is sequential work. Estimated calendar time on the local machine with focused execution:

- Phase 14 (k-means index): 4-6 hours including lecture + tests + index build benchmarking.
- Phase 15 (AVX2 kernel): 4-6 hours including the .s file and cross-checks.
- Phase 16 (IVF search): 3-5 hours.
- Phase 17 (refactor + ship): 2-3 hours.
- Phase 18 (fd-pass LB): 3-5 hours.
- Phase 19 (mmap + compose hygiene): 2-3 hours.

Total: roughly two focused working days. We will checkpoint after each Mac Mini score.

## Documentation deliverables

Lectures to write (in order):

- `09-kmeans-ivf.md` — Lloyd, k-means++, sample-train, why K=4096, why bounding boxes.
- `10-plan9-simd.md` — Plan 9 assembler, AVX2 + FMA encoding, generic fallback pattern.
- `11-two-tier-nprobe.md` — adaptive escalation, AABB-LB soundness, cascading top-5 insert.
- `12-scm-rights.md` — Unix sockets, ancillary data, fd table semantics, `so-no-forevis` protocol.
- `13-mmap-madvise.md` — page cache, madvise flags, shared-inode page sharing.

Each lecture explains the concept *first*, then references the file paths where the implementation lives — never the other way around. The goal is that someone reading the lectures in order, without looking at the code, ends up with a complete mental model.

## Definition of done

The plan is complete when:

1. Lectures 09-13 exist and are coherent.
2. The code passes all tests locally on Apple Silicon (generic kernel path).
3. The Docker image builds, gets pushed, and runs on the Mac Mini.
4. Score ≥ 5800 (Tier 2 target).
5. JOURNEY.md captures the phase-by-phase scores.
6. The grid code is gone.
7. The `main` branch is at HEAD and the `submission` branch matches.
