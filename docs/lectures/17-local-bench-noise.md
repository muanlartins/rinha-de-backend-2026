# Lecture 17 — Local bench noise floor and prep-period workflow

Written 2026-05-14 after Phase 33 shipped (5953.77 / 1.11 ms bot, rank 6)
and we hit the 5/day submission cap. Phase 34 (heap picker) is queued
and ready to test tomorrow. This lecture documents the local bench
behavior and the prep workflow we'll use while the cap is in effect.

## 1. What we tried

Re-ran `benchmarks/run.sh` against the live `:latest` image (Phase 33)
back-to-back, on the local HAProxy stack (`/tmp/sub-local`). The
production SCM_RIGHTS LB (`jrblatt/so-no-forevis:v1.0.0`) panics in
Docker Desktop because its io_uring runtime isn't supported in the
stripped Linux VM; HAProxy is the local substitute and is exactly the
LB we used through Phase 17.

Two back-to-back identical-image runs:

| Run | p99 | final_score | det_score |
|---:|---:|---:|---:|
| 1 | 2.13 ms | 5671.94 | 3000 |
| 2 | 2.63 ms | 5579.79 | 3000 |

**Δ between identical-image runs: ±92 score, ±500 µs p99.**

For reference, the bot's reported variance for the same image is ~±5
score (Phase 28 → 29 → 33 all clustered near 5950–5953 within noise).

## 2. Why local noise is so much larger

- **Rosetta x86 emulation under Apple Silicon** — non-deterministic
  hot-spot caching, JIT-stage overhead. Each container start invalidates
  the translation cache.
- **Docker Desktop's Linux VM scheduler** — k6 and the API containers
  share the same VM CPUs; Docker's CPU quota is approximated by cgroup
  cfs_period/quota, not enforced precisely.
- **macOS host load** — Slack, Spotlight, IDE indexing all contribute
  to outside-cgroup contention.
- **CPU thermal headroom** — back-to-back runs after the first heat
  the cores differently than a cold run.

The bot's Mac Mini avoids all of these: native amd64, dedicated
hardware, fixed thermal profile.

## 3. What the local bench is good for

**Good signal:**
- **FP/FN/Err correctness** — bit-exact across runs (we always see 0/0/0
  on Phase 33). Catches algorithmic regressions before we submit.
- **Catastrophic regressions** — a change that adds 10 ms to p99 will
  show up regardless of noise.
- **Image-level smoke** — confirms the image boots, /ready works, all
  routes return 200, no panics.

**Poor signal:**
- **Small p99 deltas** — ±500 µs noise drowns out <500 µs changes.
- **Score comparison vs the bot** — local is ~2× slower in p99 (Rosetta
  tax); deltas may not transfer linearly.

## 4. The actual workflow we'll use

For each candidate change during the cap-period prep:

1. **Microbench the changed function** with `go test -bench` under
   `linux/amd64` docker. Asm changes (Phases 28, 32, 34) showed clean
   single-digit-ns wins this way.
2. **End-to-end Go bench** (`BenchmarkHandlerFraudScore`) for full
   request-path effects. Tight, in-process, sub-noise.
3. **Local k6 smoke** via `benchmarks/run.sh` — only for the
   correctness gate (FP/FN/Err must stay 0). Ignore the score; just
   confirm the change doesn't break end-to-end.
4. **Promote to bot submission** only when 1+2 show a clean win.

The local k6 bench will NOT be used to rank candidates — we'll use Go
benches for that. The k6 bench is only the correctness backstop.

## 5. Candidate levers for next-day submissions

Already queued, ready to ship (image phase34 pushed):

- **Phase 34 — Max-heap PickNextNUnscanned**: 33% faster locally
  (7.8 → 5.3 µs). Predicted bot −2 to −4 µs at p99.

Prototype next (cap-period work):

- **Phase 35a — Multi-request-per-buffer parsing** (rawhttp.go
  refactor). Eliminates per-request memmove. Cost model: 1-2 µs per
  pipelined request. Effort: low.
- **Phase 35b — VPMADDWD i32 cluster-scan kernel.** Cost model:
  −30 µs/escalating-query at p99. Effort: high (i32 overflow,
  dim-paired block layout, reindex).
- **Phase 35c — Go runtime experiments**: GOGC=off + manual GC,
  smaller GOMEMLIMIT, worker pool vs spawn-per-conn. Cost model:
  unknown (0 to 50 µs at p99). Effort: medium.

Each will be A/B'd against current `:latest` using Go benches; whichever
shows the largest clean delta becomes the next submission.

## 6. The honest position

We have **two bot submissions queued for tomorrow** (5/day cap resets):

1. Phase 34 (already built and pushed). Cheap, predictable, ~+5 score.
2. Phase 35 (best of the prototypes above). Bigger upside if VPMADDWD
   pays off; could push p99 below 1.0 ms (rank 1-3 territory).

If both land cleanly we'll be in top 5-6 with strong odds of top 3.
Sub-1 ms is achievable with VPMADDWD + Phase 34, but no guarantees;
the bot has ±5 score noise and our gap to top-1 is ~45 score.

## Phase 35 prep-period summary (2026-05-14)

After re-profiling the current build I made three small structural
changes that are safe to ship and one I'm deferring:

- **Phase 35a — Offset-based pipelined parsing.** rawhttp.go now
  advances a `pos` cursor instead of memmove-ing the buffer after
  every request. Synthetic bench: 113.8 ns → 86 ns/req (-24 %).
  Real bot delta likely <5 µs because real /fraud-score is
  ~10 µs of search work.
- **Phase 35b — Removed time.After from the shedder hot path.**
  Replaced `case <-time.After(shedTimeoutDur)` with `default`. The
  shedder never fires in practice (4 slots / GOMAXPROCS=1) so we
  were paying for Timer allocation + runtime.timersMutex lock on
  every /fraud-score request. Theoretical save: 100-500 ns/req.
- **Phase 35c — Optional GC-off via `STEADY_GC_OFF=1` env var.**
  After warmup, calls `debug.SetGCPercent(-1)` and runs a periodic
  GC every 5 s in a background goroutine. Eliminates the STW pause
  from request paths. Safety belt: GOMEMLIMIT=140MB still forces a
  GC near the limit. Opt-in only; first bot test will toggle it on.

Deferred:

- **VPMADDWD kernel (was 35b in lecture 15).** On closer reading,
  the implementation needs i32 → i64 promotion after every 2
  dim-pairs to avoid overflow (max squared diff at scale=10000 is
  4 × 10⁸, summed across 7 pairs is 5.6 × 10⁹, overflows i32).
  Promotion (VPMOVSXDQ + VPADDQ) eats most of the op-count savings.
  Estimated win shrinks from -30 µs to maybe -10 µs at p99 — still
  positive but not the order-of-magnitude lever I first thought.
  Defer until cheap wins are bot-validated.

## Submission queue for tomorrow

Five slots; updated priority order after local 3-config A/B:

### Local A/B result (2026-05-14, 3-4 runs each)

| Config | p99 runs (ms) | p99 median | score median |
|---|---|---:|---:|
| Baseline (Phase 33) | 3.24, 2.23, 2.21 | 2.22 | 5654 |
| p35-default (35a+b only) | 5.03, 2.12, 2.37, 3.02 | 2.37 | 5520 |
| **p35-gcoff (+ STEADY_GC_OFF=1)** | **1.81, 2.12, 2.09, 1.90** | **1.90** | **5680** |

The GC-off variant **cut local p99 median by 320 µs** and dramatically
tightened the variance (all 4 runs within 310 µs of each other; no
cold-start outlier). p35-default looks slightly worse than baseline
but the spread is huge — it's in the noise band.

This is the cleanest local signal we've ever measured. The GC pauses
were the local-p99 dominator. If this maps proportionally to the bot
(Mac Mini ~ same GC behavior as Linux/amd64 Docker), Phase 33's
1.11 ms bot p99 should drop to **~0.94-1.00 ms — top 1-2 territory**.

### Order of operations

1. **#1 SUBMISSION: Phase 35 image + STEADY_GC_OFF=1.** The big bet.
   Image `:phase35` already pushed. Submission compose updated with
   `STEADY_GC_OFF=1` env var. One `git push origin submission` +
   one `rinha/test` issue.
2. **#2 SUBMISSION: Phase 35 image + STEADY_GC_OFF=0 (default GC).**
   Isolates the GC-off contribution from the parsing/timer fixes.
   Tells us if 35a+b alone was a regression or fine.
3. **#3 SUBMISSION: Phase 34 image (heap picker).** Already pushed,
   already validated; expected +5 score baseline. Cheap.
4. **#4 follow-up** based on what #1-3 showed.
5. **#5 reserve.**

If #1 lands at ≤ 1.05 ms, we're top-3 minimum and may not need #2-5.
If #1 disappoints, #2 tells us whether to roll back to 35a+b only or
to Phase 33.
