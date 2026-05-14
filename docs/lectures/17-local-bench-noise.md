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
