# Submission Process — what we do, when we do it, and how it gets ranked

This doc translates `references/rinha-official/docs/en/SUBMISSION.md` and `EVALUATION.md` into a concrete plan for our entry, plus what's manual vs. automatic, plus the rules around basing our work on the QRust reference.

## 1. How submissions work

There are **two** moments of public evaluation:

1. **Preview test** — *we trigger it, the Rinha Engine runs it.*
   We open a GitHub issue on `zanfranceschi/rinha-de-backend-2026` with the literal text `rinha/test` in the body (optionally followed by an `id`, used when a participant has multiple backends in their `info.json`).
   The Rinha Engine scans open issues with that description, runs the bench on the Mac Mini, posts the resulting `results.json` as a comment, and closes the issue.
   We can run **as many** of these as we want. They're the practice round.

2. **Final test** — *runs once, at the end of the event, by the organizers.* Heavier script, more requests, possibly different scenarios. That's the result that "counts."

In both cases we don't run anything on Rinha's infra ourselves. We register a backend, they run it.

### What "registering a backend" actually means

Two artifacts:

- **A `participants/<github-username>.json`** in the `zanfranceschi/rinha-de-backend-2026` repo, opened via a PR.
  In our case: `participants/muanlartins.json` containing:
  ```json
  [{
      "id": "muanlartins-go",
      "repo": "https://github.com/muanlartins/rinha-de-backend-2026"
  }]
  ```
  The `id` is what we'll reference in `rinha/test <id>` issues if we ever submit a second backend.

- **Our public repo (`github.com/muanlartins/rinha-de-backend-2026`) with two branches:**
  - `main` — holds source code (cmd/, internal/, Dockerfile, etc.).
  - `submission` — holds **only** the runtime artifacts: `docker-compose.yml`, `haproxy.cfg`, `info.json`. No source. The official rule is clear: the `submission` branch *cannot* contain source code.

The Rinha Engine clones the repo, checks out `submission`, and runs `docker compose up`. Our `docker-compose.yml` references a public image (`muanlartins/rinha-de-backend-2026:latest`) — **this image must be on a public registry (Docker Hub) and built for `linux/amd64`**. If the image is private or arm64-only, the bench fails before it even starts. This is one of two "common mistakes" the official FAQ calls out explicitly.

### Submission checklist (from the PR template)

The PR adding `participants/muanlartins.json` runs through this checklist:

- [x] Total across services respects the limit of **1 CPU** and **350 MB RAM** — ours: 0.10 + 0.45 + 0.45 = 1.00 CPU; 16 + 167 + 167 = 350 MB.
- [x] Backend exposes port **9999** — HAProxy binds `*:9999`.
- [x] Images are **linux/amd64** — Dockerfile uses `FROM --platform=linux/amd64`; build uses `GOARCH=amd64`.
- [x] Network mode is **bridge** (default; we don't override it).
- [x] No `network_mode: host` and no `privileged: true`.
- [x] At least **1 load balancer + 2 APIs** — HAProxy + 2 Go replicas.
- [x] Repo is **public** and contains both `main` and `submission` branches.
- [x] `submission` branch has `docker-compose.yml` and `info.json` at root.

## 2. The rules — what we can and cannot do

Pulled from `references/rinha-official/docs/en/FAQ.md` and `SUBMISSION.md`.

**Allowed:**
- Any language, framework, database, technology.
- Vector databases (pgvector, Qdrant, SQLite-vss, etc.) — or no DB at all (our case: in-process int16 grid index).
- **Pre-processing the dataset at container build or startup.** Explicitly endorsed: *"the more processing you move outside of runtime, the better your `p99` tends to be."*
- Exact KNN, approximate ANN, or anything else you can justify with the score.

**Not allowed:**
- Load balancers that perform fraud-detection logic. (Ours doesn't — HAProxy is pure round-robin TCP/HTTP.)
- Hiding part of your submission to prevent others from learning from it. (Repo is public and MIT-licensed.)
- Disrespecting other participants or organizers; making demands of organizers.
- **Using the test payloads as a lookup or reference for fraud detection.** This is the main "cheating" rule — you can't hard-code answers based on the test set.

**Licensing requirement:**
- *"To participate in Rinha, all your repositories must be under the MIT license."*
- Our repo has `LICENSE` (MIT) at the root; the file is on the `main` branch (the source branch). The submission branch doesn't need it but we don't enforce removal — being permissive doesn't violate anything.

## 3. About basing the algorithm on QRust

This is the part worth thinking carefully about. The literal rule is:

> Try to hide your submission or part of it (source code) to prevent others from learning from it.

That's framed as a prohibition against **hiding**, not against **learning from**. The reverse — *learning from someone else's submission* — is the explicit goal of making everyone's code public. So borrowing algorithmic ideas from QRust is fine; it's actually how this game is meant to be played.

That said, there are two things worth being clean about:

### 3a. Attribution

Even though it isn't required by the rules, doing it makes the submission honest and helps reviewers understand the lineage. Our approach:

- The repo already has `references/qrust-luanmonteiro/` as a sibling of our code, kept under its own MIT license — so the provenance is visible.
- `docs/lectures/02-algorithm-and-data-structures.md` walks through the algorithm and is up-front that QRust is the source of the **algorithmic** ideas (int16 quantization, 32-partition split via the 5-bit key on dims 9/10/11 + sentinel bits, percentile-binned grid V2, AABB lower-bound cell pruning, early-exit squared-distance kernel, fixed-size top-5 cascade).
- The Go code is our own implementation, not a translation — different memory layout (single contiguous `[]int16` for vectors), different concurrency model (UDS + stdlib `net/http` + 1 GOMAXPROCS per replica vs. QRust's Bun event loop), different partition build (cycle-sort in place vs. QRust's bucket-then-copy).

If reviewers ask, we point to that lecture and to the references folder. We don't need to make it adversarial — *learning from each other* is the spirit of Rinha.

### 3b. The test-payload rule

Make sure: **we do not look at `references/rinha-official/test/test-data.json` at runtime, ever.** It exists only because it's part of the public Rinha repo we cloned for documentation. The container doesn't have access to it — only `/resources/references.json.gz`, which is the labeled dataset that's allowed to be pre-processed.

If a reviewer pattern-matches "they read test-data.json", we want them to verify our Dockerfile only copies `cmd/` and `internal/` and that the runtime image is `FROM scratch` with no extra data baked in.

## 4. Current local placement

We've benched both QRust (the reference) and our Go entry on the **same** machine (Apple Silicon, Docker Desktop, Rosetta x86 emulation). Same k6 script, same dataset, same 1 CPU / 350 MB total cgroup split.

| Backend | p99 | FP | FN | HTTP err | failure rate | p99_score | detection_score | **final_score** |
|---------|-----|----|----|----------|--------------|-----------|-----------------|----------------|
| QRust (reference, TypeScript/Bun) | 914 ms | — | — | — | — | — | — | **~2752** |
| **Ours (Go, grid V2, LB pruning)** | **4.99 ms** | **0** | **1** | **2** | **0.01%** | **2301.59** | **2656.16** | **4957.75** |

Two important caveats on those numbers:

1. **The test environment is *not* the same as Rinha's.** Rinha runs on a **Mac Mini Late 2014** (2.6 GHz Haswell, 8 GB RAM, Ubuntu 24.04, native x86_64). Our laptop bench is `linux/amd64` containers running under **Rosetta** on Apple Silicon. Rosetta translates x86 instructions on the fly and is roughly an order of magnitude slower for CPU-bound code than native amd64 — and *much* slower for some patterns (Go's scheduler does fine, but anything JIT-like suffers). The QRust 914 ms p99 on our laptop is almost entirely Rosetta overhead; on a native Haswell it's far better.

2. **k6 from the same host competes for CPU with the SUT.** The Mac Mini is presumably a dedicated test rig with k6 running on the same box, but the contention shape is different from our laptop where Docker Desktop's VM, k6, the SUT containers, and Claude's Bash subprocesses are all sharing cores. So absolute numbers do not transfer — we should think in *order-of-magnitude*: "Go is ~180× faster than QRust *on our local rig* in p99 terms."

In other words: **the local final_score of 4957 is encouraging — it shows the algorithm is right and the implementation is clean — but the number we actually care about is what the Rinha Engine reports on a Haswell with no Rosetta tax.** Preview tests are the only way to find out.

## 5. Ranking — manual or automatic?

**Automatic.** The Rinha Engine is a bot. Once we open the PR adding `participants/muanlartins.json`, we trust the organizer to merge it. Once it's merged, we open a `rinha/test` issue, and the engine runs the bench and posts the score. There's no human in the loop on scoring.

The PR itself is reviewed (it's a regular GitHub PR; there are 142 participants merged so far, so the maintainer is steadily merging). It's a one-person project run by `zanfranceschi`; the FAQ says *"mistakes are expected"* — so we should make the PR easy to merge: small, single file added in `participants/`, no other repo changes, checklist boxes ticked.

There is **no public "leaderboard" in the repo right now** during preview season. Preview test results go into the GitHub issue comments. The final ranking is published after the final test.

## 6. Our concrete next steps

1. **One more round of optimization** — request-path JSON parser, GOMEMLIMIT/GOGC, hot-path alloc audit. (Today.)
2. **Re-bench locally** to make sure we didn't regress.
3. **Build & push `muanlartins/rinha-de-backend-2026:latest` to Docker Hub** for `linux/amd64`. Public image.
4. **Fork `zanfranceschi/rinha-de-backend-2026`** to `muanlartins/rinha-de-backend-2026-official` (just for the PR).
5. **Add `participants/muanlartins.json`** on a branch in the fork, open the PR.
6. **Wait for merge.** Once merged, open the `rinha/test muanlartins-go` issue.
7. **Read the result comment**, decide whether to iterate (we can re-test as many times as we want during preview).
8. **Hold position for the final test.**

## 7. What the result comment looks like

From the official EVALUATION.md, here's the JSON we'll get back:

```json
{
  "expected": { "total": 5000, "fraud_count": 1750, "fraud_rate": 35 },
  "p99": "5.81ms",
  "scoring": {
    "breakdown": {
      "true_positive_detections":  1735,
      "true_negative_detections":  3210,
      "false_positive_detections":   40,
      "false_negative_detections":   15,
      "http_errors":                  0
    },
    "failure_rate": "1.10%",
    "weighted_errors_E": 85,
    "error_rate_epsilon": 0.017,
    "p99_score":     { "value": 2235.83, "cut_triggered": false },
    "detection_score": { "value": 1189.20, "rate_component": 1769.55, "absolute_penalty": -580.35, "cut_triggered": false },
    "final_score": 3425.03
  }
}
```

Both `p99_score` and `detection_score` are bounded `[-3000, +3000]`. The two cutoffs we want to stay away from:

- `p99 > 2000ms` → `p99_score = -3000`. We're three orders of magnitude away from this.
- `failure_rate > 15%` → `detection_score = -3000`. We're at 0.01% locally.

The thing to optimize next is the *gap between current scores and the +3000 ceilings*:

- `p99_score`: we have 2301 / 3000. Each 10× speedup adds 1000 points. Going from 5 ms → 0.5 ms = +1000. From 0.5 ms → 0.05 ms = +1000 (capped at 3000).
- `detection_score`: we have 2656 / 3000. We lose 343 in the absolute penalty for the 13 weighted errors (1 FN × 3 + 2 Err × 5 = 13). The HTTP errors are the big chunk; fixing parse paths to never 5xx would cut them out.

So the highest-leverage improvements after this round, ordered by potential:
1. Hot-path parser that never fails (eliminates the 2 HTTP errors → +50–100 detection points).
2. Anything that cuts p99 below 1 ms.
3. Better detection — we'd need a structural algorithm change (different K, distance metric, anti-overfitting on edge cases) to reduce the 1 FN, and at one error it's not worth touching.
