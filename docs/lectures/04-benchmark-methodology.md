# Lecture 04 — Benchmark Methodology

> Goal: a repeatable way to score any submission (ours, QRust, future variants) on this machine, capture enough metrics to *understand* the score, and chart results so improvements are unambiguous.

## 1. What we are measuring, exactly

The official scoring needs only two numbers: **p99 latency** and **counts of TP/TN/FP/FN/Err**. Both come from k6 running the official `test/test.js` script against `http://localhost:9999`.

But to *understand why* a number is what it is, we also want:

| metric | source | tells us |
|---|---|---|
| p50 / p95 / p99 / p999 latency | k6 trend stats | shape of the latency distribution; tail behavior |
| TP/TN/FP/FN/Err counts | k6 custom counters | detection correctness |
| RPS achieved vs target | k6 metrics | did we keep up with the ramp? |
| HTTP error rate | k6 (`http_req_failed`) | connection refused, timeouts |
| Per-container CPU% | `docker stats` JSON | which container is the bottleneck |
| Per-container RSS | `docker stats` JSON | are we close to the memory limit? |
| Per-container page faults / IO wait | `cat /sys/fs/cgroup/.../memory.events`, `iostat` | hidden swap or mmap pressure |
| Go: heap inuse, GC pauses, goroutines | `pprof` on the running container, `/debug/vars` | GC tail latency cause |

We collect all of this into a single `benchmarks/<run-name>/` directory per benchmark run.

## 2. Local environment — what we are NOT measuring

The official test environment is a **Mac Mini Late 2014, 2.6 GHz Haswell, 8 GB, Ubuntu 24.04, linux/amd64**.

Our local dev machines are different. Two implications:

1. **Absolute numbers won't match the official score.** If we get p99 = 800 μs locally on an M-series Mac via Docker Desktop's Linux VM (which actually emulates linux/amd64 via Rosetta or runs Apple Silicon native), the *same code* on the Haswell Mini will likely show p99 = 1.5 ms or higher. The Haswell Mini is much slower than modern hardware.
2. **Relative numbers DO match.** If change X improves local p99 from 800 μs → 700 μs (a 12% improvement), that change will *also* improve the Mac Mini's p99 by approximately 12%. So we benchmark improvements as ratios, not absolute targets, until we get an official test result.

**Cross-checks to do**:
- Confirm our Docker is running linux/amd64 images (not arm64). `docker inspect <container> | jq '.[].Architecture'` should say `amd64`. Apple Silicon users need `--platform=linux/amd64` everywhere and Rosetta-x86 emulation, which is *slower* than native — so our local numbers will be even worse than the Mac Mini's. That's fine for ratios.
- Confirm cgroup limits are actually applied. `docker exec <api-container> cat /sys/fs/cgroup/cpu.max` should show `45000 100000` (45% of one core). And `cat /sys/fs/cgroup/memory.max` should show `167772160` (160 MiB ≈ 167 MB rounded).

## 3. The benchmark protocol — what we'll do every time

```
1. checkout the target branch / SHA
2. docker compose down --remove-orphans
3. docker compose build
4. docker compose up -d
5. wait for GET http://localhost:9999/ready → 200    (timeout 60 s)
6. (optional) smoke test: k6 run test/smoke.js
7. start `docker stats` collection in background → benchmarks/<run>/stats.csv
8. k6 run test/test.js                              → benchmarks/<run>/results.json
9. stop `docker stats`
10. docker compose logs > benchmarks/<run>/logs.txt
11. docker compose down
12. write benchmarks/<run>/manifest.json with git SHA, env, hardware
```

We capture **everything** to disk and never overwrite. A run is uniquely named: `<short-name>-<YYYYMMDD-HHMMSS>-<git-sha>`.

### The `manifest.json` we write per run

```jsonc
{
  "run_name": "qrust-baseline",
  "submission": "references/qrust-luanmonteiro",
  "git_sha": "d5ab304",
  "git_branch": "main",
  "timestamp_utc": "2026-05-11T16:42:00Z",
  "host": {
    "os": "darwin 25.3.0",
    "arch": "arm64",
    "cpu_model": "...",
    "docker_arch": "amd64",
    "docker_emulation": true
  },
  "compose_limits": {
    "lb": { "cpus": "0.10", "memory": "16MB" },
    "api-1": { "cpus": "0.45", "memory": "167MB" },
    "api-2": { "cpus": "0.45", "memory": "167MB" }
  },
  "k6_version": "v0.50.0",
  "test_data_sha": "..."
}
```

Always check this when comparing runs. A change in `compose_limits` invalidates the comparison; a change in `docker_emulation` invalidates everything.

## 4. The k6 invocation

The official script is at `references/rinha-official/test/test.js`. To run it against our local LB:

```bash
cd references/rinha-official
k6 run test/test.js
```

The script writes `test/results.json` in the same directory. Move it into our run folder:

```bash
mv test/results.json /Users/muanlartins/repos/rinha-de-backend-2026/benchmarks/<run>/results.json
```

The shape of `results.json` (from the official EVALUATION.md):

```json
{
  "expected": { "total": 5000, "fraud_count": 1750, ... },
  "p99": "5.81ms",
  "scoring": {
    "breakdown": {
      "true_positive_detections": 1735,
      "true_negative_detections": 3210,
      "false_positive_detections": 40,
      "false_negative_detections": 15,
      "http_errors": 0
    },
    "failure_rate": "1.10%",
    "weighted_errors_E": 85,
    "error_rate_epsilon": 0.017,
    "p99_score": { "value": 2235.83, "cut_triggered": false },
    "detection_score": { "value": 1189.20, ... },
    "final_score": 3425.03
  }
}
```

This is the **score artifact** — what we ultimately optimize. Everything else helps explain it.

### What k6 actually produces internally

In addition to `results.json`, k6 emits its own metrics stream. We capture both:

```bash
k6 run \
    --summary-export=test/results.json \
    --out json=k6-raw.json \
    test/test.js
```

`k6-raw.json` is one JSON object per HTTP request (timestamps, durations, statuses) — useful for charting latency over time. We post-process it with `jq` or a small Go tool into a CSV.

## 5. `docker stats` in JSON

While k6 runs (~120 s ramp + warmup + cooldown ≈ 150 s total), we sample container metrics:

```bash
docker stats \
    --format '{{json .}}' \
    --no-trunc \
    qrust-concept-api-1 qrust-concept-api-2 haproxy-lb \
    > stats.ndjson &
```

This writes a stream of JSON objects (one per snapshot per container, every ~1 s):

```json
{ "Name": "qrust-concept-api-1", "CPUPerc": "44.5%", "MemUsage": "143.2MiB / 167MiB", "BlockIO": "0B / 0B", "PIDs": "13" }
```

We post-process into a CSV: `time, container, cpu_percent, mem_mb`. Chart over the 150 s window to see where the API saturates CPU and where the LB starts queueing.

## 6. The smoke test — do this first, every run

`references/rinha-official/test/smoke.js` sends 5 requests with a known payload, sequentially. **Fast, cheap, catches the obvious bugs before the load test.** Always run it before kicking off `test.js`. If smoke fails:

- `status is 200`: did /ready actually return 200? Is the api container's `dataset.bin` built?
- `body is json`: did we send a response body? Truncation? Wrong content-type?
- `approved is boolean`: JSON shape error?
- `fraud_score is number`: ditto?

In our `benchmarks/run.sh`, smoke is a hard gate before `test.js`. No smoke pass → no load test.

## 7. Reproducibility — the things that bite

Three sources of run-to-run variance that we control:

### 7a. JIT / runtime warmup

Bun's JIT compiles hot functions lazily over the first ~1000 calls. The k6 ramp starts at 1 RPS, so the first second of traffic is **cold JIT**. The official scoring averages over the whole test, so cold-start latency mixes into p99. We can't change this for the official test, but for our local runs we can do a **warmup pass**: send 1000 requests through with a separate k6 invocation, then run the real test. The numbers will be 10–20% better. **Don't compare warmed-up local runs to the official test.** When we publish numbers, use a cold start.

Go has no JIT, but it has cgo init, scheduler warmup, and TLB-cold memory. Same principle: cold-start numbers are higher.

### 7b. Page cache cold vs warm

The dataset file is 87 MB int16, mmap'd. On the first scan of a partition, the OS faults pages in from disk — slow. After 30 s of traffic, the working set is hot in page cache. The k6 ramp builds up over 120 s, so this effect is **inside** the test window. The official test's p99 includes a few cold-page misses.

To reproduce: between `docker compose up` and `k6 run`, do nothing — let the test be honest about cold pages. Don't preload with a warmup unless we're benchmarking the warm regime specifically.

### 7c. Background load

CPU shared with browser, IDE, sync tools, virus scanner. The Mac Mini in the test env is dedicated — no background load. Our laptop is not. **Close everything** before running. Or run on a dedicated VM. Or run multiple times and take the median.

## 8. Charting and comparing runs

We'll build a tiny tool in `benchmarks/_tools/` (Go, single binary, ~100 lines) that:

1. Walks `benchmarks/*/results.json`.
2. Outputs a table comparing all runs across columns: `run name | p50 | p99 | p999 | TP | TN | FP | FN | Err | final_score`.
3. Outputs latency CDFs as ASCII or SVG for visual comparison.

Something like:

```
                     │   p50 │   p99 │  p999 │   FP │   FN │  Err │ score
─────────────────────┼───────┼───────┼───────┼──────┼──────┼──────┼──────
qrust-baseline       │ 167μs │ 855μs │ 2.4ms │   0  │   0  │   0  │ 5450
qrust-haproxy-tune-2 │ 159μs │ 812μs │ 2.1ms │   0  │   0  │   0  │ 5560
go-scaffold-v1       │ 412μs │ 4.1ms │ 8.9ms │   0  │   0  │   0  │ 4395
go-scaffold-v2-int16 │ 198μs │ 1.4ms │ 3.2ms │   0  │   0  │   0  │ 5311
go-grid-v2           │ 134μs │ 720μs │ 1.9ms │   0  │   0  │   0  │ 5557
```

## 9. The `benchmarks/run.sh` script — what we actually invoke

The user-facing command we'll build (proposed):

```bash
# from /Users/muanlartins/repos/rinha-de-backend-2026
./benchmarks/run.sh <submission-dir> <run-name>
# e.g.
./benchmarks/run.sh references/qrust-luanmonteiro qrust-baseline
./benchmarks/run.sh submission go-scaffold-v1
```

Does, in order:

1. Resolves `<submission-dir>` to either a path to a folder containing `docker-compose.yml` and `info.json`, or a git URL.
2. `docker compose -f <submission-dir>/docker-compose.yml up -d --build`.
3. Polls `GET http://localhost:9999/ready` until 200 or 60 s timeout.
4. Runs `k6 run references/rinha-official/test/smoke.js` — bails if it fails.
5. Starts `docker stats` collector to background.
6. Runs `k6 run --summary-export=… references/rinha-official/test/test.js`.
7. Stops collectors.
8. Writes `manifest.json` with git SHA, host info, compose limits.
9. `docker compose down`.
10. Writes a summary line to stdout: `<run-name>: p99=Xms, score=Y, FP=A FN=B`.

## 10. Final notes

The benchmark is the source of truth. Anytime we make a code or config change, we run the benchmark before claiming an improvement. Hand-wavy "this should be faster" doesn't count. We only believe a delta if the numbers agree.

Two things we *don't* need to build yet:
- **A continuous benchmark service.** Overkill for one-off comparisons. We re-run manually.
- **Real-time dashboards.** Static `results.json` + a comparison table is enough.

What we *do* need before writing Go:
- The bench harness runs end-to-end against QRust and produces a `manifest.json` + `results.json`. That's our **baseline number**.
- Whenever we change anything in the Go submission (algorithm, JSON parser, image base), we rerun the bench and append a new run dir. Five runs in, we can already see which changes were worth it.

## 11. Where this lecture ends

We now have:
- A clear list of metrics to capture per run.
- A protocol for running k6 + `docker stats` + manifest.
- Awareness of the local vs Mac-Mini gap (ratios, not absolutes).
- A plan for a small comparison tool.

Next concrete step: build `benchmarks/run.sh` and run it against QRust to capture our **baseline number**. Then we know what we're aiming to match (or beat) when the Go port comes online.
