# Rinha de Backend 2026 — Go submission

A Go submission to [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026): fraud detection over a 3M-vector reference set via KNN-5 with grid-indexed lower-bound pruning.

## Architecture

```
   client ── :9999 ──► HAProxy ── unix sockets ──► api-1 ─┐
                                                          ├─► dataset.bin (mmap, shared via page cache)
                                                  api-2 ─┘
```

- **HAProxy 3.0-alpine** as round-robin LB (16 MB / 0.10 CPU)
- **Two Go static binaries** as API replicas, listening on Unix Domain Sockets in tmpfs (~120 MB / 0.45 CPU each)
- **Shared dataset** built at first startup from `references.json.gz` into a partitioned int16-quantized binary; mmap'd by both replicas → ~87 MB resident shared via the OS page cache

The algorithm partitions the dataset by 5 bits (`is_online`, `card_present`, `unknown_merchant`, sentinel-5, sentinel-6) into 32 groups, then builds a spatial grid inside each group with axis-aligned bounding boxes per cell. Query-time uses lower-bound pruning over cells plus an early-exit squared-distance kernel — exact KNN-5, no approximation.

Total budget: **1 CPU, 350 MB RAM**.

## Layout

```
cmd/api/main.go              # entry point — load dataset, start HTTP server
internal/api/                # HTTP handlers (/ready, /fraud-score)
internal/vector/             # vectorization, normalization, quantization
internal/dataset/            # references.json.gz → dataset.bin builder + mmap loader
internal/search/             # KNN-5 search (grid V2 + LB pruning + early-exit)
deploy/                      # haproxy.cfg, dockerfile, docker-compose.yml
docs/lectures/               # detailed write-ups of the problem, algorithm, runtime, benchmarks
benchmarks/                  # harness + run script + per-run output
```

## Local development

Prerequisites: Docker, k6, Go 1.25+.

```bash
# build + run locally
docker compose -f deploy/docker-compose.yml up --build

# benchmark against the official k6 test
./benchmarks/run.sh . local-dev
```

## Score targets

The challenge scores on `score_p99 + score_det`, each in `[-3000, +3000]`. Top placement requires **zero detection errors** (so the algorithm is exact, not approximate) and **p99 ≤ 1 ms** for the latency ceiling. See [docs/lectures/01-problem-and-scoring.md](docs/lectures/01-problem-and-scoring.md) for the full scoring breakdown.
