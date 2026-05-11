# Lecture 01 — The Problem & The Scoring Game

> Audience: us. Goal: internalize the rules and the scoring shape **before** writing a single line of Go, so every implementation decision later is measurable against the right target.

## 1. The challenge in one sentence

For each incoming card transaction:

1. Transform the JSON payload into a **14-dimensional vector**.
2. Find the **5 nearest neighbors** (Euclidean distance) inside a **fixed 3,000,000-vector reference dataset**.
3. Compute `fraud_score = frauds_in_5 / 5`.
4. Respond `{ approved: fraud_score < 0.6, fraud_score }`.

Stateless. Deterministic. No database needed. No state to coordinate across instances. **The whole game is one synchronous in-memory function wrapped in HTTP.**

```
   client ──── POST /fraud-score ────► load balancer ──RR──► api-N ──┐
                                                                     │
                                                                     ▼
                                            ┌─────────────────────────────────────┐
                                            │ 1. parse JSON                       │
                                            │ 2. vectorize (14 dims, normalize)   │
                                            │ 3. KNN-5 over 3M refs               │
                                            │ 4. fraud_score = frauds / 5         │
                                            │ 5. JSON encode { approved, score }  │
                                            └─────────────────────────────────────┘
```

## 2. The contract

**Endpoints (port 9999, exposed by the LB)**

| method + path | purpose |
|---|---|
| `GET /ready` | 2xx when the dataset is loaded and the API can serve `/fraud-score` |
| `POST /fraud-score` | the actual work — payload below |

**Request body**

```jsonc
{
  "id": "tx-1329056812",
  "transaction": {
    "amount": 41.12,                    // number
    "installments": 2,                  // int
    "requested_at": "2026-03-11T18:45:53Z"  // ISO-8601 UTC
  },
  "customer": {
    "avg_amount": 82.24,                // historical avg ticket
    "tx_count_24h": 3,
    "known_merchants": ["MERC-003", "MERC-016"]  // duplicates possible
  },
  "merchant": {
    "id": "MERC-016",
    "mcc": "5411",                      // category code (string)
    "avg_amount": 60.25
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 29.23
  },
  "last_transaction": null              // OR { timestamp, km_from_current }
}
```

**Response body** — exactly:

```json
{ "approved": true, "fraud_score": 0.0 }
```

## 3. The 14 dimensions — three families that drive the algorithm

| idx | dim | formula | range | family |
|---:|---|---|---|---|
| 0 | `amount` | `clamp(amount / 10000)` | `[0,1]` | continuous |
| 1 | `installments` | `clamp(installments / 12)` | `[0,1]` | continuous (12 buckets) |
| 2 | `amount_vs_avg` | `clamp((amount / customer.avg_amount) / 10)` | `[0,1]` | continuous |
| 3 | `hour_of_day` | `hour(requested_at) / 23` | `[0,1]` | continuous (24 buckets) |
| 4 | `day_of_week` | `dow(requested_at) / 6`  (mon=0, sun=6) | `[0,1]` | continuous (7 buckets) |
| 5 | `minutes_since_last_tx` | `clamp(min / 1440)` **or `-1`** if `last_transaction == null` | `[0,1] ∪ {-1}` | sentinel |
| 6 | `km_from_last_tx` | `clamp(last_transaction.km_from_current / 1000)` **or `-1`** | `[0,1] ∪ {-1}` | sentinel |
| 7 | `km_from_home` | `clamp(km_from_home / 1000)` | `[0,1]` | continuous |
| 8 | `tx_count_24h` | `clamp(tx_count_24h / 20)` | `[0,1]` | continuous (21 buckets) |
| 9 | `is_online` | `1 if true else 0` | `{0, 1}` | **bit** |
| 10 | `card_present` | `1 if true else 0` | `{0, 1}` | **bit** |
| 11 | `unknown_merchant` | `1 if merchant.id ∉ known_merchants else 0` (inverted!) | `{0, 1}` | **bit** |
| 12 | `mcc_risk` | `mcc_risk.json[mcc] or 0.5` | `[0,1]` | look-up |
| 13 | `merchant_avg_amount` | `clamp(merchant.avg_amount / 10000)` | `[0,1]` | continuous |

**Three insights that will shape every implementation choice we make:**

1. **Bits 9, 10, 11 split the space.** If two transactions differ on `is_online`, the squared-distance contribution from that one dimension alone is `1.0` — gigantic relative to the rest, which are in `[0,1]` and usually clustered. So a vector with `is_online=true` will essentially never have a true neighbor with `is_online=false`. We can **physically partition** the dataset by these 3 bits → 8 buckets. Different bit = different bucket = never compared.

2. **The `-1` sentinels in dims 5 and 6 do the same.** A "no previous transaction" vector lives at coordinate `(-1, -1)` for those two dims; any vector with real history lives in `[0,1]²`. The squared distance between them is at least `1 + 1 = 2`. So we partition on those two bits too (`is_5_sentinel`, `is_6_sentinel`).

3. **5 bits → 32 partitions.** Inside each partition, those 3 (or 5, if sentinel) dimensions are *constant*, so the distance kernel can skip them entirely. That removes a few multiplies and — more importantly — lets the lower-bound calculation focus on the *variable* dimensions only.

```
Partition key (5 bits, 0..31):

  bit 0:  is_online           (dim 9)
  bit 1:  card_present        (dim 10)
  bit 2:  unknown_merchant    (dim 11)
  bit 3:  last_tx_5_is_null   (dim 5  == -1)
  bit 4:  last_tx_6_is_null   (dim 6  == -1)

Both query and references compute the same key; we only ever scan
the references whose key matches.
```

## 4. The reference dataset

| file | shape | use |
|---|---|---|
| `references.json.gz` | array of `{ "vector": [14 floats], "label": "fraud"\|"legit" }`, **3,000,000 rows** | the KNN haystack |
| `mcc_risk.json` | `{ "<mcc-code>": <float 0–1>, ... }`, 10 entries; default `0.5` | drives dim 12 |
| `normalization.json` | the 7 constants in the formulas above | drives normalization |

**It does not change during the test.** Pre-process it however you want — decompress, quantize, sort, build indexes, anything — **at container build or startup**. None of that processing counts against your request latency.

**Memory math, the constraint that designs the system:**

| representation | bytes per vector | total for 3M | room left of 350 MB |
|---|---:|---:|---:|
| `float64` (Go default) | 112 | 320 MB | ~10 MB — infeasible |
| `float32` | 56 | 160 MB | ~190 MB |
| `int16` (linear quantize 0..32000) | 28 | 80 MB | ~270 MB |
| `int8` (linear quantize -127..127) | 14 | 40 MB | ~310 MB |

We have to fit **two** API replicas' worth (or share the data — more on that in Lecture 03). Int16 is the comfortable middle: matches a multiply within signed 32-bit accumulators without overflow on 14 dims, AABBs stay cheap, and we keep ~3× headroom for buffers/runtime/LB. Int8 buys 2× more headroom but tightens accuracy margins on a problem that demands zero detection errors. Default to **int16** for the Go port; revisit int8 only if memory pressure shows up.

## 5. The scoring formula — and what it actually rewards

```
final_score = score_p99 + score_det          ∈ [-6000, +6000]

# latency
if p99 > 2000ms:
    score_p99 = -3000                        # hard cutoff
else:
    score_p99 = 1000 · log10(1000 / max(p99_ms, 1))   # saturates at +3000 when p99 ≤ 1ms

# detection
N            = TP + TN + FP + FN + Err
E            = 1·FP + 3·FN + 5·Err           # weighted error count
ε            = E / N
failure_rate = (FP + FN + Err) / N

if failure_rate > 0.15:
    score_det = -3000                        # hard cutoff
else:
    score_det = 1000 · log10(1/max(ε, 0.001)) − 300 · log10(1 + E)
```

### Score tables — read these like a chart

**p99 → score_p99** (all-or-nothing on either side of the 1ms / 2000ms walls)

| p99 | score_p99 |
|---:|---:|
| ≤ 1.00 ms | **3000** (saturated) |
| 1.17 ms | 2932 |
| 1.50 ms | 2824 |
| 2.00 ms | 2699 |
| 3.00 ms | 2523 |
| 5.00 ms | 2301 |
| 10.00 ms | 2000 |
| 100.00 ms | 1000 |
| > 2000 ms | -3000 |

**Detection errors → score_det** (assuming N = 5000, failure_rate ≤ 15%)

| FP | FN | Err | E | ε | score_det |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | **3000** (saturated) |
| 3 | 0 | 0 | 3 | 0.0006 | 2819 |
| 5 | 0 | 0 | 5 | 0.001 | 2767 |
| 0 | 1 | 0 | 3 | 0.0006 | 2819 |
| 0 | 0 | 1 | 5 | 0.001 | 2767 |
| 10 | 5 | 0 | 25 | 0.005 | 1877 |
| 0 | 0 | 750 | 3750 | 0.75 | -3000 (15% rate cutoff triggers first) |

### The strategic implication

Every top-10 entry on the leaderboard (snapshot from QRust's research notes, 4 May 2026) has **`score_det = 3000`** — zero detection errors. So:

```
     +-------------------------------------------------------+
     |  ABOVE THE CUT (score_det = 3000):                    |
     |     final_score is a pure function of p99             |
     |                                                       |
     |     1.0 ms → 6000     1.5 ms → 5824                   |
     |     2.0 ms → 5699     3.0 ms → 5523                   |
     +-------------------------------------------------------+
     |  BELOW THE CUT (any detection error):                 |
     |     you can have p99 = 1.0 ms and still lose          |
     |     a single FN costs you ~181 points                 |
     +-------------------------------------------------------+
```

The competition collapses to **two binary decisions and one continuous fight**:

1. Did you cross the **detection** cliff? (target: zero errors)
2. Did you cross the **failure-rate** cliff? (target: well under 15% — easy once #1 is satisfied)
3. Then it's all about **p99 latency**.

This is why every serious implementation uses an *exact* search algorithm (not approximate ANN). The grid + lower-bound pruning approach is exact: a pruned cell can be proven to contain no top-5 candidate, so skipping it doesn't change the answer. Approximate ANN would gain speed at the cost of a few wrong neighbors, which is *catastrophic* under this scoring.

## 6. The load test

From `references/rinha-official/test/test.js`:

- k6 `ramping-arrival-rate` scenario, **1 → 900 RPS over 120 s**.
- Pre-allocated VUs: 100, max VUs: 250.
- Per-request HTTP timeout: 2001 ms (so anything taking longer is counted as an Err).
- Each request gets a payload from `test/test-data.json` (~27 MB) with an `expected_approved` label.
- p99 is measured over the entire `http_req_duration` distribution; the response classification gives us TP/TN/FP/FN/Err.

The scenario is interesting because it includes a *cold-start tail* (first few seconds at low RPS, when JIT or page cache haven't fully warmed) **and** a *steady-state tail* (mid-to-late, at peak RPS). Our p99 will live somewhere in those last 30 seconds. The first second's warmup spike isn't usually the dominant tail; the steady-state queueing under contention is.

## 7. The hardware budget — the constraint that designs the system

| resource | total budget | typical split | per-replica meaning |
|---|---:|---|---|
| CPU | **1.000 core** | 0.10 LB / 0.45 api1 / 0.45 api2 | at 900 RPS, ~1.0 ms of CPU per request *if* the split is perfectly utilized |
| RAM | **350 MB** | 16 / 167 / 167 | dataset is replicated per replica (no Linux shared-memory tricks in QRust's setup) |
| Network | **bridge only** | n/a | `host` and `privileged` forbidden; bypassed via Unix Domain Sockets in tmpfs |
| Hardware | Mac Mini Late 2014 | Haswell, 2.6 GHz | AVX2 yes, AVX-512 no; modest single-thread; 8 GB host RAM |

Two pieces of arithmetic to internalize:

- **CPU budget per request, 900 RPS, 2 replicas × 0.45 cores**: `0.9 cores / 900 RPS = 1.0 ms` of CPU per request. Visiting ~180k vectors with a ~10 ns/visit kernel ≈ 1.8 ms of CPU. Already over budget on average — the system only works because (a) most queries hit smaller partitions and (b) there's idle slack between requests at lower RPS levels of the ramp.
- **Memory per replica, int16 quantization**: 3M × 14 × 2 B = 80 MB raw + grid metadata (~5 MB) + runtime overhead. Bun lands at ~167 MB. Go should be tighter — `~120–130 MB` is achievable if we control allocations.

## 8. The Go-specific takeaways from this lecture

These are the constraints from the rules and scoring that we will reach for again and again while writing the Go port:

1. **Zero allocation on the hot path.** Every `POST /fraud-score` must reuse buffers. No `json.Unmarshal` into structs that allocate maps, no `[]float64` query slice per request. Pre-allocate a per-goroutine query buffer (`sync.Pool` or per-handler-instance state) and a small response buffer.
2. **Int16 quantization in flat arrays.** Use a single `[]int16` of length `3_000_000 * 14` with explicit indexing (`vectors[i*14 + d]`). Avoid the `[][]int16` jagged-slice layout — that's an extra indirection per access and an enormous allocation count at startup.
3. **No reflection-based JSON.** `encoding/json` allocates and uses reflection. Either hand-roll a fixed-shape parser (the payload schema is known and small), use `github.com/valyala/fastjson` for streaming, or — since the response is one of six pre-computed strings — write the response with `w.Write` of a `[]byte` constant.
4. **Two `runtime` knobs to set early**: `GOGC=off` (or very high) once the dataset is loaded — GC sweeps over the multi-MB dataset are pure waste because it never becomes garbage; `GOMEMLIMIT=160MiB` (per replica) gives the runtime a hard ceiling matching the cgroup limit so the GC doesn't oscillate at the boundary.
5. **HTTP server**: stdlib `net/http` is fine for our payload size, but `fasthttp` can shave another few hundred microseconds via zero-copy request parsing. Worth benchmarking when the algorithm is otherwise stable.
6. **Listen on a Unix Domain Socket**, not a TCP port. Set `Listener` to `net.Listen("unix", "/var/run/rinha/api1.sock")`. HAProxy in the LB connects via `unix@/var/run/rinha/api1.sock`. tmpfs volume mounted at `/var/run/rinha`.

## 9. Where this lecture ends and the next begins

We now know:
- Exactly what the API has to do, and the contract.
- The shape of the 14 dims and the natural 32-partition split.
- The scoring math and the strategic implication (zero-error + minimize p99).
- The memory and CPU budget arithmetic.
- A list of Go-flavored implementation principles.

What we have **not** yet covered:
- *How* the 5 nearest neighbors are found efficiently — the grid V2 algorithm, lower-bound pruning, early-exit distance, scratch-buffer reuse.
- *Why* int16 is the right quantization width, and how to do it without losing detection accuracy.
- The infra plumbing: HAProxy config, UDS over tmpfs, cgroups split rationale.
- How to actually run the benchmark and chart results.

Those are Lectures 02, 03, and 04 respectively.
