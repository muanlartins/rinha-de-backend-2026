# Lecture 07 — The Load Shedder

> Goal: explain the 1-slot-semaphore-with-3ms-timeout pattern (borrowed from JosineyJr's top-Go submission), the math of when it pays off, and the tuning knobs we exposed via env vars so future runs can sweep without rebuilding.

## 1. What it is

A queue-bypass valve inside `fraudScoreRaw`. Before we parse the body or run the search, we try to acquire a slot from a bounded semaphore. If the semaphore is full and we don't get a slot within a configurable timeout, **we return the default "approved" response immediately** without doing any work.

```
HAProxy ── new request ──► API container ──► sem.Try(timeout=3ms)
                                                │           │
                                       acquired │           │ timeout
                                                ▼           ▼
                                          parse + search   shed → {"approved":true,"fraud_score":0}
                                          release slot
                                          respond
```

The semaphore has N slots — N = 1 in Josiney's repo; we made it tunable via `SHED_SLOTS` env var, default 4. The timeout is `SHED_TIMEOUT_MS`, default 3.

## 2. Why "approve" is the right shed default

The rinha scoring assigns weights to error kinds:

| outcome | weight in E | when it happens |
|---|---:|---|
| TP (correctly denied fraud) | 0 | search → fraud_count ≥ 3, request was fraud |
| TN (correctly approved legit) | 0 | search → fraud_count < 3, request was legit |
| FP (denied a legit) | **1** | search → fraud_count ≥ 3, request was legit |
| FN (approved a fraud) | **3** | search → fraud_count < 3, request was fraud |
| Err (HTTP timeout or 5xx) | **5** | request took > 2001 ms |

A shed request that returns `fraud_score: 0`:
- For a *legit* transaction (~56 % of payloads) → counts as TN, **zero cost**.
- For a *fraud* transaction (~44 % of payloads) → counts as FN, **cost 3**.

vs. letting the request queue and time out at 2001 ms → guaranteed **Err, cost 5**.

So **shed-as-approve dominates timeout-as-Err in expected weight: 0.56·0 + 0.44·3 = 1.32 vs 5**. We pay 1.32 per shed instead of 5 per timeout. That's why the shed body is `fraudResponses[0]`.

> An alternative would be to **shed-as-deny** (return `{"approved":false,"fraud_score":1}`), but for legit transactions that's an FP (cost 1) and for fraud it's a TP (cost 0). Expected: 0.56·1 + 0.44·0 = 0.56 per shed. *That's actually better than shed-as-approve.* We are not using that today because (a) Josiney's implementation chose approve and the survey reflected this, (b) the difference is small (~0.76 weighted-E per shed) and our shed rate should be small enough that the choice doesn't move the score much. **Open follow-up: try shed-as-deny if shed-as-approve has higher detection cost than expected.**

## 3. The math of when it pays off

For each shed-rate `s` (fraction of requests shed), the score delta vs phase 10 looks like:

```
detection delta = − s · 54100 · (0.44 · 3)            ≈ − s · 71,400  weighted E
latency  delta = (p99_score with shedder) − 999
```

The detection delta is **linear in shed rate**. For a typical full-test of 54100 requests:

| shed rate | new FNs | new E | new ε | det_score | det delta |
|---:|---:|---:|---:|---:|---:|
| 0.1 % | 24 | 72 | 1.3e-3 | 2341 | **−373** |
| 0.5 % | 119 | 357 | 6.6e-3 | 1622 | **−1092** |
| 1.0 % | 238 | 715 | 1.3e-2 | 1296 | **−1418** |
| 2.0 % | 476 | 1428 | 2.6e-2 | 968 | **−1746** |

(Assumes `ε > 0.001` so the rate term doesn't saturate.)

The latency win for capping p99 at `T` ms (rather than the original 100 ms):

| capped p99 | p99_score | latency delta |
|---:|---:|---:|
| 50 ms | 1301 | +302 |
| 30 ms | 1523 | +524 |
| 10 ms | 2000 | +1001 |
| 5 ms | 2301 | +1302 |
| 3 ms | 2523 | +1524 |
| 1 ms | 3000 | +2001 |

**Net = latency delta + detection delta.** For the load shedder to be a win, we need a careful operating point. Some scenarios:

- **Tight tuning: shed 0.1 %, cap p99 at 5 ms.** Net = 1302 − 373 = **+929**. Plausible if the shed rarely fires and only on the worst tail spikes.
- **Loose tuning: shed 1 %, cap p99 at 10 ms.** Net = 1001 − 1418 = **−417**. Bad — shed costs more than the latency win.
- **Sweet spot we hope for: shed ~0.05 %, cap p99 at ~3-5 ms.** Net ≈ +1100.

To achieve the sweet spot the slot count must be tuned to absorb normal load and only push back during real bursts. Hence the `SHED_SLOTS` env var.

## 4. Why we made it tunable

`SHED_SLOTS=1` is Josiney's setting. **It does not necessarily match our service time.** Josiney's per-request CPU is ~0.5 ms; at 900 RPS over 2 APIs = 450 RPS each, 1 slot × 1000 ms/0.5 ms = 2000 ops/sec per API, comfortable margin. Slot 1 barely ever blocks.

Our per-request CPU on Mac Mini is probably 2-5 ms (3-5× slower CPU than Josiney's local). At 450 RPS per API: 1 slot × 1000 / 3 = 333 ops/sec → **frequent blocking, high shed rate, possibly catastrophic detection**.

To avoid blind-tuning, we expose:

- `SHED_SLOTS` — N parallel slots per API. Default 4 = capacity 1000 ops/sec/API for 4 ms service time, 1500 RPS total. Margin for our 900 RPS.
- `SHED_TIMEOUT_MS` — max wait. Default 3 ms.

Both are env vars, set in compose. Sweep without rebuilding.

## 5. The implementation

In `internal/api/server.go`:

```go
var (
    shedSem        = make(chan struct{}, shedSlots)
    shedTimeoutDur = time.Duration(shedTimeoutMS) * time.Millisecond
    shedCount      atomic.Uint64
)

func (h *Handler) fraudScoreRaw(body []byte) []byte {
    // ... readiness checks ...

    select {
    case shedSem <- struct{}{}:
        defer func() { <-shedSem }()
    case <-time.After(shedTimeoutDur):
        shedCount.Add(1)
        return rawhttpResponses[0]
    }

    // ... parse + search ...
}
```

Two notes on the implementation:

- **`time.After` allocates a `*time.Timer` per call.** At 900 RPS that's ~27k allocs/sec per API. Measurable GC pressure but not catastrophic. If it shows up in pprof, switch to a pooled `time.Timer`. We haven't yet because the implementation should be correct before it's clever.
- **`shedCount` is exposed via `/debug/info`** so we can read it after a run and know how many requests were shed. Always check this first — if it's 0, the shedder didn't fire (probably tuned too loose), if it's > 2 % of total, the shedder fired too much.

## 6. The compose-level wiring

`deploy/docker-compose.yml`:

```yaml
api-1:
  environment:
    - API_SOCKET=/var/run/rinha/api1.sock
    - SHED_SLOTS=4         # default; sweep this on subsequent runs
    - SHED_TIMEOUT_MS=3    # default
```

Same on api-2. To sweep:
1. Edit env values in compose.
2. `docker compose up -d --force-recreate api-1 api-2`. No image rebuild needed.
3. Run k6, check `/debug/info` for `shed_count`.

## 7. Open questions / follow-up experiments

If the load shedder gives a measurable but not-huge bump:

- Try `SHED_SLOTS=1` and `SHED_SLOTS=2` for stricter blocking. Tighter shedder = lower p99 but more shedding.
- Try `SHED_TIMEOUT_MS=1` to push p99 cap further down at the cost of more shedding.
- Try **shed-as-deny** (return `fraudResponses[5]`) — lower per-shed cost, but flips the FN/FP balance.
- Investigate why our service time is 5-10× slower than Josiney's despite using a similar algorithm and faster language. Profile the Mac Mini run via something like `go tool pprof http://api:port/debug/pprof/profile`.

If the load shedder is a net loss:

- The 100 ms p99 isn't queue wait — it's actual CPU work. Need algorithmic / SIMD optimization.
- Or our shed-as-approve detection cost is much higher than the 0.44 fraud rate suggests (e.g., shed requests happen to be disproportionately fraud).

## 8. The single sentence

> **The load shedder caps tail latency by trading a known-small fraction of misclassifications for a known-large p99 reduction, but only pays off when service time is fast enough that shedding rarely fires.**
