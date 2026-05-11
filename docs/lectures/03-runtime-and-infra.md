# Lecture 03 — Runtime, IPC & Infrastructure

> Goal: understand the layer above the algorithm — JSON parsing, HTTP server, IPC between LB and APIs, container resource splits — and the Go-specific choices we need to make for each.

## 1. The per-request budget, broken down

If we want p99 ≤ 1.0 ms (= +3000 score), we have to land *every* request in that budget — not just the median. Reference numbers measured on QRust (Bun/TS, Mac Mini Late 2014, the official hardware):

| stage | reference (QRust) | what does it |
|---|---:|---|
| client → LB TCP accept | ~30 μs | HAProxy on port 9999 |
| LB → API (Unix Domain Socket) | ~20 μs | `http-reuse always` keeps a pool, near-zero per-req |
| HTTP request read + parse | ~30 μs | Bun.serve, native HTTP parser |
| JSON parse + vectorize (fused) | ~80 μs | hand-rolled `fastVectorizeAndQuantize` |
| Quantize float → int16 | (inside above) | merged into the parse pass |
| **KNN search** | **~650 μs** | grid-V2 + LB pruning + early-exit |
| Build response (prebuilt strings × 6) | <5 μs | one of 6 cached strings, no JSON encode |
| HTTP response write | ~30 μs | `Response` constructor + send |
| API → LB | ~10 μs | UDS keep-alive |
| LB → client | ~10 μs | TCP write |
| **TOTAL** | **~865 μs** | matches QRust's reported local p99 |

Three observations:

1. **The algorithm dominates.** Even with all the JSON / HTTP optimizations, 75% of the budget goes to the inner loop. So we can't compete by being clever above the algorithm — we have to either be faster *at* the algorithm, or shave the non-algo budget so we can absorb tail spikes in the algo time.
2. **The non-algo overhead is ~200 μs, dominated by JSON parsing and HTTP framing.** That's where Go runtime choices can save us another 50–100 μs vs Bun.
3. **Cache lookup is hash + memcmp of body bytes.** When duplicate requests arrive (rare in the official k6 test, but happens), it returns a cached response string and skips everything above. Probably not load-bearing in our case but cheap insurance.

## 2. The HTTP hot path in Go — three concrete choices

### 2a. Listener: TCP loopback vs Unix Domain Socket in tmpfs

QRust uses **Unix Domain Sockets**, with the socket file living in a **tmpfs**-mounted volume shared between LB and API containers:

```yaml
volumes:
  socket-dir: {}              # docker-compose default tmpfs-ish; for explicit tmpfs use driver_opts

services:
  api-1:
    environment:
      - API_SOCKET=/var/run/rinha/api1.sock
    volumes:
      - socket-dir:/var/run/rinha
  api-2:
    environment:
      - API_SOCKET=/var/run/rinha/api2.sock
    volumes:
      - socket-dir:/var/run/rinha
  lb:
    volumes:
      - socket-dir:/var/run/rinha
```

Why UDS over TCP loopback for the LB ↔ API hop:

| dimension | TCP `127.0.0.1` | UDS `/var/run/rinha/api1.sock` |
|---|---|---|
| Path through kernel | full TCP stack (handshake, congestion ctrl, framing, checksum) | direct in-memory copy between sockets |
| Latency per hop | 80–200 μs | 5–20 μs |
| Connection setup | full 3-way handshake (~50 μs) | path lookup + open |
| Reuse | needs `http-reuse always` or `http-keep-alive` | same, easier to keep working |
| Bridge mode | yes, but adds NAT/conntrack | volume mount, no network stack involved |

**In Go**:

```go
ln, err := net.Listen("unix", os.Getenv("API_SOCKET"))
if err != nil { log.Fatal(err) }
if err := os.Chmod(os.Getenv("API_SOCKET"), 0o666); err != nil {
    log.Fatal(err)
}
defer ln.Close()
srv := &http.Server{Handler: handler}
srv.Serve(ln)
```

The `chmod 0666` is non-obvious: HAProxy and the API run as *different uids* in their respective containers (HAProxy uses uid 99 by default, Bun uses uid 1000). The socket file inherits the API container's owner; HAProxy can't connect unless world-writable. Set `0o666` (or align uids via Dockerfile `USER`).

Also: `os.Remove(socketPath)` before `net.Listen`, because if a previous instance left the file, `Listen` will fail with "address already in use".

### 2b. HTTP server: `net/http` vs `fasthttp`

| dimension | `net/http` (stdlib) | `fasthttp` |
|---|---|---|
| Allocations per request | ~3–5 (Request, Header map, body buffer) | 0 (recycles RequestCtx) |
| Per-request overhead | 30–60 μs | 10–25 μs |
| HTTP/1.1 keep-alive | yes (default) | yes |
| HTTP/2 | yes | **no** (not relevant here — HAProxy → API is HTTP/1.1) |
| API ergonomics | familiar `http.HandlerFunc(w, r)` | `func(ctx *fasthttp.RequestCtx)` |
| Streaming | yes | yes, but different model |
| Connection pooling on client side | needs `http.Transport` care | built-in `fasthttp.Client` |
| Mature middleware ecosystem | yes | smaller |
| Testing surface | trivial with `httptest` | needs a `RequestCtx` builder |

For this challenge, every request is ~500 bytes in and ~50 bytes out, no streaming, no HTTP/2. **fasthttp's allocation profile is the right shape.** The 30–40 μs savings are real and consistent. Trade-off: less idiomatic Go.

**Recommendation**: Start with `net/http` for the scaffold (faster to write, easy to debug, can swap in benchmarks). Switch to `fasthttp` once the algorithm + JSON parser are working and we're chasing the last 100 μs. Behind the same `Handle(req []byte) (resp []byte, err error)` interface, swapping is a 20-line change.

### 2c. JSON parsing: how far down to go

Stdlib `encoding/json` is slow (~10–20 μs for a payload of this size + reflection-based allocation per field). For sub-ms p99 with consistent zero allocation, we need better:

| approach | per-call overhead | allocations | dev cost | notes |
|---|---:|---:|---|---|
| `encoding/json` | 15–30 μs | 8–15 | trivial | reflective; allocates a `map[string]any` shadow tree |
| `github.com/goccy/go-json` | 5–10 μs | 2–4 | drop-in replacement | mostly faster `encoding/json` with codegen |
| `github.com/valyala/fastjson` | 3–6 μs | 0 (uses internal arena) | one-API change | streaming, no struct binding |
| `github.com/buger/jsonparser` | 2–4 μs | 0 | very different API | direct byte-level access by JSON path |
| Hand-rolled fused parse + vectorize | 1–3 μs | 0 | high | QRust's `fast-json.ts` pattern; skip directly between known keys |

QRust uses the last one — and it's measurably the right call for this hot path. The trick: the JSON payload has **fixed key names** in **almost-fixed order**, and we only need 12 numeric fields + 2 booleans + 1 MCC string + 1 boolean derived from a string-membership test. We don't need a generic JSON parser. We need a **lexer that knows our schema**.

**Concrete pattern in Go** (sketch — full implementation in scaffold step):

```go
// Precomputed key needles
var (
    needleAmount       = []byte(`"amount":`)
    needleInstallments = []byte(`"installments":`)
    needleRequestedAt  = []byte(`"requested_at":"`)
    // ... 14 keys total
)

// pos = current scan position in the body
// out = caller-provided [14]int16 (zero alloc)
func ParseAndVectorize(body []byte, out *[14]int16) error {
    var pos int
    // amount
    if pos = bytes.Index(body[pos:], needleAmount); pos < 0 { return errFormat }
    amount, n := parseFloat(body[pos+len(needleAmount):])
    out[0] = quantize(amount / 10000)
    pos += len(needleAmount) + n
    // ... same pattern for all 14 dims
    return nil
}

func quantize(f float64) int16 {
    if f < 0 { return 0 }
    if f > 1 { return 32000 }
    return int16(f*32000 + 0.5)
}
```

A few Go-specific details:
- Use `bytes.Index` — it's hand-optimized assembly for short patterns. Don't write your own loop.
- For known-position fields (after parsing one, the next key is offset N bytes), you can skip the `bytes.Index` call entirely. ISO-8601 timestamps are exactly 20 bytes after the opening quote, so `pos += 20` works.
- For the MCC string → numeric conversion, do it as 4 byte subtractions (`(body[p]-'0')*1000 + ...`), not `strconv.Atoi`.
- For the `is_online` / `card_present` booleans, look at the first byte after the key: `t` (116) = true, `f` (102) = false. No need to parse `"true"` or `"false"` fully.
- For the `known_merchants` array vs `merchant.id` comparison (which becomes dim 11 = `unknown_merchant`), scan the array for the exact byte sequence — no string allocation.

## 3. The response side — the 6-string trick

The response body is **fully determined** by `fraud_count ∈ {0, 1, 2, 3, 4, 5}`:

```
0 → {"approved":true,"fraud_score":0}
1 → {"approved":true,"fraud_score":0.2}
2 → {"approved":true,"fraud_score":0.4}
3 → {"approved":false,"fraud_score":0.6}
4 → {"approved":false,"fraud_score":0.8}
5 → {"approved":false,"fraud_score":1}
```

Pre-compute these 6 strings at startup, store as `[6][]byte`, and on response do `w.Write(responses[fraudCount])`. **Zero JSON encoding at request time.** Saves 5–15 μs and zero allocations.

```go
var responses = [6][]byte{
    []byte(`{"approved":true,"fraud_score":0}`),
    []byte(`{"approved":true,"fraud_score":0.2}`),
    []byte(`{"approved":true,"fraud_score":0.4}`),
    []byte(`{"approved":false,"fraud_score":0.6}`),
    []byte(`{"approved":false,"fraud_score":0.8}`),
    []byte(`{"approved":false,"fraud_score":1}`),
}
```

Be mindful of the response format that the test expects. The k6 script only looks at the `approved` field for classification, but the docs require `fraud_score` to be a number. We match exactly.

## 4. The body-response cache — when does it help?

QRust includes an FNV-1a-hashed `BodyResponseCache` keyed on the request body bytes. If the same payload comes in twice, it skips everything (parse, vectorize, search) and returns the cached response string.

**Will it help on the official test?** Looking at `references/rinha-official/test/test.js`:

```js
export default function () {
    const idx = exec.scenario.iterationInTest;
    if (idx >= testData.length) return;
    const entry = testData[idx];
    // ...
    http.post('http://localhost:9999/fraud-score', JSON.stringify(entry.request), ...);
}
```

Each k6 iteration uses a *different* payload by index. **No duplicates** in the official test. The cache will never hit.

**Should we skip it then?** Two arguments for keeping it:
- The *final* test "uses a different script — likely heavier" — could include repeats.
- The CPU cost when there are no hits is just a 500-byte FNV-1a hash + 0 memcmps — about 1 μs. Trivial.

**Recommendation**: include it in the Go port behind an env flag, off by default for preview tests, on for the final.

## 5. HAProxy configuration

The QRust haproxy.cfg (lightly annotated):

```haproxy
global
    maxconn 4096                          # high enough that we don't queue at LB

defaults
    mode http
    option dontlognull                    # don't log healthcheck connections
    timeout connect 1000ms                # connecting to upstream
    timeout client  5000ms                # idle client
    timeout server  5000ms                # idle server
    timeout http-keep-alive 1000ms        # connection reuse window

frontend fraud_front
    bind *:9999                           # only port exposed to host
    default_backend fraud_back

backend fraud_back
    balance roundrobin                    # spec requires "simple round-robin"
    http-reuse always                     # keep a pool of connections to upstreams
    option httpchk
    http-check send meth GET uri /ready ver HTTP/1.1 hdr Host api-1 hdr Connection close
    http-check expect status 200
    server api1 unix@/var/run/rinha/api1.sock check inter 5000 fall 2 rise 1 maxconn 512
    server api2 unix@/var/run/rinha/api2.sock check inter 5000 fall 2 rise 1 maxconn 512
```

Tuning rationale (from QRust's `DECISIONS.md`):

- **`http-reuse always`**: the LB keeps a pool of open connections to each upstream. Otherwise it would handshake the UDS for every request — adding ~20 μs each time. This setting alone is worth ~15 μs at p99.
- **`balance roundrobin`**: required by spec. Avoid `leastconn` (which interrogates queue depth on each request).
- **`inter 5000`**: healthcheck every 5 s. Lower values (`inter 1000`) caused tail-latency spikes (p999 jumped from 3.6 ms to 33 ms) because healthcheck connections compete with real traffic at peak load.
- **`maxconn 512` per server**: caps queue depth toward each API. Higher values just queue requests at the LB without helping throughput; lower values cause k6 to see connection-refused.
- **`http-check send ... hdr Connection close`**: healthcheck requests open and immediately close a new connection, so they don't poison the reuse pool.

**Go-relevant detail**: HAProxy doesn't care that the upstream is Go vs Bun vs C — it just speaks HTTP/1.1 over a UDS. We reuse this config almost verbatim.

## 6. Docker cgroups — the resource split

Total budget: 1.0 CPU, 350 MB. QRust's split (from their `docker-compose.yml`):

| service | CPU | RAM |
|---|---:|---:|
| haproxy-lb | 0.10 | 16 MB |
| qrust-api-1 | 0.45 | 167 MB |
| qrust-api-2 | 0.45 | 167 MB |
| **total** | **1.00** | **350 MB** |

QRust's `DECISIONS.md` notes they swept other splits (`0.46/0.46/0.08`, `0.47/0.47/0.06`, `0.475/0.475/0.05`) and **all lower-LB-CPU splits caused tail spikes and request launch failures**. So 0.10 CPU for the LB is the empirical floor on this hardware.

Why 0.45/0.45 and not 0.5/0.5 on the APIs? Because the LB needs that 0.10 to handle the round-robin + UDS + reuse pool at 900 RPS, and going lower turns the LB into a bottleneck.

**For the Go port**: same starting split. Go's `net/http` or `fasthttp` will use slightly less CPU per request than Bun at 450 RPS each, but the difference is marginal — start with the same numbers, sweep later if we have headroom.

**Memory**: 167 MB / replica is comfortable for Go with int16 quantization:

| component | Bun (QRust) | Go (target) |
|---|---:|---:|
| Runtime + stdlib | ~70 MB | ~15 MB |
| `dataset.bin` (mmap'd or in-memory) | ~87 MB | ~87 MB |
| Grid metadata (cells, AABBs, sorted indexes) | ~5 MB | ~5 MB |
| Buffers + slack | ~5 MB | ~13 MB |
| **per-replica total** | **~167 MB** | **~120 MB** |

The Go runtime is much smaller than Bun's. We could tighten the per-replica memory limit to ~150 MB and give that 30 MB back to a third API instance or just leave it as slack. Don't touch unless we have a reason.

## 7. Image and build

QRust's Dockerfile is one-stage: `FROM oven/bun:1.3`, copy source, run script. Image size ~280 MB. Bun is heavy.

**Go target**: multi-stage with `scratch` base, static binary, ~5 MB total. The image isn't directly scored, but a small image cold-starts faster on the Mac Mini.

```dockerfile
# stage 1: build
FROM --platform=linux/amd64 golang:1.23-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o /api ./cmd/api

# stage 2: runtime
FROM scratch
COPY --from=builder /api /api
ENTRYPOINT ["/api"]
```

The `-trimpath` + `-ldflags='-s -w'` strips debug info and source paths. `CGO_ENABLED=0` keeps it static and portable. `--platform=linux/amd64` is critical — the Mac Mini in the test env is amd64; if we build on Apple Silicon and forget, our image won't run.

**One nuance**: if we use AVX2 inline asm in the distance kernel, the binary still works on hosts without AVX2 (the asm file is conditional via `//go:build amd64`), but we need a runtime check (`golang.org/x/sys/cpu`) to fall back to pure-Go on unsupported CPUs. The Haswell Mac Mini has AVX2, so we're fine — but the safety net keeps us portable.

## 8. Dataset preparation — build time vs startup

QRust's `prepare.ts` reads `references.json.gz` (provided by the test runner under `/resources`) and writes `dataset.bin` to disk. This happens **at container startup**, not build time, because:

1. The references.json.gz is **not in the image** — it's mounted from `/resources` by the test runner.
2. The prep takes ~15 s (decompress, parse, partition, quantize, sort, in-place permutation), which fits in the `submission_health_check_retries: 20 × interval_ms: 3000` budget (= 60 s of grace before the test starts).

We do the same in Go:

```go
// main.go (cmd/api)
if !exists("/data/dataset.bin") {
    if err := prepare("/resources/references.json.gz", "/data/dataset.bin"); err != nil {
        log.Fatal(err)
    }
}
ds := mmap("/data/dataset.bin")     // direct cast: []int16 view onto the file
// ... start HTTP server, mark /ready as 200
```

Notes:
- `/data` is a shared volume between the two replicas. Only one needs to do the prep; the other waits. Use a file-lock pattern or simply: both attempt; second one sees the file exists and skips.
- `mmap` (via `golang.org/x/exp/mmap` or `syscall.Mmap`) means the data file shows up in the process address space without ever being copied to userland. **Both API replicas mmap'ing the same file → the OS page cache holds one physical copy.** This effectively shares the dataset between replicas at zero memory cost. Massive win.
- However, cgroup memory accounting: under cgroup v2, `memory.max` counts page cache that the cgroup faults in. The first replica to read pages will own them. The second's reads of already-cached pages are essentially free. So if we set both replicas' RSS limit to ~120 MB and the dataset is ~87 MB shared, both replicas see "they're using 30–35 MB of unique memory + the 87 MB of shared page cache". The math works out in practice; in theory cgroup accounting can be tricky. We measure.

## 9. Go-specific runtime knobs

These are the levers we'll pull once the system is otherwise stable:

### `GOGC` and `GOMEMLIMIT`

After the dataset is loaded and we've done a final `runtime.GC()`, freeze the heap. Two approaches:

1. **Disable GC entirely**: `debug.SetGCPercent(-1)`. Best p99/p999. Only safe if the steady-state hot path is **zero-allocation**. We aim for this.
2. **Soft limit with backoff**: `GOGC=200` (allow heap to grow 2× before GC) + `GOMEMLIMIT=150MiB` (cgroup-aligned hard ceiling). Safer if any allocations slip in.

**Recommendation**: start with #2 during development. Switch to #1 when pprof confirms zero alloc.

### `GOMAXPROCS`

The container has 0.45 CPU. Go defaults `GOMAXPROCS` to the number of CPUs visible to the process, which in a cgroup-limited container shows the *host's* CPU count (default value, you can read `/proc/cpuinfo`), not the limit. **Set `GOMAXPROCS=1`** explicitly via env var or `runtime.GOMAXPROCS(1)` early in `main`. With more threads than CPU shares, Go's scheduler thrashes on cooperative preemption.

### Goroutines per request

`net/http`'s `Server` spawns one goroutine per connection (not per request). With keep-alive + HAProxy's connection pool, we'll see ~512 long-lived goroutines per API replica. That's fine — Go goroutines are cheap, but it means we can't put hot-path state in goroutine-local storage (no such thing). Use `sync.Pool` if state is per-request, or just allocate fresh tiny buffers (the GC will keep up at our allocation rates).

### Avoiding `interface{}` and reflection

Every method call through an interface goes through an indirect jump (1 extra cycle), and the compiler can't inline. On the hot path, use concrete types everywhere. If you need polymorphism (e.g., distance kernel variants for different partition shapes), use a switch on a single byte + branch tables — the compiler often emits a computed-goto.

### Plan 9 assembly for the distance kernel

When pure-Go's autovectorizer fails (it almost always does on int16 SIMD), write the inner loop in `internal/vec/dist_amd64.s`:

```asm
// func dotInt16AVX2(q, v *int16, n int, threshold int64) int64
// Returns sum(q[i]-v[i])^2 for i in [0,n), early-exit if accumulator >= threshold.
// Implementation uses VPSUBW, VPMADDWD, VPADDD, VPHADDD.
```

This is the *last* optimization to attempt, only after the algorithm + JSON + HTTP path are clean. Asm is hard to debug and easy to get wrong; the pure-Go fallback must always produce identical results.

## 10. The full Go stack — a target diagram

```
                       host:9999
                          │
                ┌─────────▼─────────┐
                │     HAProxy 3.0   │     CPU: 0.10   RAM: 16MB
                │  round-robin LB   │
                └────┬───────┬──────┘
                     │       │
       /var/run/rinha/api1.sock      /var/run/rinha/api2.sock     ← tmpfs UDS
                     │       │
       ┌─────────────┼───────┼───────────┐
       │             │       │           │
       │   ┌─────────▼──┐ ┌──▼─────────┐ │       CPU: 0.45 each
       │   │  API-1     │ │   API-2    │ │       RAM: 120MB each
       │   │  Go static │ │  Go static │ │       Image: scratch
       │   │   binary   │ │   binary   │ │
       │   │            │ │            │ │
       │   │  HTTP/1.1  │ │  HTTP/1.1  │ │       net/http or fasthttp
       │   │  fused JSON│ │  fused JSON│ │       hand-rolled parser
       │   │  KNN-5     │ │  KNN-5     │ │       grid V2 + LB pruning
       │   │            │ │            │ │       int16 quantized
       │   │   mmap()  ─┼─┼─►  mmap() │ │
       │   └─────┬──────┘ └──────┬─────┘ │
       │         │               │       │
       │         └───────┬───────┘       │
       │                 ▼               │
       │       /data/dataset.bin         │       persistent volume
       │       (~87 MB int16+labels)     │       shared page cache
       └─────────────────────────────────┘
                         │
                         ▼
               /resources/references.json.gz   ← mounted by test runner
```

## 11. Implementation order for the Go scaffold

When we get to writing code, the order matters — each step should produce a runnable artifact:

1. **`cmd/api` scaffold**: `main.go` that listens on UDS, serves `GET /ready` (200) and `POST /fraud-score` (returns a hardcoded `fraud_score: 0`). Dockerfile, docker-compose.yml, haproxy.cfg. **Goal**: `curl http://localhost:9999/fraud-score -d '{}'` returns 200. This validates the entire infra stack.
2. **Brute-force search + stdlib JSON**: real vectorize, real KNN-5 over a small fixture (e.g., the example-references.json). **Goal**: correctness against the example-payloads expected answers.
3. **Real dataset loader**: stream `references.json.gz`, write `dataset.bin`, mmap. **Goal**: `/ready` returns 200 after ~15 s startup; first request succeeds.
4. **Quantization → int16**: switch from float to int16 throughout. **Goal**: no detection changes vs step 2.
5. **Partition layer**: bit-key + per-partition scan. **Goal**: same accuracy, faster.
6. **Grid V2 layer**: cells + AABBs + LB pruning. **Goal**: same accuracy, much faster.
7. **Fused JSON parser**: replace stdlib with hand-rolled lexer. **Goal**: zero allocations on the hot path (verified with pprof).
8. **Prebuilt response strings + body cache**: drop the response encoder. **Goal**: response side <5 μs.
9. **Optional: fasthttp swap, AVX2 asm distance kernel**. **Goal**: chase the last 100 μs.
10. **Benchmark sweep, compare with QRust baseline.**

## 12. Where this lecture ends and the next begins

We now understand:
- The per-request budget breakdown and where time goes.
- TCP loopback vs UDS in tmpfs (and the 70+ μs difference).
- `net/http` vs `fasthttp` and when to switch.
- JSON parsing options from generic to fused.
- HAProxy config that works at 900 RPS.
- Docker cgroup splits and why 0.10/0.45/0.45 is the empirical sweet spot.
- The dataset prep pipeline (gzip stream → partitioned int16 bin).
- Go runtime knobs (GOGC, GOMEMLIMIT, GOMAXPROCS, asm).
- The full target stack and the implementation order.

What we have **not** yet covered:
- How to **measure** all of this rigorously: k6 invocation, results parsing, per-container metrics, charting runs against each other.

That's Lecture 04.
