# Lecture 13 — mmap, madvise, and Page-Cache Sharing

This lecture covers the final layer of Tier 3: how to load the 80 MB IVF index into the runtime so it (a) doesn't count against the Go heap, (b) is shared between two replicas via the kernel page cache, and (c) is pre-faulted so the first request doesn't pay page-fault cost.

## The mental model

Linux memory is organized in 4 KB pages. Three kinds of pages matter:

- **Anonymous pages**: allocated by `malloc`, `make`, etc. Backed by RAM (and swap if pressed). Process-private.
- **File-backed pages**: returned by `mmap()` on a file. The kernel reads pages from disk on first access (page fault). Multiple processes that mmap the same inode see the same backing pages.
- **Page cache**: the kernel's read cache for file-backed pages. Lives until evicted under memory pressure.

When we `read()` an 80 MB file into a Go slice:

1. The kernel reads 80 MB from disk into anonymous pages allocated for our process.
2. The Go runtime considers those pages part of the heap.
3. The Go GC scans the slice on every collection (it's a typed pointer slice, so GC walks it).
4. If two API replicas read the same file, the kernel keeps two copies — one per process — totaling 160 MB.

When we `mmap()` an 80 MB file:

1. The kernel sets up a virtual memory mapping in our process. **No pages are actually read until we touch them.**
2. On first touch of each page, the kernel reads the page from disk into the page cache, then maps it into our address space.
3. The Go runtime considers the slice "off-heap" (it points into a `mmap`-managed region). GC ignores it.
4. If two API replicas mmap the same inode, the kernel keeps **one copy** in the page cache, shared between both processes' address spaces.

So mmap gives us three wins: free heap pressure, free inode sharing, free GC reduction.

## `madvise` flags we use

Linux exposes `madvise(addr, len, advice)` to hint to the kernel about how we'll use a mapping. The relevant flags:

### `MADV_RANDOM`

Tells the kernel "I will read pages in random order, don't do readahead." By default, Linux readahead is aggressive: when you touch page N of a file, the kernel preemptively pulls pages N+1, N+2, ..., N+128 into the page cache, betting on sequential access. For a random-access pattern, those prefetched pages waste bandwidth (we won't touch them) and waste cache (they evict pages we want).

Our IVF search hits clusters at random in the block file. After centroid-distance ranking, we visit cluster k's blocks; then cluster k'; then cluster k''. Each cluster's blocks are contiguous, so within a cluster we benefit from readahead — but across clusters, no. The empirical sweet spot per Repo B (lemesdaniel-zig, rank 7) is `MADV_RANDOM`: small intra-cluster scans don't lose much, and we save substantial bandwidth on the inter-cluster jumps.

### `MADV_POPULATE_READ` (Linux 5.14+)

Tells the kernel "read these pages from disk and map them into me right now." Without this, the first time a page is touched it triggers a synchronous page fault and the user request waits ~100 µs while the kernel does the read.

We call this once at startup, after mmap, before opening the listener. The 80 MB is fully populated before any request arrives. First-request latency is then bound by the cold L1/L2/L3 caches, not by page-fault latency.

If the kernel is older than 5.14, we fall back to manually touching every page in a loop:

```go
const pageSize = 4096
for i := 0; i < len(mmapped); i += pageSize {
    _ = mmapped[i]
}
```

Same effect, ~50% slower than `MADV_POPULATE_READ` because the loop incurs ~80 million touch operations vs one syscall.

### `MADV_HUGEPAGE`

Tells the kernel "consider promoting these pages to transparent hugepages." A regular 4 KB page costs one TLB entry. An 80 MB file at 4 KB pages needs 20 000 TLB entries — far more than Haswell's TLB capacity (1024 entries). Result: every access misses the TLB and incurs a page-walk.

A 2 MB hugepage costs one TLB entry per 2 MB. 80 MB / 2 MB = 40 entries — fits comfortably in the TLB.

Transparent hugepages aren't always granted — the kernel decides based on memory fragmentation. `MADV_HUGEPAGE` opts us into the eligibility. We don't lose anything if the kernel chooses not to promote.

## Two replicas, one page cache

Docker compose lets us reference the same image by URI for both `api1` and `api2`. The image contains `/resources/index.bin` at a specific inode. Both replicas mount it from the same image layer, so the kernel sees them mapping the same inode — and shares the page cache.

This is critical for fitting within 350 MB of total memory. If we read-into-heap, the index occupies 2 × 80 MB = 160 MB of anonymous RAM. If we mmap from the same inode, the index occupies 80 MB of page cache + nearly zero per-process overhead.

For comparison, the contest hardware has 350 MB. We have:

| Component | Read-into-heap | Mmap-shared |
|---|---:|---:|
| Two api containers, Go runtime + heap | ~40 MB × 2 = 80 MB | ~30 MB × 2 = 60 MB |
| Index data (Go-managed) | 80 MB × 2 = 160 MB | 0 MB |
| Index data (page cache, kernel) | — | 80 MB |
| LB container | ~30 MB | ~30 MB |
| Total | **270 MB** | **170 MB** |

100 MB headroom either way, but mmap is the cleaner win and reduces GC pressure on every collection.

## Implementation pattern

```go
import (
    "os"
    "golang.org/x/sys/unix"
)

func LoadMmap(path string) (*IVFIndex, error) {
    f, err := os.Open(path)
    if err != nil { return nil, err }
    defer f.Close()

    stat, _ := f.Stat()
    size := int(stat.Size())

    data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE)
    if err != nil { return nil, err }

    // Hint advice. Failures are silently ignored — best-effort.
    _ = unix.Madvise(data, unix.MADV_RANDOM)
    _ = unix.Madvise(data, unix.MADV_POPULATE_READ)
    _ = unix.Madvise(data, unix.MADV_HUGEPAGE)

    idx := parseHeaderAndSliceViews(data)
    idx.raw = data  // keep alive
    return idx, nil
}
```

The "parse header and slice views" step walks the binary format and uses `unsafe.Slice` to create typed slices pointing into the mmap'd region. No copying. The slices are valid as long as `idx.raw` is not unmapped.

To release: `idx.raw, idx = nil, nil; runtime.GC()` — the unmapper is the garbage collector via a finalizer, or we can call `unix.Munmap(idx.raw)` explicitly.

## What we are *not* doing

- **mlock**. Pins pages to RAM, prevents swap. Our containers don't have swap configured anyway, and the rinha environment's eviction pressure is minimal. We skip the complexity.

- **HugeTLBFS** (`MAP_HUGETLB`). Reserves explicit 2 MB pages from a static pool. Requires kernel config and admin setup. Transparent hugepages via `MADV_HUGEPAGE` are sufficient.

- **DAX direct I/O**. For NVMe persistent memory. Not relevant on the contest hardware.

## Profiling notes

If the mmap version is slower than the heap version (which would be surprising), the diagnostic chain is:

1. Are pages actually populated at startup? Check `/proc/self/smaps` for the mapping; `Anonymous: 0 KB` and `Rss: ~80 MB` means populated. `Rss: 0 KB` means populate didn't run.
2. Is `MADV_RANDOM` actually applied? Use `bpftrace` or `strace` to confirm the syscall succeeded.
3. Are hugepages granted? Check `/proc/self/smaps` for `AnonHugePages: N kB` — should be a multiple of 2048 (KB per hugepage).
4. Is the page cache being evicted? `vmtouch -v /resources/index.bin` shows currently-resident page count vs total.

## What "warmup" does on top of mmap

After mmap + populate, the pages are in the page cache but not in the CPU caches. The first 50-500 requests still pay cold L1/L2/L3 cache costs. The "warmup" pattern is to run a few hundred deterministic queries before opening the listener, so:

- The branch predictor learns the hot paths.
- The L1/L2 instruction cache is loaded with our hot kernel code.
- Pages most likely to be accessed first are warmed in L2/L3.

The actual user benefit is small — by the time the bench's 100k requests have run, all caches are saturated naturally. But the *cold-start* p99 (first second of the bench) is reduced from ~50 ms to ~5 ms, which can affect the bench's p99 calculation if it includes that period.

We do warmup with a deterministic subset of `test-data.json` (the first 500 entries), one query per loop iteration. Total warmup time: ~500 × 50 µs = 25 ms. Hidden behind the listener startup.

## References

- `man 2 mmap`, `man 2 madvise` — Linux man pages.
- `madvise(2)` flags table: `linux/include/uapi/asm-generic/mman-common.h`.
- "Linux Kernel Development" by R. Love, chapter on the page cache.
- Postgres' `mmap` vs `read` decision tree in `backend/storage/file/fd.c`.
