package main

import (
	"errors"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/api"
	"github.com/muanlartins/rinha-de-backend-2026/internal/fdpass"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
)

const (
	indexPath   = "/resources/index.bin"
	warmupIters = 500
)

func main() {
	runtime.GOMAXPROCS(1)

	debug.SetGCPercent(200)
	debug.SetMemoryLimit(140 << 20)

	socketPath := os.Getenv("API_SOCKET")
	if socketPath == "" {
		log.Fatal("API_SOCKET env var is required")
	}

	handler := api.NewHandler()

	if err := api.ListenRaw(socketPath, handler); err != nil {
		log.Fatalf("listen on %q: %v", socketPath, err)
	}
	log.Printf("listening on %s", socketPath)

	// Optional fd-passing control channel. The fd-passing LB (jrblatt/
	// so-no-forevis) sends accepted TCP client fds over this socket via
	// SCM_RIGHTS; we adopt them as if we had locally accept()ed. The
	// regular UDS accept loop above still serves as a fallback when the
	// LB falls back to TCP proxy. See docs/lectures/12-scm-rights.md.
	if ctrlPath := os.Getenv("API_CTRL_SOCKET"); ctrlPath != "" {
		fdCh, _, err := fdpass.Listen(ctrlPath)
		if err != nil {
			log.Printf("WARN: fdpass listen on %q failed: %v", ctrlPath, err)
		} else {
			log.Printf("fdpass listening on %s", ctrlPath)
			go api.ServeFDChannel(fdCh, handler)
		}
	}

	if _, err := os.Stat(indexPath); errors.Is(err, os.ErrNotExist) {
		log.Printf("WARN: %s not found; coming up in stub mode", indexPath)
		handler.MarkReady()
	} else {
		log.Printf("mmap'ing IVF index from %s ...", indexPath)
		t0 := time.Now()
		idx, err := ivf.LoadMmap(indexPath)
		if err != nil {
			log.Fatalf("mmap index: %v", err)
		}
		log.Printf("index mmapped: N=%d K=%d blocks=%d in %s",
			idx.N, idx.K, idx.Blocks, time.Since(t0))

		// In-process warmup before the handler accepts requests. Warms
		// CPU caches and branch predictor with hot search code paths so
		// the first real requests don't pay cold-cache penalty.
		t1 := time.Now()
		d := api.Warmup(idx, warmupIters)
		log.Printf("warmup: %d iters in %s (avg %s)", warmupIters, d, d/time.Duration(warmupIters))

		runtime.GC()
		handler.SetIndex(idx)
		log.Printf("ready in %s (total since startup: %s)", time.Since(t1), time.Since(t0))

		// Phase 35c (opt-in via STEADY_GC_OFF=1) — disable Go's
		// automatic GC after warmup and instead run GC in a background
		// goroutine on a fixed timer. The hot path has no heap
		// escapes (verified via `go build -gcflags=-m` 2026-05-14)
		// and IVFScratch + read buffers come from sync.Pool, so the
		// heap should stay nearly flat during request handling. Auto
		// GC firing mid-request adds 50-500 µs of STW pause; running
		// it on a timer between requests removes that tail.
		//
		// Safety: GOMEMLIMIT=140MB is still enforced; if a leak
		// surfaces, Go will GC anyway before OOM.
		if os.Getenv("STEADY_GC_OFF") == "1" {
			debug.SetGCPercent(-1)
			log.Printf("steady-state GC disabled; periodic GC every 5s")
			go func() {
				t := time.NewTicker(5 * time.Second)
				defer t.Stop()
				for range t.C {
					runtime.GC()
				}
			}()
		}
	}

	select {}
}
