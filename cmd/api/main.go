package main

import (
	"errors"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/api"
	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

const indexPath = "/resources/index.bin"

func main() {
	// runtime.NumCPU reports host CPUs, not the cgroup share — pin to the
	// container's 0.45 CPU.
	runtime.GOMAXPROCS(1)

	// See docs/CODE_NOTES.md "Why GOMAXPROCS(1) and the GOGC/GOMEMLIMIT pair".
	debug.SetGCPercent(200)
	debug.SetMemoryLimit(140 << 20)

	socketPath := os.Getenv("API_SOCKET")
	if socketPath == "" {
		log.Fatal("API_SOCKET env var is required")
	}

	if err := os.Remove(socketPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		log.Fatalf("remove stale socket %q: %v", socketPath, err)
	}

	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		log.Fatalf("listen on %q: %v", socketPath, err)
	}
	defer ln.Close()

	if err := os.Chmod(socketPath, 0o666); err != nil {
		log.Fatalf("chmod socket %q: %v", socketPath, err)
	}

	handler := api.NewHandler()

	// Start serving before the dataset load so /ready can return 503 while
	// HAProxy waits.
	srv := &http.Server{Handler: handler}
	go func() {
		log.Printf("listening on %s", socketPath)
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("serve: %v", err)
		}
	}()

	if _, err := os.Stat(indexPath); errors.Is(err, os.ErrNotExist) {
		log.Printf("WARN: %s not found; coming up in stub mode", indexPath)
		handler.MarkReady()
	} else {
		log.Printf("loading pre-built index from %s ...", indexPath)
		t0 := time.Now()
		ds, err := dataset.LoadIndex(indexPath)
		if err != nil {
			log.Fatalf("load index: %v", err)
		}
		log.Printf("index loaded: %d vectors in %s", ds.Count, time.Since(t0))

		runtime.GC()
		handler.SetDataset(ds)
	}

	select {}
}
