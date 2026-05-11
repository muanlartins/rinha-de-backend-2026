// Rinha de Backend 2026 — Go submission entry point.
package main

import (
	"errors"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/api"
	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

const referencesPath = "/resources/references.json.gz"

func main() {
	// Cgroup-limited containers report all host CPUs via NumCPU; pin to 1 so
	// the scheduler doesn't oversubscribe the 0.45 CPU share.
	runtime.GOMAXPROCS(1)

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

	// Start the HTTP server before we touch the dataset so /ready can return
	// 503 while we're loading. HAProxy waits for /ready=200 before sending
	// real traffic.
	srv := &http.Server{Handler: handler}
	go func() {
		log.Printf("listening on %s", socketPath)
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("serve: %v", err)
		}
	}()

	// Try to load the reference dataset. If the file doesn't exist (e.g.,
	// smoke-only invocation, no /resources volume), come up ready in stub mode
	// so the smoke test still passes.
	if _, err := os.Stat(referencesPath); errors.Is(err, os.ErrNotExist) {
		log.Printf("WARN: %s not found; coming up in stub mode", referencesPath)
		handler.MarkReady()
	} else {
		log.Printf("loading dataset from %s ...", referencesPath)
		t0 := time.Now()
		ds, err := dataset.LoadFromGzipJSON(referencesPath)
		if err != nil {
			log.Fatalf("load dataset: %v", err)
		}
		log.Printf("dataset loaded: %d vectors in %s", ds.Count, time.Since(t0))

		t1 := time.Now()
		ds.Partition()
		log.Printf("dataset partitioned in %s", time.Since(t1))

		t2 := time.Now()
		ds.BuildGrid()
		log.Printf("grid built in %s", time.Since(t2))

		// Force a GC pass to release transient build allocations before we
		// start serving traffic.
		runtime.GC()

		handler.SetDataset(ds)
	}

	// Block the main goroutine; the http.Server runs in a goroutine above.
	select {}
}
