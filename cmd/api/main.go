package main

import (
	"errors"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/api"
	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

const indexPath = "/resources/index.bin"

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
