package main

import (
	"errors"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/api"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
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
		log.Printf("loading IVF index from %s ...", indexPath)
		t0 := time.Now()
		f, err := os.Open(indexPath)
		if err != nil {
			log.Fatalf("open index: %v", err)
		}
		idx, err := ivf.Load(f)
		f.Close()
		if err != nil {
			log.Fatalf("load index: %v", err)
		}
		log.Printf("index loaded: N=%d K=%d blocks=%d in %s",
			idx.N, idx.K, idx.Blocks, time.Since(t0))

		runtime.GC()
		handler.SetIndex(idx)
	}

	select {}
}
