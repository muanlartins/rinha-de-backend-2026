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
