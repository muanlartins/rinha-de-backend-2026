package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: build-index <references.json.gz> <out.bin>")
		os.Exit(2)
	}
	in, out := os.Args[1], os.Args[2]

	t0 := time.Now()
	ds, err := dataset.LoadFromGzipJSON(in)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	log.Printf("load: %d vectors in %s", ds.Count, time.Since(t0))

	t1 := time.Now()
	ds.BuildGrid()
	log.Printf("build-grid: %s", time.Since(t1))

	t3 := time.Now()
	if err := ds.SaveIndex(out); err != nil {
		log.Fatalf("save: %v", err)
	}
	info, _ := os.Stat(out)
	log.Printf("saved %d bytes to %s in %s", info.Size(), out, time.Since(t3))

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	log.Printf("heap inuse %.1f MB, alloc total %.1f MB",
		float64(m.HeapInuse)/(1<<20), float64(m.TotalAlloc)/(1<<20))
}
