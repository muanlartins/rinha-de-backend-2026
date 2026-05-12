package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
)

// build-index reads a gzipped references.json, trains a k-means IVF index
// over it, and writes the binary index file consumed by the runtime.
//
// Usage: build-index <references.json.gz> <out.bin>
//
// The build is deterministic: rerunning with the same input produces the
// byte-identical output. See docs/lectures/09-kmeans-ivf.md § Determinism.
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
	centroids := ivf.TrainKMeans(ds.Vectors, ds.Count)
	log.Printf("train: k=%d sample=%d iters=%d in %s",
		ivf.K, ivf.SampleSize, ivf.MaxIters, time.Since(t1))

	t2 := time.Now()
	assign := ivf.AssignAll(ds.Vectors, ds.Count, &centroids)
	log.Printf("assign-all: %s", time.Since(t2))

	t3 := time.Now()
	ivf.SetLabelSource(ds.Labels)
	idx, err := ivf.Build(ds.Vectors, ds.Count, &centroids, assign)
	if err != nil {
		log.Fatalf("build: %v", err)
	}
	log.Printf("build: %d blocks in %s", idx.Blocks, time.Since(t3))

	t4 := time.Now()
	if err := idx.SerializeToFile(out); err != nil {
		log.Fatalf("save: %v", err)
	}
	info, _ := os.Stat(out)
	log.Printf("saved %d bytes (%.1f MB) to %s in %s",
		info.Size(), float64(info.Size())/(1<<20), out, time.Since(t4))

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	log.Printf("heap inuse %.1f MB, alloc total %.1f MB",
		float64(m.HeapInuse)/(1<<20), float64(m.TotalAlloc)/(1<<20))
	log.Printf("total wall time: %s", time.Since(t0))
}
