package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
	"github.com/muanlartins/rinha-de-backend-2026/internal/search"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

type entry struct {
	Request          json.RawMessage `json:"request"`
	ExpectedApproved bool            `json:"expected_approved"`
}

func main() {
	f, err := os.Open("/resources/index.bin")
	if err != nil {
		log.Fatal(err)
	}
	idx, err := ivf.Load(f)
	if err != nil {
		log.Fatal(err)
	}
	f.Close()

	tf, _ := os.Open("references/rinha-official/test/test-data.json")
	var top struct {
		Entries []entry `json:"entries"`
	}
	json.NewDecoder(tf).Decode(&top)
	tf.Close()

	type Q struct {
		qf [dataset.Dims]float32
		qi [dataset.Dims]int16
	}
	queries := make([]Q, 0, len(top.Entries))
	for _, e := range top.Entries {
		var raw [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &raw) {
			continue
		}
		var q Q
		for d := 0; d < dataset.Dims; d++ {
			q.qf[d] = float32(raw[d])
			q.qi[d] = raw[d]
		}
		queries = append(queries, q)
	}
	log.Printf("loaded %d queries", len(queries))

	var scratch search.IVFScratch
	// Warm up
	for i := 0; i < 1000 && i < len(queries); i++ {
		_ = search.FraudCountIVF(&queries[i].qf, &queries[i].qi, idx, &scratch)
	}

	times := make([]int64, len(queries))
	t0 := time.Now()
	for i, q := range queries {
		s := time.Now()
		_ = search.FraudCountIVF(&q.qf, &q.qi, idx, &scratch)
		times[i] = time.Since(s).Nanoseconds()
	}
	total := time.Since(t0)
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	p50 := times[len(times)*50/100]
	p90 := times[len(times)*90/100]
	p99 := times[len(times)*99/100]
	p999 := times[len(times)*999/1000]
	mx := times[len(times)-1]
	mean := total.Nanoseconds() / int64(len(times))
	fmt.Printf("count=%d total=%s mean=%dns p50=%dns p90=%dns p99=%dns p999=%dns max=%dns\n",
		len(times), total, mean, p50, p90, p99, p999, mx)
}
