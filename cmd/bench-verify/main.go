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

func approvedFromCount(c uint8) bool { return float64(c)/5.0 < 0.6 }

func main() {
	f, _ := os.Open("/resources/index.bin")
	idx, _ := ivf.Load(f)
	f.Close()
	tf, _ := os.Open("references/rinha-official/test/test-data.json")
	var top struct{ Entries []entry `json:"entries"` }
	json.NewDecoder(tf).Decode(&top)
	tf.Close()

	type Q struct {
		qf [dataset.Dims]float32
		qi [dataset.Dims]int16
		ea bool
	}
	queries := make([]Q, 0, len(top.Entries))
	for _, e := range top.Entries {
		var raw [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &raw) { continue }
		var q Q
		for d := 0; d < dataset.Dims; d++ {
			q.qf[d] = float32(raw[d]); q.qi[d] = raw[d]
		}
		q.ea = e.ExpectedApproved
		queries = append(queries, q)
	}
	log.Printf("loaded %d queries", len(queries))

	var scratch search.IVFScratch
	// Warm
	for i := 0; i < 1000 && i < len(queries); i++ {
		_ = search.FraudCountIVF(&queries[i].qf, &queries[i].qi, idx, &scratch)
	}

	fp, fn := 0, 0
	times := make([]int64, len(queries))
	for i, q := range queries {
		s := time.Now()
		cnt := search.FraudCountIVF(&q.qf, &q.qi, idx, &scratch)
		times[i] = time.Since(s).Nanoseconds()
		a := approvedFromCount(cnt)
		switch {
		case a && !q.ea: fn++
		case !a && q.ea: fp++
		}
	}
	sort.Slice(times, func(i,j int)bool{return times[i]<times[j]})
	fmt.Printf("FP=%d FN=%d  p50=%dns p90=%dns p99=%dns p999=%dns max=%dns\n",
		fp, fn, times[len(times)/2], times[len(times)*9/10],
		times[len(times)*99/100], times[len(times)*999/1000], times[len(times)-1])
}
