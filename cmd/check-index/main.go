// check-index loads an IVF index file and runs FraudCountIVF against the
// first 10 fraud and 10 legit entries of test-data.json. Used to isolate
// whether the bug lives in the on-disk index, in Load, or in the api wiring.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/ivf"
	"github.com/muanlartins/rinha-de-backend-2026/internal/search"
	"github.com/muanlartins/rinha-de-backend-2026/internal/vector"
)

func main() {
	if len(os.Args) < 3 {
		log.Fatal("usage: check-index <index.bin> <test-data.json>")
	}
	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	idx, err := ivf.Load(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	var labelFrauds int
	for _, b := range idx.Labels {
		if b == 1 {
			labelFrauds++
		}
	}
	log.Printf("loaded: N=%d K=%d blocks=%d label_frauds=%d", idx.N, idx.K, idx.Blocks, labelFrauds)

	tf, err := os.Open(os.Args[2])
	if err != nil {
		log.Fatal(err)
	}
	type entry struct {
		Request          json.RawMessage `json:"request"`
		ExpectedApproved bool            `json:"expected_approved"`
	}
	var top struct {
		Entries []entry `json:"entries"`
	}
	if err := json.NewDecoder(tf).Decode(&top); err != nil {
		log.Fatal(err)
	}
	tf.Close()

	var scratch search.IVFScratch
	// Run first 20 entries, print fraud count + expected
	for i, e := range top.Entries[:20] {
		var qi [dataset.Stride]int16
		if !vector.VectorizeFast(e.Request, &qi) {
			fmt.Printf("entry %d: PARSE FAIL\n", i)
			continue
		}
		var qiArr [dataset.Dims]int16
		var qf [dataset.Dims]float32
		for d := 0; d < dataset.Dims; d++ {
			qiArr[d] = qi[d]
			qf[d] = float32(qi[d])
		}
		count := search.FraudCountIVF(&qf, &qiArr, idx, &scratch)
		fmt.Printf("entry %d: count=%d expected=%v\n", i, count, e.ExpectedApproved)
	}
}
