package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

func main() {
	t0 := time.Now()
	ds, err := dataset.LoadFromGzipJSON("references/rinha-official/resources/references.json.gz")
	if err != nil {
		log.Fatal(err)
	}
	loadDur := time.Since(t0)

	t1 := time.Now()
	ds.Partition()
	partDur := time.Since(t1)

	t2 := time.Now()
	ds.BuildGrid()
	gridDur := time.Since(t2)

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("load:      %d vectors in %s\n", ds.Count, loadDur)
	fmt.Printf("partition: %s\n", partDur)
	fmt.Printf("grid:      %s\n", gridDur)
	fmt.Printf("heap inuse: %.1f MB, alloc total: %.1f MB\n",
		float64(m.HeapInuse)/(1<<20), float64(m.TotalAlloc)/(1<<20))

	// Sanity check: every vector in partition k's range should compute key k.
	fmt.Println("\nverifying partition layout...")
	mismatches := 0
	for k := 0; k < dataset.NumPartitions; k++ {
		start := ds.PartitionStarts[k]
		count := ds.PartitionCounts[k]
		for i := start; i < start+count; i++ {
			var v [dataset.Dims]int16
			base := int(i) * dataset.Dims
			copy(v[:], ds.Vectors[base:base+dataset.Dims])
			if int(dataset.ComputeKey(&v)) != k {
				mismatches++
				if mismatches <= 5 {
					fmt.Printf("  MISMATCH at idx=%d: expected key=%d got key=%d\n",
						i, k, dataset.ComputeKey(&v))
				}
			}
		}
	}
	if mismatches == 0 {
		fmt.Printf("  OK: all %d vectors land in the correct partition\n", ds.Count)
	} else {
		fmt.Printf("  FAIL: %d mismatches\n", mismatches)
	}

	fmt.Println("\npartition + grid layout:")
	totalFraud := 0
	for k := 0; k < dataset.NumPartitions; k++ {
		count := ds.PartitionCounts[k]
		if count == 0 {
			continue
		}
		start := ds.PartitionStarts[k]
		// fraud count within this partition
		f := 0
		for i := start; i < start+count; i++ {
			if ds.Labels[i] == 1 {
				f++
			}
		}
		totalFraud += f
		ncells := 0
		if ds.Partitions[k] != nil {
			ncells = ds.Partitions[k].NumCells
		}
		fmt.Printf("  k=%02d (%s): n=%d cells=%d fraud=%d (%.1f%%)\n",
			k, keyBits(uint8(k)), count, ncells, f, 100*float64(f)/float64(count))
	}
	fmt.Printf("\ntotal fraud across all partitions: %d (%.2f%%)\n",
		totalFraud, 100*float64(totalFraud)/float64(ds.Count))
}

func keyBits(k uint8) string {
	bits := ""
	if k&1 != 0 {
		bits += "online "
	}
	if k&2 != 0 {
		bits += "present "
	}
	if k&4 != 0 {
		bits += "unknown_merch "
	}
	if k&8 != 0 {
		bits += "sentinel5 "
	}
	if k&16 != 0 {
		bits += "sentinel6 "
	}
	if bits == "" {
		bits = "all-zero"
	}
	return bits
}
