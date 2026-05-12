package dataset

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSaveLoadRoundtrip: build from gzip, save, load, compare structures.
// The loaded dataset must be byte-identical to the in-memory one.
func TestSaveLoadRoundtrip(t *testing.T) {
	path := "../../references/rinha-official/resources/references.json.gz"
	if _, err := os.Stat(path); err != nil {
		t.Skip("references unavailable")
	}

	ds, err := LoadFromGzipJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	ds.BuildGrid()

	tmpFile := filepath.Join(t.TempDir(), "rt.bin")
	if err := ds.SaveIndex(tmpFile); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := LoadIndex(tmpFile)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if loaded.Count != ds.Count {
		t.Fatalf("count: got %d want %d", loaded.Count, ds.Count)
	}
	for p := 0; p < NumPartitions; p++ {
		if loaded.PartitionStarts[p] != ds.PartitionStarts[p] {
			t.Fatalf("partition %d start: got %d want %d", p, loaded.PartitionStarts[p], ds.PartitionStarts[p])
		}
		if loaded.PartitionCounts[p] != ds.PartitionCounts[p] {
			t.Fatalf("partition %d count: got %d want %d", p, loaded.PartitionCounts[p], ds.PartitionCounts[p])
		}
		if loaded.Grids[p].NumCells != ds.Grids[p].NumCells {
			t.Fatalf("partition %d numCells: got %d want %d", p, loaded.Grids[p].NumCells, ds.Grids[p].NumCells)
		}
	}
	for i := 0; i < ds.Count*Stride; i++ {
		if loaded.Vectors[i] != ds.Vectors[i] {
			t.Fatalf("vector mismatch at idx %d: got %d want %d", i, loaded.Vectors[i], ds.Vectors[i])
		}
	}
	for i := 0; i < ds.Count; i++ {
		if loaded.Labels[i] != ds.Labels[i] {
			t.Fatalf("label mismatch at idx %d: got %d want %d", i, loaded.Labels[i], ds.Labels[i])
		}
	}
}
