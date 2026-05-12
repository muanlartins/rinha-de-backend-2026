package ivf

import (
	"bytes"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// TestIndexRoundTrip serializes a built index, deserializes it, and confirms
// every field matches byte-for-byte. If this fails, Load and Serialize have
// drifted apart.
func TestIndexRoundTrip(t *testing.T) {
	ds := loadDataset(t)
	if ds == nil {
		return
	}

	centroids := TrainKMeans(ds.Vectors, ds.Count)
	assign := AssignAll(ds.Vectors, ds.Count, &centroids)
	SetLabelSource(ds.Labels)
	idx, err := Build(ds.Vectors, ds.Count, &centroids, assign)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	var buf bytes.Buffer
	if err := idx.Serialize(&buf); err != nil {
		t.Fatalf("serialize: %v", err)
	}

	got, err := Load(&buf)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if got.N != idx.N || got.K != idx.K || got.Blocks != idx.Blocks {
		t.Fatalf("header mismatch: got N=%d K=%d Blocks=%d, want N=%d K=%d Blocks=%d",
			got.N, got.K, got.Blocks, idx.N, idx.K, idx.Blocks)
	}
	assertF32Equal(t, "centroids", got.Centroids, idx.Centroids)
	assertU32Equal(t, "offsets", got.Offsets, idx.Offsets)
	assertI16Equal(t, "bboxMin", got.BboxMin, idx.BboxMin)
	assertI16Equal(t, "bboxMax", got.BboxMax, idx.BboxMax)
	assertU8Equal(t, "labels", got.Labels, idx.Labels)
	assertI16Equal(t, "blockData", got.BlockData, idx.BlockData)
}

// TestClusterOffsetsConsistent confirms that every block lies within some
// cluster's range and that offsets are monotonically increasing.
func TestClusterOffsetsConsistent(t *testing.T) {
	ds := loadDataset(t)
	if ds == nil {
		return
	}

	centroids := TrainKMeans(ds.Vectors, ds.Count)
	assign := AssignAll(ds.Vectors, ds.Count, &centroids)
	SetLabelSource(ds.Labels)
	idx, err := Build(ds.Vectors, ds.Count, &centroids, assign)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	for c := 0; c < K; c++ {
		if idx.Offsets[c] > idx.Offsets[c+1] {
			t.Fatalf("offsets[%d]=%d > offsets[%d]=%d", c, idx.Offsets[c], c+1, idx.Offsets[c+1])
		}
	}
	if idx.Offsets[K] != idx.Blocks {
		t.Fatalf("offsets[K]=%d != Blocks=%d", idx.Offsets[K], idx.Blocks)
	}
}

// TestBboxContainsClusterVectors: every vector v of cluster c has
// bmin[c][d] <= v[d] <= bmax[c][d] for every d. This must hold because we
// derived the bbox from the cluster's vectors. A mismatch means we wrote
// the wrong cluster's bbox or the wrong cluster's vectors.
func TestBboxContainsClusterVectors(t *testing.T) {
	ds := loadDataset(t)
	if ds == nil {
		return
	}

	centroids := TrainKMeans(ds.Vectors, ds.Count)
	assign := AssignAll(ds.Vectors, ds.Count, &centroids)
	SetLabelSource(ds.Labels)
	idx, err := Build(ds.Vectors, ds.Count, &centroids, assign)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	// Pick 100 random vectors, verify their cluster's bbox contains them.
	for trial := 0; trial < 100; trial++ {
		id := uint32((trial * 31337) % ds.Count)
		c := assign[id]
		base := int(id) * dataset.Stride
		for d := 0; d < dataset.Dims; d++ {
			v := ds.Vectors[base+d]
			mn := idx.BboxMin[int(c)*bboxLanes+d]
			mx := idx.BboxMax[int(c)*bboxLanes+d]
			if v < mn || v > mx {
				t.Errorf("ref %d cluster %d dim %d: value %d outside bbox [%d, %d]",
					id, c, d, v, mn, mx)
			}
		}
	}
}

// TestBlocksContainOnlyClusterVectors: every non-phantom lane of every
// block of cluster c must be one of the vectors originally assigned to c.
// We verify this by reconstructing the cluster's bucket and checking the
// block lanes match a set membership.
func TestBlocksContainOnlyClusterVectors(t *testing.T) {
	ds := loadDataset(t)
	if ds == nil {
		return
	}

	centroids := TrainKMeans(ds.Vectors, ds.Count)
	assign := AssignAll(ds.Vectors, ds.Count, &centroids)
	SetLabelSource(ds.Labels)
	idx, err := Build(ds.Vectors, ds.Count, &centroids, assign)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	// Bucket vectors by cluster for verification.
	buckets := make([][]uint32, K)
	for id, c := range assign {
		buckets[c] = append(buckets[c], uint32(id))
	}

	// Spot-check 10 clusters.
	for _, c := range []int{0, 17, 100, 500, 1000, 2000, 3000, 3999, K - 1, 42} {
		want := make(map[[dataset.Dims]int16]bool, len(buckets[c]))
		for _, id := range buckets[c] {
			var v [dataset.Dims]int16
			base := int(id) * dataset.Stride
			for d := 0; d < dataset.Dims; d++ {
				v[d] = ds.Vectors[base+d]
			}
			want[v] = true
		}

		start := int(idx.Offsets[c])
		end := int(idx.Offsets[c+1])
		nLanes := len(buckets[c])
		for b := start; b < end; b++ {
			for lane := 0; lane < blockLanes; lane++ {
				laneIdx := (b-start)*blockLanes + lane
				if laneIdx >= nLanes {
					break // phantom lane
				}
				var v [dataset.Dims]int16
				for d := 0; d < dataset.Dims; d++ {
					v[d] = idx.BlockData[b*blockStride+d*blockLanes+lane]
				}
				if !want[v] {
					t.Errorf("cluster %d block %d lane %d: vector %v not in original bucket",
						c, b, lane, v)
					return
				}
			}
		}
	}
}

func assertF32Equal(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: %g != %g", name, i, got[i], want[i])
			return
		}
	}
}

func assertU32Equal(t *testing.T, name string, got, want []uint32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: %d != %d", name, i, got[i], want[i])
			return
		}
	}
}

func assertI16Equal(t *testing.T, name string, got, want []int16) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: %d != %d", name, i, got[i], want[i])
			return
		}
	}
}

func assertU8Equal(t *testing.T, name string, got, want []uint8) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: %d != %d", name, i, got[i], want[i])
			return
		}
	}
}
