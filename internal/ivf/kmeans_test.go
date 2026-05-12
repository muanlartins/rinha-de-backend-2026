package ivf

import (
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

var (
	loadOnce sync.Once
	loadedDS *dataset.Dataset
)

func loadDataset(tb testing.TB) *dataset.Dataset {
	loadOnce.Do(func() {
		path := "../../references/rinha-official/resources/references.json.gz"
		ds, err := dataset.LoadFromGzipJSON(path)
		if err != nil {
			tb.Logf("references unavailable: %v", err)
			return
		}
		loadedDS = ds
	})
	if loadedDS == nil {
		tb.Skip("references unavailable")
	}
	return loadedDS
}

// TestKMeansConvergesOnSample is the smallest sanity check: pull a small
// sample, train on it, and assert inertia is strictly decreasing across
// iterations. If it isn't, Lloyd's is broken.
func TestKMeansConvergesOnSample(t *testing.T) {
	ds := loadDataset(t)
	if ds == nil {
		return
	}

	rng := rand.New(rand.NewPCG(rngSeed[0], rngSeed[1]))
	sample := drawSample(ds.Vectors, ds.Count, SampleSize, rng)

	var centroids [K][dataset.Dims]float32
	initKMeansPlusPlus(sample, &centroids, rng)

	prevInertia := -1.0
	for iter := 0; iter < MaxIters; iter++ {
		assign := assignSample(sample, &centroids)
		inertia := Inertia(sample, assign, &centroids)
		updateCentroids(sample, assign, &centroids)

		if prevInertia >= 0 && inertia > prevInertia*1.0001 {
			t.Fatalf("iter %d: inertia %g > previous %g — Lloyd not decreasing",
				iter, inertia, prevInertia)
		}
		prevInertia = inertia
	}
	t.Logf("final inertia after %d iters: %g", MaxIters, prevInertia)
}

// TestKMeansDeterministic confirms that two runs with the same seed produce
// byte-identical centroids. This is the property that lets us bake the
// index into the Docker image without surprise score variance.
func TestKMeansDeterministic(t *testing.T) {
	ds := loadDataset(t)
	if ds == nil {
		return
	}

	c1 := TrainKMeans(ds.Vectors, ds.Count)
	c2 := TrainKMeans(ds.Vectors, ds.Count)

	for c := 0; c < K; c++ {
		for d := 0; d < dataset.Dims; d++ {
			if c1[c][d] != c2[c][d] {
				t.Fatalf("centroid %d dim %d differs across runs: %g vs %g",
					c, d, c1[c][d], c2[c][d])
			}
		}
	}
}

// TestAssignAllPopulatesAllClusters confirms that the full-data assignment
// reaches every cluster at least once. An empty cluster is fine in theory
// (k-means++ can produce them on pathological data), but more than a handful
// of empties signals a training bug.
func TestAssignAllPopulatesAllClusters(t *testing.T) {
	ds := loadDataset(t)
	if ds == nil {
		return
	}

	centroids := TrainKMeans(ds.Vectors, ds.Count)
	assign := AssignAll(ds.Vectors, ds.Count, &centroids)

	var counts [K]uint32
	for _, c := range assign {
		counts[c]++
	}
	empty := 0
	for c := 0; c < K; c++ {
		if counts[c] == 0 {
			empty++
		}
	}
	if empty > K/100 {
		t.Fatalf("too many empty clusters: %d (>%d allowed)", empty, K/100)
	}
	t.Logf("empty clusters: %d / %d", empty, K)

	min, max := counts[0], counts[0]
	for c := 0; c < K; c++ {
		if counts[c] < min {
			min = counts[c]
		}
		if counts[c] > max {
			max = counts[c]
		}
	}
	avg := uint32(ds.Count) / K
	t.Logf("cluster sizes: min=%d max=%d avg=%d", min, max, avg)
	if max > avg*20 {
		t.Fatalf("hot cluster has %d vectors (>20x avg %d) — too unbalanced", max, avg)
	}
}
