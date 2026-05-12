package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

const (
	k      = 5
	dims   = dataset.Dims
	stride = dataset.Stride
)

// FraudCount is the exact brute-force scanner over the flat ds.Vectors slab.
// Only callable when ds.Vectors is non-nil (i.e., the dataset was just built
// from JSON, not loaded from the persisted index). Used by tests.
func FraudCount(query *[stride]int16, ds *dataset.Dataset) int {
	if ds.Vectors == nil {
		panic("FraudCount: ds.Vectors is nil — only available after LoadFromGzipJSON, not LoadIndex")
	}
	vectors := ds.Vectors
	labels := ds.Labels
	n := ds.Count

	tk := newTopK()
	d0, d1, d2, d3, d4 := tk.d0, tk.d1, tk.d2, tk.d3, tk.d4
	l0, l1, l2, l3, l4 := tk.lab0, tk.lab1, tk.lab2, tk.lab3, tk.lab4

	for i := 0; i < n; i++ {
		base := i * stride
		var sum int64
		for d := 0; d < stride; d++ {
			t := int64(query[d]) - int64(vectors[base+d])
			sum += t * t
			if sum >= d4 {
				break
			}
		}
		if sum >= d4 {
			continue
		}
		lab := labels[i]
		switch {
		case sum < d0:
			d4, l4 = d3, l3
			d3, l3 = d2, l2
			d2, l2 = d1, l1
			d1, l1 = d0, l0
			d0, l0 = sum, lab
		case sum < d1:
			d4, l4 = d3, l3
			d3, l3 = d2, l2
			d2, l2 = d1, l1
			d1, l1 = sum, lab
		case sum < d2:
			d4, l4 = d3, l3
			d3, l3 = d2, l2
			d2, l2 = sum, lab
		case sum < d3:
			d4, l4 = d3, l3
			d3, l3 = sum, lab
		default:
			d4, l4 = sum, lab
		}
	}

	frauds := 0
	if l0 == 1 {
		frauds++
	}
	if l1 == 1 {
		frauds++
	}
	if l2 == 1 {
		frauds++
	}
	if l3 == 1 {
		frauds++
	}
	if l4 == 1 {
		frauds++
	}
	return frauds
}
