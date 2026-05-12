package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// FraudCountBrute scans the entire dataset and returns the exact KNN-5
// fraud count. O(N · D). Used only by tests to validate the grid search.
func FraudCountBrute(query *[stride]int16, ds *dataset.Dataset) int {
	vectors := ds.Vectors
	labels := ds.Labels
	n := ds.Count

	const inf int64 = 1 << 62
	d0, d1, d2, d3, d4 := inf, inf, inf, inf, inf
	var l0, l1, l2, l3, l4 uint8

	for i := 0; i < n; i++ {
		base := i * stride
		var sum int64
		for d := 0; d < dataset.Dims; d++ {
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
		d0, d1, d2, d3, d4, l0, l1, l2, l3, l4 = topKInsert(
			sum, lab, d0, d1, d2, d3, d4, l0, l1, l2, l3, l4)
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
