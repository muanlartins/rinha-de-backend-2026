package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// FraudCountBrute scans the entire dataset and returns the exact KNN-5
// fraud count. O(N · D). Used by tests to validate IVF search.
func FraudCountBrute(query *[dataset.Stride]int16, ds *dataset.Dataset) int {
	vectors := ds.Vectors
	labels := ds.Labels
	n := ds.Count

	const inf int64 = 1 << 62
	d0, d1, d2, d3, d4 := inf, inf, inf, inf, inf
	var l0, l1, l2, l3, l4 uint8

	for i := 0; i < n; i++ {
		base := i * dataset.Stride
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
		d0, d1, d2, d3, d4, l0, l1, l2, l3, l4 = bruteInsert(
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

// bruteInsert is the cascading 5-slot top-K shift the brute force uses.
// Distances are kept ascending: d0 <= d1 <= d2 <= d3 <= d4. A new (d, l)
// inserts at the right position via cascaded compare-and-shift.
func bruteInsert(d int64, l uint8, d0, d1, d2, d3, d4 int64, l0, l1, l2, l3, l4 uint8) (int64, int64, int64, int64, int64, uint8, uint8, uint8, uint8, uint8) {
	if d >= d4 {
		return d0, d1, d2, d3, d4, l0, l1, l2, l3, l4
	}
	switch {
	case d < d0:
		return d, d0, d1, d2, d3, l, l0, l1, l2, l3
	case d < d1:
		return d0, d, d1, d2, d3, l0, l, l1, l2, l3
	case d < d2:
		return d0, d1, d, d2, d3, l0, l1, l, l2, l3
	case d < d3:
		return d0, d1, d2, d, d3, l0, l1, l2, l, l3
	default:
		return d0, d1, d2, d3, d, l0, l1, l2, l3, l
	}
}
