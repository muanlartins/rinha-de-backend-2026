package search

// topK tracks the 5 nearest neighbors observed so far. Labels are carried
// alongside distances so the final fraud count is a tally over l0..l4
// (no second pass through the labels array).
type topK struct {
	d0, d1, d2, d3, d4         int64
	lab0, lab1, lab2, lab3, lab4 uint8
}

func newTopK() topK {
	const inf = int64(1 << 62)
	return topK{d0: inf, d1: inf, d2: inf, d3: inf, d4: inf}
}
