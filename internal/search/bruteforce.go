package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

const (
	k      = 5
	dims   = dataset.Dims
	stride = dataset.Stride
)

// FraudCount is the reference brute-force scanner. Kept for verification of
// the grid path; not used in production.
func FraudCount(query *[stride]int16, ds *dataset.Dataset) int {
	vectors := ds.Vectors
	labels := ds.Labels
	n := ds.Count

	const inf = int64(1 << 62)

	var (
		d0, d1, d2, d3, d4 = inf, inf, inf, inf, inf
		i0, i1, i2, i3, i4 = -1, -1, -1, -1, -1
	)

	q0 := int32(query[0])
	q1 := int32(query[1])
	q2 := int32(query[2])
	q3 := int32(query[3])
	q4 := int32(query[4])
	q5 := int32(query[5])
	q6 := int32(query[6])
	q7 := int32(query[7])
	q8 := int32(query[8])
	q9 := int32(query[9])
	q10 := int32(query[10])
	q11 := int32(query[11])
	q12 := int32(query[12])
	q13 := int32(query[13])

	for i := 0; i < n; i++ {
		base := i * stride

		t := q0 - int32(vectors[base])
		dist := int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q1 - int32(vectors[base+1])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q2 - int32(vectors[base+2])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q3 - int32(vectors[base+3])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q4 - int32(vectors[base+4])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q5 - int32(vectors[base+5])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q6 - int32(vectors[base+6])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q7 - int32(vectors[base+7])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q8 - int32(vectors[base+8])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q9 - int32(vectors[base+9])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q10 - int32(vectors[base+10])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q11 - int32(vectors[base+11])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q12 - int32(vectors[base+12])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}
		t = q13 - int32(vectors[base+13])
		dist += int64(t) * int64(t)
		if dist >= d4 {
			continue
		}

		switch {
		case dist < d0:
			d4, i4 = d3, i3
			d3, i3 = d2, i2
			d2, i2 = d1, i1
			d1, i1 = d0, i0
			d0, i0 = dist, i
		case dist < d1:
			d4, i4 = d3, i3
			d3, i3 = d2, i2
			d2, i2 = d1, i1
			d1, i1 = dist, i
		case dist < d2:
			d4, i4 = d3, i3
			d3, i3 = d2, i2
			d2, i2 = dist, i
		case dist < d3:
			d4, i4 = d3, i3
			d3, i3 = dist, i
		default:
			d4, i4 = dist, i
		}
	}

	frauds := 0
	if i0 >= 0 && labels[i0] == 1 {
		frauds++
	}
	if i1 >= 0 && labels[i1] == 1 {
		frauds++
	}
	if i2 >= 0 && labels[i2] == 1 {
		frauds++
	}
	if i3 >= 0 && labels[i3] == 1 {
		frauds++
	}
	if i4 >= 0 && labels[i4] == 1 {
		frauds++
	}
	return frauds
}
