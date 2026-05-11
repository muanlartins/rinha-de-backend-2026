package search

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// FraudCountPartitioned returns the number of fraud labels among the 5 nearest
// neighbors of query, scanning *only* the partition that the query belongs to.
//
// Correctness: a reference vector in a different partition disagrees with the
// query on at least one of {is_online, card_present, unknown_merchant,
// last_tx_sentinel}. Each such disagreement contributes at least
// `QuantScale^2 = 32000^2` (or `(2*32000)^2` for the sentinel bit-3/4) to the
// squared distance — orders of magnitude larger than within-partition
// distances. So the true top-5 always lives in the query's own partition,
// provided the partition has ≥5 references (which holds for every partition
// in the official dataset).
func FraudCountPartitioned(query *[dims]int16, ds *dataset.Dataset) int {
	key := dataset.ComputeKey(query)
	start := int(ds.PartitionStarts[key])
	count := int(ds.PartitionCounts[key])
	if count == 0 {
		return 0
	}

	vectors := ds.Vectors
	labels := ds.Labels
	end := start + count

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
	q12 := int32(query[12])
	q13 := int32(query[13])
	// Dims 9, 10, 11 are partition-pinned: q[d] == v[d] for every v in this
	// partition, so they contribute exactly 0 to the squared distance and
	// the inner loop can skip them entirely.
	//
	// Dims 5, 6: if this partition is a sentinel partition (bit 3 or 4 set
	// in the key), they're also pinned to SentinelInt and we skip them. We
	// branch once outside the loop instead of per-vector.
	isSentinel5 := (key & 0x08) != 0
	isSentinel6 := (key & 0x10) != 0

	for i := start; i < end; i++ {
		base := i * dims

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
		if !isSentinel5 {
			t = q5 - int32(vectors[base+5])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
		}
		if !isSentinel6 {
			t = q6 - int32(vectors[base+6])
			dist += int64(t) * int64(t)
			if dist >= d4 {
				continue
			}
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
