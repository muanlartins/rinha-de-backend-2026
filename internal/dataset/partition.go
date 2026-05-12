package dataset

const NumPartitions = 32

// ComputeKey: 5-bit partition key. See docs/CODE_NOTES.md for the bit layout.
func ComputeKey(v *[Stride]int16) uint8 {
	var key uint8
	if v[9] != 0 {
		key |= 1
	}
	if v[10] != 0 {
		key |= 2
	}
	if v[11] != 0 {
		key |= 4
	}
	if v[5] == SentinelInt {
		key |= 8
	}
	if v[6] == SentinelInt {
		key |= 16
	}
	return key
}

func computeKeyByIndex(vectors []int16, i int) uint8 {
	base := i * Stride
	var key uint8
	if vectors[base+9] != 0 {
		key |= 1
	}
	if vectors[base+10] != 0 {
		key |= 2
	}
	if vectors[base+11] != 0 {
		key |= 4
	}
	if vectors[base+5] == SentinelInt {
		key |= 8
	}
	if vectors[base+6] == SentinelInt {
		key |= 16
	}
	return key
}

func (ds *Dataset) Partition() {
	counts := [NumPartitions]uint32{}
	for i := 0; i < ds.Count; i++ {
		counts[computeKeyByIndex(ds.Vectors, i)]++
	}
	ds.PartitionCounts = counts

	var starts [NumPartitions]uint32
	for i := 1; i < NumPartitions; i++ {
		starts[i] = starts[i-1] + counts[i-1]
	}
	ds.PartitionStarts = starts

	keys := make([]uint8, ds.Count)
	for i := 0; i < ds.Count; i++ {
		keys[i] = computeKeyByIndex(ds.Vectors, i)
	}

	// src is the inverse permutation: src[p] = original index of the vector
	// that belongs at position p. The cycle walker below expects this form
	// (see docs/CODE_NOTES.md "Cycle-sort permutation" for why).
	cursors := starts
	src := make([]uint32, ds.Count)
	for i := 0; i < ds.Count; i++ {
		k := keys[i]
		src[cursors[k]] = uint32(i)
		cursors[k]++
	}

	visited := make([]bool, ds.Count)
	var buf [Stride]int16
	for i := 0; i < ds.Count; i++ {
		if visited[i] || src[i] == uint32(i) {
			visited[i] = true
			continue
		}
		copy(buf[:], ds.Vectors[i*Stride:(i+1)*Stride])
		labelBuf := ds.Labels[i]
		keyBuf := keys[i]
		j := uint32(i)
		for {
			visited[j] = true
			next := src[j]
			if next == uint32(i) {
				copy(ds.Vectors[j*Stride:(j+1)*Stride], buf[:])
				ds.Labels[j] = labelBuf
				keys[j] = keyBuf
				break
			}
			copy(ds.Vectors[j*Stride:(j+1)*Stride], ds.Vectors[next*Stride:(next+1)*Stride])
			ds.Labels[j] = ds.Labels[next]
			keys[j] = keys[next]
			j = next
		}
	}
}
