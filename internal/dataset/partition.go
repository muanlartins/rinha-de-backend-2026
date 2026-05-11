package dataset

const NumPartitions = 32

// Partition key bit layout (matches the reference QRust implementation so the
// dataset and query agree on every vector's home partition):
//
//   bit 0: query[9]  != 0   (is_online)
//   bit 1: query[10] != 0   (card_present)
//   bit 2: query[11] != 0   (unknown_merchant)
//   bit 3: query[5]  == sentinel (no previous-tx minutes)
//   bit 4: query[6]  == sentinel (no previous-tx km)
func ComputeKey(v *[Dims]int16) uint8 {
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
	base := i * Dims
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

// Partition reorders the dataset's vectors and labels in-place so that all
// vectors with the same partition key live in a contiguous block, and fills
// in PartitionStarts/PartitionCounts. Counting sort, O(N * Dims).
func (ds *Dataset) Partition() {
	// Count.
	counts := [NumPartitions]uint32{}
	for i := 0; i < ds.Count; i++ {
		counts[computeKeyByIndex(ds.Vectors, i)]++
	}
	ds.PartitionCounts = counts

	// Prefix-sum into start offsets.
	var starts [NumPartitions]uint32
	for i := 1; i < NumPartitions; i++ {
		starts[i] = starts[i-1] + counts[i-1]
	}
	ds.PartitionStarts = starts

	// Compute every vector's partition key, then permute in place using a
	// cycle decomposition. We need the SOURCE map (src[p] = original index of
	// the vector that should end up at position p), so we first build the
	// forward dest map and then invert it.
	keys := make([]uint8, ds.Count)
	for i := 0; i < ds.Count; i++ {
		keys[i] = computeKeyByIndex(ds.Vectors, i)
	}

	cursors := starts
	src := make([]uint32, ds.Count)
	for i := 0; i < ds.Count; i++ {
		k := keys[i]
		// vector i lands at cursors[k], so src[cursors[k]] = i
		src[cursors[k]] = uint32(i)
		cursors[k]++
	}

	// Permute: follow cycles of src[]. At each step, vectors[j] receives
	// vectors[src[j]] (the vector that belongs at j).
	visited := make([]bool, ds.Count)
	var buf [Dims]int16
	for i := 0; i < ds.Count; i++ {
		if visited[i] || src[i] == uint32(i) {
			visited[i] = true
			continue
		}
		copy(buf[:], ds.Vectors[i*Dims:(i+1)*Dims])
		labelBuf := ds.Labels[i]
		keyBuf := keys[i]
		j := uint32(i)
		for {
			visited[j] = true
			next := src[j]
			if next == uint32(i) {
				copy(ds.Vectors[j*Dims:(j+1)*Dims], buf[:])
				ds.Labels[j] = labelBuf
				keys[j] = keyBuf
				break
			}
			copy(ds.Vectors[j*Dims:(j+1)*Dims], ds.Vectors[next*Dims:(next+1)*Dims])
			ds.Labels[j] = ds.Labels[next]
			keys[j] = keys[next]
			j = next
		}
	}
}
