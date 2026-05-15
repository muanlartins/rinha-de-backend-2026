package specialist

import (
	"sort"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// Build constructs a SpecialistIndex from a flat dataset.Dataset (the
// same shape we feed to internal/ivf.Build). Algorithm:
//
//  1. Compute PartitionKey for every reference; bucket by key.
//  2. For each non-empty partition, build a balanced KD-tree by
//     recursive median-split on the widest-range dimension.
//  3. Leaf nodes hold contiguous blocks of LANES=8 vectors in the
//     same dim-major layout the AVX2 kernel expects.
//  4. Sort partitions by population descending so the most-likely
//     same-key partition is at a predictable index (cosmetic; key
//     lookup is by content, not position).
//
// Determinism: the seed is fixed at build time via stable sort orders.
// Identical input produces identical output across runs.
func Build(ds *dataset.Dataset) *SpecialistIndex {
	N := ds.Count

	// Step 1: bucket references by partition key.
	buckets := make(map[uint32][]uint32, 256)
	var v [dataset.Dims]int16
	for i := 0; i < N; i++ {
		base := i * dataset.Stride
		for d := 0; d < dataset.Dims; d++ {
			v[d] = ds.Vectors[base+d]
		}
		key := PartitionKey(&v)
		buckets[key] = append(buckets[key], uint32(i))
	}

	// Step 2: sort keys (deterministic ordering).
	keys := make([]uint32, 0, len(buckets))
	for k := range buckets {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	idx := &SpecialistIndex{
		N:         uint32(N),
		PartCount: uint32(len(keys)),
	}

	// Working state for the builder.
	var allBlocks [][dataset.Dims * LANES]int16
	var allLabels [][LANES]uint8
	var nodes []Node

	// Step 3: build a KD-tree per partition.
	for _, key := range keys {
		members := buckets[key]
		root := buildKDTree(ds, members, &allBlocks, &allLabels, &nodes)
		rootNode := nodes[root]
		idx.Partitions = append(idx.Partitions, Partition{
			Key:  key,
			Root: uint32(root),
			Min:  rootNode.Min,
			Max:  rootNode.Max,
		})
	}

	idx.Nodes = nodes
	idx.NodeCount = uint32(len(nodes))
	idx.BlockCount = uint32(len(allBlocks))

	// Flatten blocks and labels into a single contiguous slice. Block
	// b's vector at dim d, lane l lives at:
	//   BlockData[b*Dims*LANES + d*LANES + l]
	// which is exactly what kernel.ScanBlock8AVX2 expects.
	idx.BlockData = make([]int16, idx.BlockCount*uint32(dataset.Dims)*uint32(LANES))
	idx.Labels = make([]uint8, idx.BlockCount*uint32(LANES))
	for b, blk := range allBlocks {
		copy(idx.BlockData[b*dataset.Dims*LANES:], blk[:])
		copy(idx.Labels[b*LANES:], allLabels[b][:])
	}

	return idx
}

// buildKDTree recursively partitions members by widest-dim median.
// Returns the index into nodes[] for this subtree's root.
//
// Leaves (members ≤ LeafSize) materialize ceil(len/LANES) blocks in
// allBlocks and store the starting block index. Internal nodes recurse
// on the two halves.
func buildKDTree(
	ds *dataset.Dataset,
	members []uint32,
	allBlocks *[][dataset.Dims * LANES]int16,
	allLabels *[][LANES]uint8,
	nodes *[]Node,
) int {
	// Compute bbox over members.
	var nmin, nmax [16]int16
	for d := 0; d < 16; d++ {
		nmin[d] = 32767
		nmax[d] = -32768
	}
	for _, m := range members {
		base := int(m) * dataset.Stride
		for d := 0; d < dataset.Dims; d++ {
			v := ds.Vectors[base+d]
			if v < nmin[d] {
				nmin[d] = v
			}
			if v > nmax[d] {
				nmax[d] = v
			}
		}
	}
	// Pad lanes 14, 15 to zero (matches the kernel's expectation that
	// padded lanes contribute zero to the squared-distance sum).
	nmin[14], nmin[15] = 0, 0
	nmax[14], nmax[15] = 0, 0

	nodeIdx := len(*nodes)
	*nodes = append(*nodes, Node{Left: -1, Right: -1, Min: nmin, Max: nmax})

	if len(members) <= LeafSize {
		// Leaf: materialize blocks.
		startBlock := uint32(len(*allBlocks))
		blocks := (len(members) + LANES - 1) / LANES

		for b := 0; b < blocks; b++ {
			var blk [dataset.Dims * LANES]int16
			var lbls [LANES]uint8
			for l := 0; l < LANES; l++ {
				i := b*LANES + l
				if i < len(members) {
					m := members[i]
					base := int(m) * dataset.Stride
					for d := 0; d < dataset.Dims; d++ {
						blk[d*LANES+l] = ds.Vectors[base+d]
					}
					lbls[l] = ds.Labels[int(m)]
				} else {
					// Phantom lane: set dims to int16 max so the
					// squared distance is always large (effectively
					// excludes this lane from the top-K). Labels 0.
					for d := 0; d < dataset.Dims; d++ {
						blk[d*LANES+l] = 32767
					}
					lbls[l] = 0
				}
			}
			*allBlocks = append(*allBlocks, blk)
			*allLabels = append(*allLabels, lbls)
		}

		// Update the freshly-pushed node entry.
		(*nodes)[nodeIdx].BlockStart = startBlock
		(*nodes)[nodeIdx].Len = uint32(len(members))
		return nodeIdx
	}

	// Internal node: split on widest dim, recurse.
	splitDim := widestDim(&nmin, &nmax)

	// Stable sort on the split dim.
	sort.SliceStable(members, func(i, j int) bool {
		ai := int(members[i])*dataset.Stride + splitDim
		bi := int(members[j])*dataset.Stride + splitDim
		return ds.Vectors[ai] < ds.Vectors[bi]
	})

	mid := len(members) / 2
	left := buildKDTree(ds, members[:mid], allBlocks, allLabels, nodes)
	right := buildKDTree(ds, members[mid:], allBlocks, allLabels, nodes)

	// Note: nodes slice may have grown and reallocated during the
	// recursive calls. nodeIdx still indexes the correct slot in the
	// final slice.
	(*nodes)[nodeIdx].Left = int32(left)
	(*nodes)[nodeIdx].Right = int32(right)
	// Internal nodes have no blocks of their own.
	(*nodes)[nodeIdx].BlockStart = 0
	(*nodes)[nodeIdx].Len = 0

	return nodeIdx
}

// widestDim returns the dimension index with the largest min/max
// spread. Tie-break by lowest index. Returns 0..13 (Dims-1).
func widestDim(nmin, nmax *[16]int16) int {
	best := 0
	bestWidth := int32(nmax[0]) - int32(nmin[0])
	for d := 1; d < dataset.Dims; d++ {
		w := int32(nmax[d]) - int32(nmin[d])
		if w > bestWidth {
			bestWidth = w
			best = d
		}
	}
	return best
}
