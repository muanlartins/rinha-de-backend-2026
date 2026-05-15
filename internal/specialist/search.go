package specialist

import (
	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"github.com/muanlartins/rinha-de-backend-2026/internal/kernel"
)

// kernelSafety mirrors search.kernelSafety — adds margin to the f32
// worst-distance threshold passed into the AVX2 kernel so f32 rounding
// can't prune a candidate whose true i64 distance is within margin.
const kernelSafety = 65536

// TreeStackCap is the max recursion depth we cope with during search.
// For partitions with up to ~10^6 members and LeafSize=128, tree depth
// is ⌈log₂(10⁶/128)⌉ ≈ 13. 64 is comfortable.
const TreeStackCap = 64

// SearchScratch is the per-query workspace used by FraudCount. Allocate
// from a sync.Pool — every field is touched on the hot path.
//
// Sizes:
//   PartBounds   384×16 B = 6 KB
//   StackNodes   64×4 B   = 256 B
//   StackBounds  64×4 B   = 256 B
//   Top5         ~80 B (5×i64 + 5×u8)
//   BlockSum     8×4 B    = 32 B
type SearchScratch struct {
	// One entry per partition with the bbox-LB and the partition idx,
	// used to sort partitions by ascending LB after the same-key one
	// has been processed.
	PartBounds [MaxPartitions]struct {
		Bound float32
		Idx   uint32
	}
	StackNodes  [TreeStackCap]uint32
	StackBounds [TreeStackCap]float32
	Top         Top5
	BlockSum    [LANES]float32
}

// Top5 is a local copy of search.Top5 — kept here to avoid an import
// cycle while we prototype. We can extract to a shared package later if
// we ship Phase 36 to production.
type Top5 struct {
	dist  [5]int64
	label [5]uint8
}

func (t *Top5) Reset() {
	for i := 0; i < 5; i++ {
		t.dist[i] = 1 << 62
		t.label[i] = 0
	}
}

func (t *Top5) WorstI64() int64 { return t.dist[4] }

func (t *Top5) WorstF32() float32 {
	return float32(t.dist[4]) + kernelSafety
}

func (t *Top5) InsertI64(d int64, l uint8) {
	if d >= t.dist[4] {
		return
	}
	pos := 4
	for pos > 0 && t.dist[pos-1] > d {
		t.dist[pos] = t.dist[pos-1]
		t.label[pos] = t.label[pos-1]
		pos--
	}
	t.dist[pos] = d
	t.label[pos] = l
}

func (t *Top5) FraudCount() uint8 {
	return t.label[0] + t.label[1] + t.label[2] + t.label[3] + t.label[4]
}

// FraudCount runs the specialist search and returns the count of frauds
// (0..5) among the top-5 nearest references to qi.
//
//   qi:  int16 quantized query (length 14)
//   qf:  f32 view of the same data (caller-cached for kernel + bbox-LB)
//   idx: the loaded SpecialistIndex
//   sc:  per-query scratch; not shared across goroutines
//
// Algorithm (KeyFirst mode):
//
//   1. Compute query's partition key (O(1)).
//   2. Find the matching-key partition; walk its KD-tree first to
//      warm-start the top-5. This is the "specialist" — references
//      sharing the same categorical pattern are the most likely
//      candidates for true nearest neighbors.
//   3. For every other partition, compute root-bbox-LB.
//   4. Sort the others by ascending bbox-LB.
//   5. Walk each remaining partition's KD-tree in order, pruning by
//      bbox-LB at every node. Stop when bbox-LB >= worst-of-top-5.
//
// This is exact-safe (modulo f32→i64 kernelSafety margin) — bbox-LB is
// a sound lower bound, so any pruned subtree provably can't improve
// top-5. The same-key warm-up gives tight worst-of-top-5 quickly,
// causing aggressive pruning of all other partitions.
func FraudCount(
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *SpecialistIndex,
	sc *SearchScratch,
) uint8 {
	sc.Top.Reset()

	// Pad qf lanes 14, 15 to zero so AABB-LB and kernel work consistently.
	// Caller is responsible for qf[0..13]; padding is in the SearchScratch
	// owning context which has dataset.Dims-sized arrays. We don't pad
	// here; AABBLowerBound below masks to 14 dims internally via
	// per-dim accumulation, and the asm kernel uses lanes 14/15 with
	// zero values when the BlockData is padded to zero diff (lanes
	// 14/15 contribute zero to the squared sum).
	//
	// Actually — our existing AABBLowerBoundF32 expects 16-lane bboxes
	// with lanes 14/15 zeroed. Build.go writes lanes 14/15 = 0 for
	// every node. The caller's qf is 14-lane; we treat the imaginary
	// q[14] = q[15] = 0 implicitly when comparing to bboxes whose
	// lanes 14/15 are 0.

	key := PartitionKey(qi)

	// Step 1: find same-key partition, if any. Process it first.
	sameKeyIdx := idx.PartitionByKey(key)
	if sameKeyIdx >= 0 {
		p := &idx.Partitions[sameKeyIdx]
		// Root bbox-LB. (For the same-key partition, lb is usually 0
		// because the query lies inside the bbox — same categorical
		// pattern means same coarse position.)
		lb := aabbLowerBoundF32(qf, &p.Min, &p.Max)
		if lb < sc.Top.WorstF32() {
			searchKDTree(p.Root, lb, qf, qi, idx, sc)
		}
	}

	// Step 2: collect bbox-LB for every OTHER partition.
	nOther := 0
	for i, p := range idx.Partitions {
		if int32(i) == sameKeyIdx {
			continue
		}
		lb := aabbLowerBoundF32(qf, &p.Min, &p.Max)
		sc.PartBounds[nOther].Bound = lb
		sc.PartBounds[nOther].Idx = uint32(i)
		nOther++
	}

	// Step 3: sort others by ascending bbox-LB (closer partitions first
	// → faster early-exit on subsequent iterations).
	//
	// Hand-rolled insertion sort to avoid `sort.Slice`'s closure
	// allocation. With nOther typically < 200 and most partitions
	// landing in roughly sorted order (categorical similarity →
	// monotone bbox-LB pattern), insertion sort is competitive with
	// quicksort and allocation-free.
	bounds := sc.PartBounds[:nOther]
	for i := 1; i < nOther; i++ {
		cur := bounds[i]
		j := i - 1
		for j >= 0 && bounds[j].Bound > cur.Bound {
			bounds[j+1] = bounds[j]
			j--
		}
		bounds[j+1] = cur
	}

	// Step 4: walk remaining partitions in ascending-LB order. Break
	// as soon as the bbox-LB exceeds worst-of-top-5 (any further
	// partition is also too far away).
	for i := 0; i < nOther; i++ {
		entry := bounds[i]
		worst := sc.Top.WorstF32()
		if entry.Bound >= worst {
			break
		}
		p := &idx.Partitions[entry.Idx]
		searchKDTree(p.Root, entry.Bound, qf, qi, idx, sc)
	}

	return sc.Top.FraudCount()
}

// searchKDTree descends the KD-tree rooted at `root` iteratively,
// using an explicit stack to avoid Go's call overhead. At each internal
// node it computes bbox-LB for both children and visits the nearer one
// first, pushing the farther one onto the stack if it could still
// improve top-5.
func searchKDTree(
	root uint32,
	rootBound float32,
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *SpecialistIndex,
	sc *SearchScratch,
) {
	stackLen := 0
	current := root
	currentBound := rootBound

	for {
		// Re-check bound against current worst (worst may have tightened
		// since this node was pushed).
		worst := sc.Top.WorstF32()
		if currentBound < worst {
			node := &idx.Nodes[current]
			if node.Left < 0 || node.Right < 0 {
				// Leaf: scan its blocks.
				scanLeaf(node, qf, qi, idx, sc)
			} else {
				// Internal: compute lb for both children, visit near first.
				lNode := &idx.Nodes[node.Left]
				rNode := &idx.Nodes[node.Right]
				lb := aabbLowerBoundF32(qf, &lNode.Min, &lNode.Max)
				rb := aabbLowerBoundF32(qf, &rNode.Min, &rNode.Max)

				var nearIdx uint32
				var nearBound, farBound float32
				var farIdx uint32
				if lb <= rb {
					nearIdx, nearBound = uint32(node.Left), lb
					farIdx, farBound = uint32(node.Right), rb
				} else {
					nearIdx, nearBound = uint32(node.Right), rb
					farIdx, farBound = uint32(node.Left), lb
				}

				// Push far child if it could still improve.
				if farBound < worst && stackLen < TreeStackCap {
					sc.StackNodes[stackLen] = farIdx
					sc.StackBounds[stackLen] = farBound
					stackLen++
				}

				// Descend into near child.
				if nearBound < worst {
					current = nearIdx
					currentBound = nearBound
					continue
				}
			}
		}

		// Pop next from stack, or terminate.
		if stackLen == 0 {
			return
		}
		stackLen--
		current = sc.StackNodes[stackLen]
		currentBound = sc.StackBounds[stackLen]
	}
}

// scanLeaf walks the contiguous blocks of LANES=8 vectors in this leaf,
// passes each block to the AVX2 kernel for f32 squared-distance, then
// re-ranks alive lanes via exact i64 distance to feed Top5.InsertI64.
// Identical pattern to the IVF scanCluster — same kernel, same Top5.
func scanLeaf(
	node *Node,
	qf *[dataset.Dims]float32,
	qi *[dataset.Dims]int16,
	idx *SpecialistIndex,
	sc *SearchScratch,
) {
	const blockStride = dataset.Dims * LANES
	startBlock := node.BlockStart
	// A leaf has ceil(len / LANES) blocks.
	numBlocks := (node.Len + LANES - 1) / LANES

	for b := uint32(0); b < numBlocks; b++ {
		blockBase := int(startBlock+b) * blockStride
		worstF32 := sc.Top.WorstF32()
		alive := kernel.ScanBlock8AVX2(
			&qf[0],
			&idx.BlockData[blockBase],
			worstF32,
			&sc.BlockSum,
		)
		if !alive {
			continue
		}
		labelBase := int(startBlock+b) * LANES
		for lane := 0; lane < LANES; lane++ {
			if sc.BlockSum[lane] >= worstF32 {
				continue
			}
			ssd := exactI64Dist(qi, &idx.BlockData[blockBase], lane)
			sc.Top.InsertI64(ssd, idx.Labels[labelBase+lane])
		}
	}
}

// exactI64Dist computes the i64 squared Euclidean distance between qi
// and the lane-th vector in the given block. Block layout is dim-major:
// block[d*LANES + lane].
func exactI64Dist(qi *[dataset.Dims]int16, block *int16, lane int) int64 {
	var ssd int64
	blockSlice := (*[dataset.Dims * LANES]int16)(ptr(block))
	for d := 0; d < dataset.Dims; d++ {
		diff := int64(qi[d]) - int64(blockSlice[d*LANES+lane])
		ssd += diff * diff
	}
	return ssd
}

// aabbLowerBoundF32 computes the squared-distance lower bound from q
// to the bounding box [min, max]. Per dimension, if q[d] is inside
// [min[d], max[d]] the contribution is 0; otherwise it's the squared
// distance from q[d] to the nearest box edge.
//
// We use f32 throughout; the safety margin in WorstF32() compensates
// for f32 rounding. Identical algorithm to search.AABBLowerBoundF32 in
// the IVF code — we keep a local copy to avoid the import cycle.
func aabbLowerBoundF32(q *[dataset.Dims]float32, min, max *[16]int16) float32 {
	var sum float32
	for d := 0; d < dataset.Dims; d++ {
		qd := q[d]
		mn := float32(min[d])
		mx := float32(max[d])
		var diff float32
		switch {
		case qd < mn:
			diff = mn - qd
		case qd > mx:
			diff = qd - mx
		default:
			continue
		}
		sum += diff * diff
	}
	return sum
}
