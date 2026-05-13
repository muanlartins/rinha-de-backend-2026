package ivf

import (
	"math"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// ComputeRadii fills idx.Radii with the max f32 euclidean distance from each
// cluster centroid to any of its members. Used by the search hot path for
// triangle-inequality pruning: for any member x of cluster c,
//
//	dist(q, x) >= |dist(q, centroid_c) - radius_c|
//
// so a cluster is skippable whenever the squared gap exceeds worst-of-top-5.
// Strictly tighter than AABB-LB on round clusters; complementary on
// elongated ones (we apply both, in order: radius first because it's a
// single sqrt + sub, AABB is 14 lane-wise comparisons).
//
// Phantom lanes (padding in the last block of a cluster) hold int16-max in
// every dim — they inflate the radius slightly, which loses some pruning
// power on the partial last block but is always sound (a bigger radius can
// only weaken the bound, never violate it).
//
// Cost: K * total_members * 14 dims of sub+mul. For K=4096, N=3M, ~42M ops,
// ~30-50ms at startup. Amortised across all queries the index serves.
func ComputeRadii(idx *IVFIndex) {
	idx.Radii = make([]float32, idx.K)
	K := int(idx.K)
	var centroid [dataset.Dims]float32
	for c := 0; c < K; c++ {
		startBlock := idx.Offsets[c]
		endBlock := idx.Offsets[c+1]
		if startBlock == endBlock {
			continue
		}
		// Hoist centroid out of the inner loops. Centroids are SoA
		// (idx.Centroids[d*K + c]), so the natural per-member access
		// pattern is strided and cache-hostile. Loading the 14 floats
		// once per cluster turns 14 strided loads per member into 14
		// sequential local-array reads.
		for d := 0; d < dataset.Dims; d++ {
			centroid[d] = idx.Centroids[d*K+c]
		}
		var maxDistSq float32
		for b := startBlock; b < endBlock; b++ {
			blockBase := int(b) * blockStride
			for lane := 0; lane < blockLanes; lane++ {
				var distSq float32
				for d := 0; d < dataset.Dims; d++ {
					v := float32(idx.BlockData[blockBase+d*blockLanes+lane])
					diff := v - centroid[d]
					distSq += diff * diff
				}
				if distSq > maxDistSq {
					maxDistSq = distSq
				}
			}
		}
		idx.Radii[c] = float32(math.Sqrt(float64(maxDistSq)))
	}
}
