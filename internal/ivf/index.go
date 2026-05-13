package ivf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"unsafe"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// IVFIndex is the runtime representation produced by the offline builder
// and consumed by the search hot path. After serialization round-trip, the
// slices below are views into a single mmap-able byte buffer (when loaded
// via LoadMmap) or independent heap allocations (when built in-process).
type IVFIndex struct {
	// Counts.
	N      uint32 // total vectors
	K      uint32 // clusters (== ivf.K, but stored for header sanity)
	Blocks uint32 // total blocks (== sum over clusters of blocks-per-cluster)

	// Centroids, SoA across all clusters: f32[d*K + c]. Length 14*K.
	Centroids []float32

	// Cluster offsets in block units. Length K+1.
	//   block_start_of_cluster_c  = Offsets[c]
	//   block_count_of_cluster_c  = Offsets[c+1] - Offsets[c]
	Offsets []uint32

	// Per-cluster bounding box, cluster-major with 16-wide padding. Length
	// 16*K each. Lanes 14 and 15 are zero (see lecture 09).
	BboxMin []int16
	BboxMax []int16

	// Per-cluster radius: max f32 euclidean distance from each centroid to
	// any of its members. Computed at load time (ComputeRadii), not
	// serialized. Used by scanCluster for the triangle-inequality LB,
	// strictly tighter than AABB-LB on round clusters. Length K.
	Radii []float32

	// Per-block-lane fraud label. Length Blocks*8. Phantom lanes are zero.
	Labels []uint8

	// Block payload: dim-major panels of 8 vectors each. Length Blocks*14*8.
	// Layout: BlockData[b*112 + d*8 + lane] is dim-d lane of block b.
	BlockData []int16

	// raw is the backing byte buffer when LoadMmap is used. Holding the
	// reference prevents the GC from unmapping the region.
	raw []byte
}

// File format constants. See docs/lectures/09-kmeans-ivf.md § On-disk format.
const (
	magic       = "IVF8"
	version     = uint32(5)
	headerSize  = 32 // bytes
	bboxLanes   = 16 // 14 real + 2 zero pad
	blockLanes  = 8
	blockStride = dataset.Dims * blockLanes // 14*8 = 112 int16 = 224 bytes
)

// Build constructs an IVFIndex from the trained centroids and an assignment
// table mapping each global vector id to its cluster. This is the offline
// driver — it runs once at Docker build time. After Build, the returned
// IVFIndex is ready for Serialize.
func Build(vectors []int16, count int, centroids *[K][dataset.Dims]float32, assign []uint16) (*IVFIndex, error) {
	if len(assign) != count {
		return nil, fmt.Errorf("assign length %d != count %d", len(assign), count)
	}

	idx := &IVFIndex{
		N: uint32(count),
		K: K,
	}

	// 1. Bucket vector ids by cluster, in order. We allocate K small slices.
	//    For 3M vectors and K=4096, average bucket size is ~732, peak maybe
	//    2-3x that. Total memory: ~12 MB for the id slices.
	buckets := make([][]uint32, K)
	for c := range buckets {
		buckets[c] = make([]uint32, 0, count/K+8)
	}
	for i := uint32(0); i < uint32(count); i++ {
		c := assign[i]
		buckets[c] = append(buckets[c], i)
	}

	// 2. Compute total blocks. Each cluster contributes ceil(len/8) blocks.
	var totalBlocks uint32
	idx.Offsets = make([]uint32, K+1)
	for c := 0; c < K; c++ {
		idx.Offsets[c] = totalBlocks
		n := len(buckets[c])
		blocks := uint32((n + blockLanes - 1) / blockLanes)
		totalBlocks += blocks
	}
	idx.Offsets[K] = totalBlocks
	idx.Blocks = totalBlocks

	// 3. Centroids SoA: f32[d*K + c].
	idx.Centroids = make([]float32, dataset.Dims*K)
	for c := 0; c < K; c++ {
		for d := 0; d < dataset.Dims; d++ {
			idx.Centroids[d*K+c] = centroids[c][d]
		}
	}

	// 4. Bbox per cluster, computed from the actual vectors in the bucket.
	idx.BboxMin = make([]int16, bboxLanes*K)
	idx.BboxMax = make([]int16, bboxLanes*K)
	for c := 0; c < K; c++ {
		ids := buckets[c]
		if len(ids) == 0 {
			// Empty cluster: bbox set so LB is "always large" — never gets
			// scanned. We pick a single-point box at +max so any query has
			// lb >= (q-int16max)^2 which dominates worst_top5 by a wide
			// margin. Using cluster-major lane index 0..15 zeroes 14-15.
			for d := 0; d < bboxLanes; d++ {
				idx.BboxMin[c*bboxLanes+d] = int16(32767)
				idx.BboxMax[c*bboxLanes+d] = int16(32767)
			}
			continue
		}
		var mn, mx [dataset.Dims]int16
		for d := 0; d < dataset.Dims; d++ {
			mn[d] = int16(32767)
			mx[d] = int16(-32768)
		}
		for _, id := range ids {
			base := int(id) * dataset.Stride
			for d := 0; d < dataset.Dims; d++ {
				v := vectors[base+d]
				if v < mn[d] {
					mn[d] = v
				}
				if v > mx[d] {
					mx[d] = v
				}
			}
		}
		for d := 0; d < dataset.Dims; d++ {
			idx.BboxMin[c*bboxLanes+d] = mn[d]
			idx.BboxMax[c*bboxLanes+d] = mx[d]
		}
		// Lanes 14 and 15 remain zero from make(); LB contribution will be
		// max(0, 0-0) + max(0, 0-0) = 0 regardless of query, so no effect.
	}

	// 5. Labels per block-lane and the block payload itself.
	idx.Labels = make([]uint8, int(totalBlocks)*blockLanes)
	idx.BlockData = make([]int16, int(totalBlocks)*blockStride)

	// Optional intra-cluster ordering: sort by distance to centroid so the
	// most-likely-to-survive candidates come first in the block stream.
	// This is the "Golden Optimization" (lecture 05 § post-grid revival).
	// It only matters once the SIMD early-exit kernel is in place, but
	// committing it here means phase 15 inherits it for free.
	for c := 0; c < K; c++ {
		ids := buckets[c]
		if len(ids) == 0 {
			continue
		}
		sortIDsByCentroidDist(ids, vectors, &centroids[c])

		blockStart := int(idx.Offsets[c])
		for i, id := range ids {
			b := blockStart + i/blockLanes
			lane := i % blockLanes
			src := int(id) * dataset.Stride
			for d := 0; d < dataset.Dims; d++ {
				idx.BlockData[b*blockStride+d*blockLanes+lane] = vectors[src+d]
			}
			idx.Labels[b*blockLanes+lane] = idx.fraudLabelAt(id)
		}
		// Pad the partial last block with INT16_MAX in unused lanes so
		// phantom distances never beat real candidates.
		used := len(ids) % blockLanes
		if used != 0 {
			b := blockStart + len(ids)/blockLanes
			for lane := used; lane < blockLanes; lane++ {
				for d := 0; d < dataset.Dims; d++ {
					idx.BlockData[b*blockStride+d*blockLanes+lane] = 32767
				}
				// Label stays 0 (zero-padded) — irrelevant because the lane's
				// distance is huge; it can't appear in top-5.
			}
		}
	}

	ComputeRadii(idx)
	return idx, nil
}

// labelLookup is the per-build accessor for the source labels. We do this
// because the assignment phase has already happened and we need to translate
// global ids → fraud bit when writing block-lane labels. The label slice
// lives on the caller's `Dataset` and is closed over by a small adapter.
//
// To keep the API surface small, we attach it as a method on IVFIndex via a
// package-level `labelSource` pointer set by Build. We avoid a global; the
// pointer lives in the IVFIndex struct.
type labelSource struct{ s []uint8 }

var currentLabelSource labelSource

func (idx *IVFIndex) fraudLabelAt(id uint32) uint8 {
	return currentLabelSource.s[id]
}

// SetLabelSource is called by the builder before Build, so block-lane labels
// can be populated as the block stream is materialized. This is a tiny
// piece of mutable global state, but it's only used at index-build time
// (offline, single-threaded), so the simplification is worth it.
func SetLabelSource(labels []uint8) {
	currentLabelSource.s = labels
}

// sortIDsByCentroidDist sorts ids in place by their distance to the given
// centroid (closest first). Stable across builds because we go through
// sort.SliceStable.
func sortIDsByCentroidDist(ids []uint32, vectors []int16, centroid *[dataset.Dims]float32) {
	dists := make([]float64, len(ids))
	for i, id := range ids {
		base := int(id) * dataset.Stride
		s := 0.0
		for d := 0; d < dataset.Dims; d++ {
			diff := float64(vectors[base+d]) - float64(centroid[d])
			s += diff * diff
		}
		dists[i] = s
	}
	sort.SliceStable(ids, func(a, b int) bool { return dists[a] < dists[b] })
}

// Serialize writes the index to w in the v5 binary format described in
// docs/lectures/09-kmeans-ivf.md.
func (idx *IVFIndex) Serialize(w io.Writer) error {
	header := make([]byte, headerSize)
	copy(header[0:4], magic)
	binary.LittleEndian.PutUint32(header[4:8], version)
	binary.LittleEndian.PutUint32(header[8:12], idx.N)
	binary.LittleEndian.PutUint32(header[12:16], idx.K)
	binary.LittleEndian.PutUint32(header[16:20], idx.Blocks)
	binary.LittleEndian.PutUint32(header[20:24], uint32(dataset.QuantScale))
	if _, err := w.Write(header); err != nil {
		return err
	}

	if err := writeFloat32Slice(w, idx.Centroids); err != nil {
		return err
	}
	if err := writeUint32Slice(w, idx.Offsets); err != nil {
		return err
	}
	if err := writeInt16Slice(w, idx.BboxMin); err != nil {
		return err
	}
	if err := writeInt16Slice(w, idx.BboxMax); err != nil {
		return err
	}
	if _, err := w.Write(idx.Labels); err != nil {
		return err
	}
	if err := writeInt16Slice(w, idx.BlockData); err != nil {
		return err
	}
	return nil
}

// SerializeToFile is a convenience wrapper for Serialize.
func (idx *IVFIndex) SerializeToFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return idx.Serialize(f)
}

// Load reads an index from the v5 binary format into independent heap
// allocations. Suitable for tests and for the in-process build path.
//
// Production deploys should use LoadMmap (added in phase 19) so two replicas
// running the same image share kernel page cache.
func Load(r io.Reader) (*IVFIndex, error) {
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(header[0:4]) != magic {
		return nil, fmt.Errorf("bad magic: %q", header[0:4])
	}
	if v := binary.LittleEndian.Uint32(header[4:8]); v != version {
		return nil, fmt.Errorf("bad version: %d (want %d)", v, version)
	}

	idx := &IVFIndex{
		N:      binary.LittleEndian.Uint32(header[8:12]),
		K:      binary.LittleEndian.Uint32(header[12:16]),
		Blocks: binary.LittleEndian.Uint32(header[16:20]),
	}

	idx.Centroids = make([]float32, int(dataset.Dims)*int(idx.K))
	if err := readFloat32Slice(r, idx.Centroids); err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}
	idx.Offsets = make([]uint32, int(idx.K)+1)
	if err := readUint32Slice(r, idx.Offsets); err != nil {
		return nil, fmt.Errorf("read offsets: %w", err)
	}
	idx.BboxMin = make([]int16, bboxLanes*int(idx.K))
	if err := readInt16Slice(r, idx.BboxMin); err != nil {
		return nil, fmt.Errorf("read bbox min: %w", err)
	}
	idx.BboxMax = make([]int16, bboxLanes*int(idx.K))
	if err := readInt16Slice(r, idx.BboxMax); err != nil {
		return nil, fmt.Errorf("read bbox max: %w", err)
	}
	idx.Labels = make([]uint8, int(idx.Blocks)*blockLanes)
	if _, err := io.ReadFull(r, idx.Labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	idx.BlockData = make([]int16, int(idx.Blocks)*blockStride)
	if err := readInt16Slice(r, idx.BlockData); err != nil {
		return nil, fmt.Errorf("read blocks: %w", err)
	}
	ComputeRadii(idx)
	return idx, nil
}

func writeFloat32Slice(w io.Writer, s []float32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := w.Write(b)
	return err
}

func writeUint32Slice(w io.Writer, s []uint32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := w.Write(b)
	return err
}

func writeInt16Slice(w io.Writer, s []int16) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 2*len(s))
	_, err := w.Write(b)
	return err
}

func readFloat32Slice(r io.Reader, s []float32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := io.ReadFull(r, b)
	return err
}

func readUint32Slice(r io.Reader, s []uint32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := io.ReadFull(r, b)
	return err
}

func readInt16Slice(r io.Reader, s []int16) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 2*len(s))
	_, err := io.ReadFull(r, b)
	return err
}
