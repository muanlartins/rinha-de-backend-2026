package specialist

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// On-disk format for the specialist index. All little-endian, all
// integer types unsigned where possible.
//
//   Magic        [8]byte  = "SPCST001"
//   Version      uint32
//   QuantScale   uint32   (must match dataset.QuantScale at runtime)
//   N            uint32   (total reference count, before padding)
//   PartCount    uint32   (number of populated partitions, ≤ MaxPartitions)
//   NodeCount    uint32   (total KD-tree nodes across all partitions)
//   BlockCount   uint32   (total blocks of LANES=8 vectors)
//   reserved     [12]byte (alignment + future use)
//
//   Partition directory:  PartCount × PartitionEntry
//     key       uint32
//     root      uint32 (index into Nodes[])
//     min       [16]int16  (14 real + 2 zero pad)
//     max       [16]int16
//   sizeof(PartitionEntry) = 4 + 4 + 32 + 32 = 72 B
//
//   Node directory:  NodeCount × NodeEntry
//     left      int32  (-1 = leaf)
//     right     int32  (-1 = leaf)
//     blockStart uint32 (starting block index in Blocks[])
//     len       uint32 (number of vectors in this leaf; 0 for internal nodes)
//     min       [16]int16
//     max       [16]int16
//   sizeof(NodeEntry) = 4+4+4+4+32+32 = 80 B
//
//   Blocks:  BlockCount × (Dims × LANES) int16  (block-major: [d][lane])
//   Labels:  BlockCount × LANES uint8
//
// On a 1M ref dataset with leaf_size=64: ~16k leaves, ~32k nodes total,
// ~125k blocks. Sizes: ~2.5MB nodes + ~28MB blocks + ~1MB labels.

const (
	Magic        = "SPCST001"
	Version      = uint32(1)
	HeaderSize   = 48
	PartEntrySize = 72
	NodeEntrySize = 80
	LANES        = 8

	// LeafSize is the threshold for stopping KD-tree recursion. fksegundo
	// uses around 64-128 in practice. Smaller leaves = more pruning, more
	// nodes; larger leaves = fewer nodes, less pruning. The right value
	// depends on (a) block-scan cost in the kernel and (b) bbox-LB cost
	// per node. Tunable; calibrate empirically.
	LeafSize = 128
)

// Partition is one entry in the partition directory.
type Partition struct {
	Key  uint32
	Root uint32
	Min  [16]int16
	Max  [16]int16
}

// Node is one entry in the KD-tree node directory.
type Node struct {
	Left, Right int32
	BlockStart  uint32
	Len         uint32
	Min, Max    [16]int16
}

// SpecialistIndex is the runtime representation of the on-disk format.
// All slices alias into the same backing []byte when LoadMmap is used;
// the index file is mmap-able read-only with no parsing beyond pointer
// arithmetic over fixed-size entries.
type SpecialistIndex struct {
	N          uint32
	PartCount  uint32
	NodeCount  uint32
	BlockCount uint32

	Partitions []Partition
	Nodes      []Node
	BlockData  []int16 // length = BlockCount * Dims * LANES
	Labels     []uint8 // length = BlockCount * LANES

	raw []byte // backing buffer (mmap or heap)
}

// PartitionByKey returns the index into Partitions[] for the given key,
// or -1 if the key has no populated partition. Linear scan — PartCount
// is small (50-150 in practice) so this is fast.
func (idx *SpecialistIndex) PartitionByKey(key uint32) int32 {
	for i := range idx.Partitions {
		if idx.Partitions[i].Key == key {
			return int32(i)
		}
	}
	return -1
}

// Serialize writes the index to w in the on-disk format described in
// the package doc. Used by cmd/build-specialist at Docker build time.
func (idx *SpecialistIndex) Serialize(w io.Writer) error {
	var header [HeaderSize]byte
	copy(header[0:8], Magic)
	binary.LittleEndian.PutUint32(header[8:12], Version)
	binary.LittleEndian.PutUint32(header[12:16], uint32(dataset.QuantScale))
	binary.LittleEndian.PutUint32(header[16:20], idx.N)
	binary.LittleEndian.PutUint32(header[20:24], idx.PartCount)
	binary.LittleEndian.PutUint32(header[24:28], idx.NodeCount)
	binary.LittleEndian.PutUint32(header[28:32], idx.BlockCount)
	// reserved [32:48]

	if _, err := w.Write(header[:]); err != nil {
		return err
	}

	// Partitions.
	var partBuf [PartEntrySize]byte
	for i := uint32(0); i < idx.PartCount; i++ {
		p := &idx.Partitions[i]
		binary.LittleEndian.PutUint32(partBuf[0:4], p.Key)
		binary.LittleEndian.PutUint32(partBuf[4:8], p.Root)
		writeI16Array(partBuf[8:40], &p.Min)
		writeI16Array(partBuf[40:72], &p.Max)
		if _, err := w.Write(partBuf[:]); err != nil {
			return err
		}
	}

	// Nodes.
	var nodeBuf [NodeEntrySize]byte
	for i := uint32(0); i < idx.NodeCount; i++ {
		n := &idx.Nodes[i]
		binary.LittleEndian.PutUint32(nodeBuf[0:4], uint32(n.Left))
		binary.LittleEndian.PutUint32(nodeBuf[4:8], uint32(n.Right))
		binary.LittleEndian.PutUint32(nodeBuf[8:12], n.BlockStart)
		binary.LittleEndian.PutUint32(nodeBuf[12:16], n.Len)
		writeI16Array(nodeBuf[16:48], &n.Min)
		writeI16Array(nodeBuf[48:80], &n.Max)
		if _, err := w.Write(nodeBuf[:]); err != nil {
			return err
		}
	}

	// Blocks.
	blockBytes := idx.BlockCount * uint32(dataset.Dims) * uint32(LANES) * 2
	buf := make([]byte, blockBytes)
	for i, v := range idx.BlockData {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}
	if _, err := w.Write(buf); err != nil {
		return err
	}

	// Labels.
	if _, err := w.Write(idx.Labels); err != nil {
		return err
	}
	return nil
}

// Load deserializes from r into a fresh SpecialistIndex backed by heap
// memory. For mmap-backed loading, see LoadMmap (linux-only) in
// mmap_linux.go.
func Load(r io.Reader) (*SpecialistIndex, error) {
	all, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return parseInto(all)
}

func parseInto(raw []byte) (*SpecialistIndex, error) {
	if len(raw) < HeaderSize {
		return nil, fmt.Errorf("specialist: file too short (%d bytes)", len(raw))
	}
	if string(raw[0:8]) != Magic {
		return nil, fmt.Errorf("specialist: bad magic %q", raw[0:8])
	}
	ver := binary.LittleEndian.Uint32(raw[8:12])
	if ver != Version {
		return nil, fmt.Errorf("specialist: version %d unsupported", ver)
	}
	scale := binary.LittleEndian.Uint32(raw[12:16])
	if scale != uint32(dataset.QuantScale) {
		return nil, fmt.Errorf("specialist: index built with QuantScale=%d but runtime is %d",
			scale, dataset.QuantScale)
	}
	n := binary.LittleEndian.Uint32(raw[16:20])
	partCount := binary.LittleEndian.Uint32(raw[20:24])
	nodeCount := binary.LittleEndian.Uint32(raw[24:28])
	blockCount := binary.LittleEndian.Uint32(raw[28:32])

	off := HeaderSize
	idx := &SpecialistIndex{
		N:          n,
		PartCount:  partCount,
		NodeCount:  nodeCount,
		BlockCount: blockCount,
		raw:        raw,
	}

	// Parse partitions.
	idx.Partitions = make([]Partition, partCount)
	for i := uint32(0); i < partCount; i++ {
		p := &idx.Partitions[i]
		p.Key = binary.LittleEndian.Uint32(raw[off : off+4])
		p.Root = binary.LittleEndian.Uint32(raw[off+4 : off+8])
		readI16Array(raw[off+8:off+40], &p.Min)
		readI16Array(raw[off+40:off+72], &p.Max)
		off += PartEntrySize
	}

	// Parse nodes.
	idx.Nodes = make([]Node, nodeCount)
	for i := uint32(0); i < nodeCount; i++ {
		n := &idx.Nodes[i]
		n.Left = int32(binary.LittleEndian.Uint32(raw[off : off+4]))
		n.Right = int32(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		n.BlockStart = binary.LittleEndian.Uint32(raw[off+8 : off+12])
		n.Len = binary.LittleEndian.Uint32(raw[off+12 : off+16])
		readI16Array(raw[off+16:off+48], &n.Min)
		readI16Array(raw[off+48:off+80], &n.Max)
		off += NodeEntrySize
	}

	// Blocks: BlockCount * Dims * LANES int16
	blocksLen := int(blockCount) * dataset.Dims * LANES
	blocksBytes := blocksLen * 2
	if off+blocksBytes > len(raw) {
		return nil, fmt.Errorf("specialist: blocks truncated (need %d more)", off+blocksBytes-len(raw))
	}
	idx.BlockData = make([]int16, blocksLen)
	for i := 0; i < blocksLen; i++ {
		idx.BlockData[i] = int16(binary.LittleEndian.Uint16(raw[off : off+2]))
		off += 2
	}

	// Labels: BlockCount * LANES uint8
	labelsLen := int(blockCount) * LANES
	if off+labelsLen > len(raw) {
		return nil, fmt.Errorf("specialist: labels truncated")
	}
	idx.Labels = make([]uint8, labelsLen)
	copy(idx.Labels, raw[off:off+labelsLen])

	return idx, nil
}

func writeI16Array(dst []byte, src *[16]int16) {
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint16(dst[i*2:], uint16(src[i]))
	}
}

func readI16Array(src []byte, dst *[16]int16) {
	for i := 0; i < 16; i++ {
		dst[i] = int16(binary.LittleEndian.Uint16(src[i*2:]))
	}
}
