//go:build linux

package ivf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
	"golang.org/x/sys/unix"
)

// LoadMmap is the production path: mmap the index file read-only and
// reinterpret it in place as the IVFIndex slices. Two replicas running
// the same image share inodes → kernel keeps a single page-cache copy of
// the 84 MB index across both processes.
//
// Applies MADV_RANDOM (we touch clusters in centroid-sorted order, not
// sequentially), MADV_POPULATE_READ (pre-fault all pages at startup so
// first request doesn't pay a page fault per cluster), MADV_HUGEPAGE
// (84 MB / 2 MB hugepages = 42 TLB entries vs ~20 000 4 KB pages).
//
// See docs/lectures/13-mmap-madvise.md.
func LoadMmap(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	size := int(stat.Size())

	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	// Best-effort advice. Failures are silently ignored.
	_ = unix.Madvise(data, unix.MADV_RANDOM)
	_ = unix.Madvise(data, unix.MADV_POPULATE_READ)
	_ = unix.Madvise(data, unix.MADV_HUGEPAGE)

	// MADV_POPULATE_READ is best-effort and the kernel may defer the
	// page-in. Force every page resident with an explicit read sweep —
	// touch one byte per 4 KB page so the page-cache load is synchronous
	// before we start serving requests. Cheap (~10 ms for 84 MB) and
	// removes the cold-page tail in the first hundred queries.
	prefaultSum := byte(0)
	for i := 0; i < len(data); i += 4096 {
		prefaultSum ^= data[i]
	}
	prefaultBlackhole(prefaultSum)

	// mlock the entire mapping so the kernel can't evict pages under
	// memory pressure (the bot's Mac Mini runs us alongside k6 + dockerd;
	// without mlock we've seen 50–500 µs tails on individual cluster
	// scans when a page faults back in). Requires RLIMIT_MEMLOCK headroom
	// in the container; compose sets memlock soft/hard to the index size.
	// Best-effort: if mlock returns EPERM (limit too low) we silently
	// continue — the MADV_POPULATE_READ + pre-fault still help.
	_ = unix.Mlock(data)

	idx, err := parseMmappedIndex(data)
	if err != nil {
		_ = unix.Munmap(data)
		return nil, err
	}
	ComputeRadii(idx)
	return idx, nil
}

//go:noinline
func prefaultBlackhole(b byte) { prefaultSink = b }

var prefaultSink byte

// parseMmappedIndex constructs IVFIndex slices as unsafe views into the
// mmap'd region. Caller keeps idx.raw alive so the mapping stays valid.
func parseMmappedIndex(data []byte) (*IVFIndex, error) {
	if len(data) < headerSize {
		return nil, fmt.Errorf("file too small for header")
	}
	if string(data[0:4]) != magic {
		return nil, fmt.Errorf("bad magic: %q", data[0:4])
	}
	if v := binary.LittleEndian.Uint32(data[4:8]); v != version {
		return nil, fmt.Errorf("bad version: %d (want %d)", v, version)
	}

	idx := &IVFIndex{
		N:      binary.LittleEndian.Uint32(data[8:12]),
		K:      binary.LittleEndian.Uint32(data[12:16]),
		Blocks: binary.LittleEndian.Uint32(data[16:20]),
		raw:    data,
	}

	off := headerSize
	centroidsLen := dataset.Dims * int(idx.K)
	idx.Centroids = byteToFloat32(data[off : off+centroidsLen*4])
	off += centroidsLen * 4

	offsetsLen := int(idx.K) + 1
	idx.Offsets = byteToUint32(data[off : off+offsetsLen*4])
	off += offsetsLen * 4

	bboxLen := bboxLanes * int(idx.K)
	idx.BboxMin = byteToInt16(data[off : off+bboxLen*2])
	off += bboxLen * 2

	idx.BboxMax = byteToInt16(data[off : off+bboxLen*2])
	off += bboxLen * 2

	labelsLen := int(idx.Blocks) * blockLanes
	idx.Labels = data[off : off+labelsLen]
	off += labelsLen

	blockDataLen := int(idx.Blocks) * blockStride
	idx.BlockData = byteToInt16(data[off : off+blockDataLen*2])
	off += blockDataLen * 2

	if off != len(data) {
		return nil, fmt.Errorf("size mismatch: parsed %d, file %d", off, len(data))
	}
	return idx, nil
}

// Munmap releases the mmap'd region. Optional — the GC won't unmap on
// its own (since raw is a byte slice from unix.Mmap, not a Go-managed
// allocation), so production should hold the IVFIndex for process
// lifetime and skip Munmap.
func (idx *IVFIndex) Munmap() error {
	if idx.raw == nil {
		return nil
	}
	err := unix.Munmap(idx.raw)
	idx.raw = nil
	return err
}

func byteToFloat32(b []byte) []float32 {
	if len(b)%4 != 0 {
		panic("byteToFloat32: not 4-byte aligned")
	}
	n := len(b) / 4
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
}

func byteToUint32(b []byte) []uint32 {
	if len(b)%4 != 0 {
		panic("byteToUint32: not 4-byte aligned")
	}
	n := len(b) / 4
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), n)
}

func byteToInt16(b []byte) []int16 {
	if len(b)%2 != 0 {
		panic("byteToInt16: not 2-byte aligned")
	}
	n := len(b) / 2
	return unsafe.Slice((*int16)(unsafe.Pointer(&b[0])), n)
}

// silence unused-when-debugging warnings
var _ = io.EOF
