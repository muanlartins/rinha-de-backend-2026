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

	// Phase 39 — Tier 1.2: mlock to force pages permanently resident.
	// Eliminates any chance of swap or eviction under memory pressure.
	// Our cgroup gives us 165 MB; the 84 MB index easily fits with
	// room for the Go heap. RLIMIT_MEMLOCK is usually 64 MB by default
	// in Docker — if mlock fails, we just skip (madvise+populate_read
	// already gives us residency for steady-state, just not enforced).
	_ = unix.Mlock(data)

	// Phase 39 — Tier 1.3: explicit page-stride read after MADV_WILLNEED /
	// POPULATE_READ. POPULATE_READ schedules the fault but the OS may
	// resolve it lazily; this loop forces every page resident NOW. Used
	// by crepao-da-massa (src/index.hpp warm_pages). Cost: one memory-
	// bandwidth pass over 84 MB at startup (~10 ms). Run before parsing
	// so cache is warm when we build the slice views.
	warmPages(data)

	idx, err := parseMmappedIndex(data)
	if err != nil {
		_ = unix.Munmap(data)
		return nil, err
	}
	ComputeRadii(idx)
	return idx, nil
}

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

// warmPages walks the mmap region with a 4 KB-stride dummy read, forcing
// the kernel page-fault handler to make every page resident NOW (rather
// than lazily on first access). Loads `volatile` style: the accumulator
// is read at the end so the compiler can't optimize the loop away.
//
// Cost: ~10 ms for an 84 MB index (1 byte per 4 KB page = ~20 000 reads,
// each ~500 ns to page-fault and 1 ns to read after fault). One-shot at
// startup. After this, all subsequent accesses hit RAM with zero faults.
func warmPages(data []byte) {
	const pageSize = 4096
	var acc byte
	for i := 0; i < len(data); i += pageSize {
		acc ^= data[i]
	}
	// Read acc into a global so the compiler treats it as having a side
	// effect (otherwise it could optimize the whole loop away under -O).
	warmPagesAccumulator = acc
}

// warmPagesAccumulator is a sink to prevent dead-code elimination of
// warmPages. The actual value is meaningless; only the assignment side
// effect matters.
var warmPagesAccumulator byte

// silence unused-when-debugging warnings
var _ = io.EOF
