package dataset

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	indexMagic   uint32 = 0x52494E48 // "RINH"
	indexVersion uint32 = 4          // v4 = partition + grid (post-IVF revival)
)

// SaveIndex writes:
//   [header] [vectors] [labels] [per-partition grids]
//
// Each grid record is variable-length (cells × ~60 B). The header includes
// total counts for sanity; per-partition record-length is implicit from the
// numCells field.
func (ds *Dataset) SaveIndex(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header: 16 B magic+version+count+_ + 32 starts + 32 counts.
	var hdr [16 + 4*NumPartitions*2]byte
	binary.LittleEndian.PutUint32(hdr[0:4], indexMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], indexVersion)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(ds.Count))
	for p := 0; p < NumPartitions; p++ {
		binary.LittleEndian.PutUint32(hdr[16+p*4:], ds.PartitionStarts[p])
		binary.LittleEndian.PutUint32(hdr[16+NumPartitions*4+p*4:], ds.PartitionCounts[p])
	}
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}

	// Vectors + Labels.
	if _, err := f.Write(int16Bytes(ds.Vectors)); err != nil {
		return err
	}
	if _, err := f.Write(ds.Labels); err != nil {
		return err
	}

	// Per-partition grid records.
	for p := 0; p < NumPartitions; p++ {
		g := &ds.Grids[p]
		// numCells (u32) + bounds (BoundsCount × i16)
		var head [4 + BoundsCount*2]byte
		binary.LittleEndian.PutUint32(head[0:4], g.NumCells)
		for b := 0; b < BoundsCount; b++ {
			binary.LittleEndian.PutUint16(head[4+b*2:], uint16(g.Bounds[b]))
		}
		if _, err := f.Write(head[:]); err != nil {
			return err
		}
		// Cells: count cells × (start u32 + count u32 + bboxMn[Dims] i16 + bboxMx[Dims] i16)
		const cellBytes = 8 + 2*Dims*2 // 64 bytes
		buf := make([]byte, int(g.NumCells)*cellBytes)
		off := 0
		for c := uint32(0); c < g.NumCells; c++ {
			cell := &g.Cells[c]
			binary.LittleEndian.PutUint32(buf[off:], cell.Start)
			binary.LittleEndian.PutUint32(buf[off+4:], cell.Count)
			for d := 0; d < Dims; d++ {
				binary.LittleEndian.PutUint16(buf[off+8+d*2:], uint16(cell.BboxMn[d]))
				binary.LittleEndian.PutUint16(buf[off+8+Dims*2+d*2:], uint16(cell.BboxMx[d]))
			}
			off += cellBytes
		}
		if _, err := f.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

func LoadIndex(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var hdr [16 + 4*NumPartitions*2]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	magic := binary.LittleEndian.Uint32(hdr[0:4])
	version := binary.LittleEndian.Uint32(hdr[4:8])
	if magic != indexMagic {
		return nil, fmt.Errorf("bad magic: 0x%08x", magic)
	}
	if version != indexVersion {
		return nil, fmt.Errorf("bad version: %d (want %d)", version, indexVersion)
	}
	count := int(binary.LittleEndian.Uint64(hdr[8:16]))

	ds := &Dataset{
		Count:   count,
		Vectors: make([]int16, count*Stride),
		Labels:  make([]uint8, count),
	}
	for p := 0; p < NumPartitions; p++ {
		ds.PartitionStarts[p] = binary.LittleEndian.Uint32(hdr[16+p*4:])
		ds.PartitionCounts[p] = binary.LittleEndian.Uint32(hdr[16+NumPartitions*4+p*4:])
	}

	if _, err := io.ReadFull(f, int16Bytes(ds.Vectors)); err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}
	if _, err := io.ReadFull(f, ds.Labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}

	for p := 0; p < NumPartitions; p++ {
		var head [4 + BoundsCount*2]byte
		if _, err := io.ReadFull(f, head[:]); err != nil {
			return nil, fmt.Errorf("read grid header p=%d: %w", p, err)
		}
		numCells := binary.LittleEndian.Uint32(head[0:4])
		g := &ds.Grids[p]
		g.NumCells = numCells
		for b := 0; b < BoundsCount; b++ {
			g.Bounds[b] = int16(binary.LittleEndian.Uint16(head[4+b*2:]))
		}
		if numCells == 0 {
			continue
		}
		const cellBytes = 8 + 2*Dims*2
		buf := make([]byte, int(numCells)*cellBytes)
		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, fmt.Errorf("read cells p=%d: %w", p, err)
		}
		g.Cells = make([]Cell, numCells)
		off := 0
		for c := uint32(0); c < numCells; c++ {
			cell := &g.Cells[c]
			cell.Start = binary.LittleEndian.Uint32(buf[off:])
			cell.Count = binary.LittleEndian.Uint32(buf[off+4:])
			for d := 0; d < Dims; d++ {
				cell.BboxMn[d] = int16(binary.LittleEndian.Uint16(buf[off+8+d*2:]))
				cell.BboxMx[d] = int16(binary.LittleEndian.Uint16(buf[off+8+Dims*2+d*2:]))
			}
			off += cellBytes
		}
	}
	return ds, nil
}

func int16Bytes(s []int16) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafeSlice(&s[0], len(s)*2)
}
