package dataset

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// indexMagic identifies a valid serialized dataset+IVF index.
const indexMagic uint32 = 0x52494E48 // "RINH"
const indexVersion uint32 = 1

// SaveIndex writes the full dataset (vectors, labels, partitions, IVF) to
// path in a compact binary format. The runtime loads it directly without
// rebuilding from references.json.gz.
func (ds *Dataset) SaveIndex(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var hdr [16]byte
	binary.LittleEndian.PutUint32(hdr[0:4], indexMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], indexVersion)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(ds.Count))
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}

	if _, err := f.Write(int16Bytes(ds.Vectors)); err != nil {
		return err
	}
	if _, err := f.Write(ds.Labels); err != nil {
		return err
	}
	for k := 0; k < NumPartitions; k++ {
		var pbuf [8]byte
		binary.LittleEndian.PutUint32(pbuf[0:4], ds.PartitionStarts[k])
		binary.LittleEndian.PutUint32(pbuf[4:8], ds.PartitionCounts[k])
		if _, err := f.Write(pbuf[:]); err != nil {
			return err
		}
	}

	for k := 0; k < NumPartitions; k++ {
		pg := ds.IVF[k]
		var cbuf [4]byte
		var numClusters uint32
		if pg != nil {
			numClusters = uint32(len(pg.Clusters))
		}
		binary.LittleEndian.PutUint32(cbuf[:], numClusters)
		if _, err := f.Write(cbuf[:]); err != nil {
			return err
		}
		if pg == nil {
			continue
		}
		for _, c := range pg.Clusters {
			if _, err := f.Write(int16Bytes(c.Centroid[:])); err != nil {
				return err
			}
			var meta [8]byte
			binary.LittleEndian.PutUint32(meta[0:4], c.Start)
			binary.LittleEndian.PutUint32(meta[4:8], c.Count)
			if _, err := f.Write(meta[:]); err != nil {
				return err
			}
			if _, err := f.Write(int16Bytes(c.BboxMin[:])); err != nil {
				return err
			}
			if _, err := f.Write(int16Bytes(c.BboxMax[:])); err != nil {
				return err
			}
		}
	}

	return nil
}

// LoadIndex reverses SaveIndex.
func LoadIndex(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var hdr [16]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	magic := binary.LittleEndian.Uint32(hdr[0:4])
	version := binary.LittleEndian.Uint32(hdr[4:8])
	if magic != indexMagic {
		return nil, fmt.Errorf("bad magic: 0x%08x (want 0x%08x)", magic, indexMagic)
	}
	if version != indexVersion {
		return nil, fmt.Errorf("bad version: %d (want %d)", version, indexVersion)
	}
	count := int(binary.LittleEndian.Uint64(hdr[8:16]))

	ds := &Dataset{
		Count:   count,
		Vectors: make([]int16, count*Stride),
		Labels:  make([]uint8, count),
		IVF:     make([]*IVFPartition, NumPartitions),
	}
	if _, err := io.ReadFull(f, int16Bytes(ds.Vectors)); err != nil {
		return nil, fmt.Errorf("read vectors: %w", err)
	}
	if _, err := io.ReadFull(f, ds.Labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}
	for k := 0; k < NumPartitions; k++ {
		var pbuf [8]byte
		if _, err := io.ReadFull(f, pbuf[:]); err != nil {
			return nil, fmt.Errorf("read partition %d: %w", k, err)
		}
		ds.PartitionStarts[k] = binary.LittleEndian.Uint32(pbuf[0:4])
		ds.PartitionCounts[k] = binary.LittleEndian.Uint32(pbuf[4:8])
	}
	for k := 0; k < NumPartitions; k++ {
		var cbuf [4]byte
		if _, err := io.ReadFull(f, cbuf[:]); err != nil {
			return nil, fmt.Errorf("read partition %d numClusters: %w", k, err)
		}
		numClusters := int(binary.LittleEndian.Uint32(cbuf[:]))
		if numClusters == 0 {
			continue
		}
		pg := &IVFPartition{Clusters: make([]Cluster, numClusters)}
		for ci := 0; ci < numClusters; ci++ {
			c := &pg.Clusters[ci]
			if _, err := io.ReadFull(f, int16Bytes(c.Centroid[:])); err != nil {
				return nil, fmt.Errorf("read centroid %d: %w", ci, err)
			}
			var meta [8]byte
			if _, err := io.ReadFull(f, meta[:]); err != nil {
				return nil, fmt.Errorf("read cluster meta %d: %w", ci, err)
			}
			c.Start = binary.LittleEndian.Uint32(meta[0:4])
			c.Count = binary.LittleEndian.Uint32(meta[4:8])
			if _, err := io.ReadFull(f, int16Bytes(c.BboxMin[:])); err != nil {
				return nil, fmt.Errorf("read bboxMin %d: %w", ci, err)
			}
			if _, err := io.ReadFull(f, int16Bytes(c.BboxMax[:])); err != nil {
				return nil, fmt.Errorf("read bboxMax %d: %w", ci, err)
			}
		}
		ds.IVF[k] = pg
	}
	return ds, nil
}

// int16Bytes reinterprets a []int16 as a []byte without copying.
func int16Bytes(s []int16) []byte {
	if len(s) == 0 {
		return nil
	}
	// On all supported platforms, int16 is exactly 2 bytes and the runtime
	// uses native byte order. We accept the host endianness as the on-disk
	// endianness; the file is built and consumed on linux/amd64 only.
	return unsafeSlice(&s[0], len(s)*2)
}
