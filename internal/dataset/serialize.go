package dataset

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	indexMagic   uint32 = 0x52494E48 // "RINH"
	indexVersion uint32 = 3          // v3 = flat IVF, block-major data + centroids
)

// SaveIndex writes Blocks + BlockLabels + IVF (clusters + centroid blocks).
func (ds *Dataset) SaveIndex(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var hdr [40]byte
	binary.LittleEndian.PutUint32(hdr[0:4], indexMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], indexVersion)
	binary.LittleEndian.PutUint64(hdr[8:16], uint64(ds.Count))
	binary.LittleEndian.PutUint64(hdr[16:24], uint64(len(ds.Blocks)))
	binary.LittleEndian.PutUint64(hdr[24:32], uint64(len(ds.BlockLabels)))
	binary.LittleEndian.PutUint64(hdr[32:40], uint64(len(ds.IVF.CentroidBlocks)))
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := f.Write(int16Bytes(ds.Blocks)); err != nil {
		return err
	}
	if _, err := f.Write(ds.BlockLabels); err != nil {
		return err
	}
	if _, err := f.Write(int16Bytes(ds.IVF.CentroidBlocks)); err != nil {
		return err
	}

	var nc [4]byte
	binary.LittleEndian.PutUint32(nc[:], uint32(len(ds.IVF.Clusters)))
	if _, err := f.Write(nc[:]); err != nil {
		return err
	}
	for _, c := range ds.IVF.Clusters {
		var meta [16]byte
		binary.LittleEndian.PutUint32(meta[0:4], c.BlockStart)
		binary.LittleEndian.PutUint32(meta[4:8], c.LabelStart)
		binary.LittleEndian.PutUint32(meta[8:12], c.Count)
		binary.LittleEndian.PutUint32(meta[12:16], c.NumBlocks)
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
	return nil
}

func LoadIndex(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var hdr [40]byte
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
	blocksLen := int(binary.LittleEndian.Uint64(hdr[16:24]))
	labelsLen := int(binary.LittleEndian.Uint64(hdr[24:32]))
	centroidsLen := int(binary.LittleEndian.Uint64(hdr[32:40]))

	ds := &Dataset{
		Count:       count,
		Blocks:      make([]int16, blocksLen),
		BlockLabels: make([]uint8, labelsLen),
		IVF:         &IVFIndex{CentroidBlocks: make([]int16, centroidsLen)},
	}
	if _, err := io.ReadFull(f, int16Bytes(ds.Blocks)); err != nil {
		return nil, fmt.Errorf("read blocks: %w", err)
	}
	if _, err := io.ReadFull(f, ds.BlockLabels); err != nil {
		return nil, fmt.Errorf("read blocklabels: %w", err)
	}
	if _, err := io.ReadFull(f, int16Bytes(ds.IVF.CentroidBlocks)); err != nil {
		return nil, fmt.Errorf("read centroidblocks: %w", err)
	}

	var nc [4]byte
	if _, err := io.ReadFull(f, nc[:]); err != nil {
		return nil, fmt.Errorf("read numClusters: %w", err)
	}
	numClusters := int(binary.LittleEndian.Uint32(nc[:]))
	ds.IVF.Clusters = make([]Cluster, numClusters)
	for ci := 0; ci < numClusters; ci++ {
		c := &ds.IVF.Clusters[ci]
		var meta [16]byte
		if _, err := io.ReadFull(f, meta[:]); err != nil {
			return nil, fmt.Errorf("read cluster meta %d: %w", ci, err)
		}
		c.BlockStart = binary.LittleEndian.Uint32(meta[0:4])
		c.LabelStart = binary.LittleEndian.Uint32(meta[4:8])
		c.Count = binary.LittleEndian.Uint32(meta[8:12])
		c.NumBlocks = binary.LittleEndian.Uint32(meta[12:16])
		if _, err := io.ReadFull(f, int16Bytes(c.BboxMin[:])); err != nil {
			return nil, fmt.Errorf("read bboxMin %d: %w", ci, err)
		}
		if _, err := io.ReadFull(f, int16Bytes(c.BboxMax[:])); err != nil {
			return nil, fmt.Errorf("read bboxMax %d: %w", ci, err)
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
