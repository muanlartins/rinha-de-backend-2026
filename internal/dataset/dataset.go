// Package dataset loads the 3M-vector reference set into memory as int16-
// quantized values.
//
// Streams references.json.gz with a hand-rolled byte-level parser to avoid the
// 300+ MB transient allocation that stdlib's encoding/json incurs on a 3M-entry
// array. Quantizes during the fill pass so the final footprint is just the
// int16 buffer (~84 MB) plus the label byte array (~3 MB).
//
// Quantization: real-valued dims in [0,1] → [0, 32000]. Sentinel -1 → -32000.
// Maximum squared distance per dim within a partition fits in int32, summed
// across 14 dims fits in int64.
package dataset

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"os"
)

const (
	Dims         = 14
	QuantScale   = 32000
	SentinelInt  = -32000
	SentinelReal = -1.0
)

// Dataset is the in-memory reference set.
//
// Vectors is laid out flat: vectors[i*Dims + d] is dim d of vector i.
// Labels[i] is 1 for "fraud", 0 for "legit".
//
// After Partition() runs, vectors with the same partition key live in a
// contiguous block; PartitionStarts[k] and PartitionCounts[k] locate that
// block for partition key k. After BuildGrid() runs, Partitions[k] holds the
// per-partition cell index for grid-based search.
type Dataset struct {
	Vectors         []int16
	Labels          []uint8
	Count           int
	PartitionStarts [NumPartitions]uint32
	PartitionCounts [NumPartitions]uint32
	Partitions      []*Partition
}

// LoadFromGzipJSON streams the official references.json.gz into memory in two
// passes: first counts entries (so we allocate Vectors and Labels exactly),
// then fills them in-place. Per-entry heap allocation is zero.
func LoadFromGzipJSON(path string) (*Dataset, error) {
	count, err := scan(path, nil)
	if err != nil {
		return nil, fmt.Errorf("count: %w", err)
	}

	ds := &Dataset{
		Vectors: make([]int16, count*Dims),
		Labels:  make([]uint8, count),
		Count:   count,
	}

	idx := 0
	got, err := scan(path, func(vec [Dims]float32, isFraud bool) {
		base := idx * Dims
		for d, v := range vec {
			ds.Vectors[base+d] = quantize(v)
		}
		if isFraud {
			ds.Labels[idx] = 1
		}
		idx++
	})
	if err != nil {
		return nil, fmt.Errorf("fill: %w", err)
	}
	if got != count {
		return nil, fmt.Errorf("count mismatch: pass1=%d pass2=%d", count, got)
	}
	return ds, nil
}

// Quantize converts a normalized float dim value to int16. Returns the sentinel
// constant for -1.0; otherwise scales [0,1] to [0, QuantScale] with clamping
// and round-half-up.
func Quantize(v float32) int16 {
	return quantize(v)
}

func quantize(v float32) int16 {
	if v <= SentinelReal+1e-6 && v >= SentinelReal-1e-6 {
		return SentinelInt
	}
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return QuantScale
	}
	return int16(v*QuantScale + 0.5)
}

// scan opens path as gzip+JSON and invokes cb (if non-nil) for every entry.
// Returns the count of entries seen.
func scan(path string, cb func(vec [Dims]float32, isFraud bool)) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return 0, err
	}
	defer gz.Close()

	br := bufio.NewReaderSize(gz, 1<<20)

	count := 0
	const vectorKey = `"vector":[`
	const labelKey = `,"label":"`

	var vec [Dims]float32

	for {
		// Find the start of the next entry's vector array.
		if err := readUntil(br, vectorKey); err != nil {
			if err == io.EOF {
				return count, nil
			}
			return count, fmt.Errorf("scan vector key (entry %d): %w", count, err)
		}

		// 14 comma-separated floats followed by ']'.
		for i := 0; i < Dims; i++ {
			v, last, err := readFloat(br)
			if err != nil {
				return count, fmt.Errorf("scan dim %d (entry %d): %w", i, count, err)
			}
			vec[i] = v
			if i < Dims-1 && last == ']' {
				return count, fmt.Errorf("unexpected ']' at dim %d (entry %d)", i, count)
			}
			if i == Dims-1 && last != ']' {
				return count, fmt.Errorf("expected ']' after dim %d (entry %d), got %q", i, count, last)
			}
		}

		// Label key then label string.
		if err := readUntil(br, labelKey); err != nil {
			return count, fmt.Errorf("scan label key (entry %d): %w", count, err)
		}
		isFraud, err := readLabel(br)
		if err != nil {
			return count, fmt.Errorf("scan label (entry %d): %w", count, err)
		}

		if cb != nil {
			cb(vec, isFraud)
		}
		count++
	}
}

// readUntil consumes bytes until needle is fully matched. The keys we use
// ("\"vector\":[", ",\"label\":\"") have no proper prefix that's also a suffix,
// so we can reset matched on mismatch without backtracking.
func readUntil(br *bufio.Reader, needle string) error {
	nlen := len(needle)
	matched := 0
	for {
		b, err := br.ReadByte()
		if err != nil {
			return err
		}
		if b == needle[matched] {
			matched++
			if matched == nlen {
				return nil
			}
			continue
		}
		matched = 0
		if b == needle[0] {
			matched = 1
		}
	}
}

// readFloat parses a JSON number. Returns the parsed value and the
// terminator byte (e.g., ',' or ']').
func readFloat(br *bufio.Reader) (val float32, last byte, err error) {
	var v float64
	var sign float64 = 1
	state := 0 // 0=start, 1=intpart, 2=fracpart, 3=expsign, 4=exppart

	var frac float64 = 0.1
	var expSign int = 1
	var exp int = 0

	for {
		b, rerr := br.ReadByte()
		if rerr != nil {
			return 0, 0, rerr
		}
		switch {
		case b == '-' && state == 0:
			sign = -1
		case b >= '0' && b <= '9':
			switch state {
			case 0, 1:
				state = 1
				v = v*10 + float64(b-'0')
			case 2:
				v += float64(b-'0') * frac
				frac *= 0.1
			case 3, 4:
				state = 4
				exp = exp*10 + int(b-'0')
			}
		case b == '.' && state == 1:
			state = 2
		case (b == 'e' || b == 'E') && (state == 1 || state == 2):
			state = 3
		case (b == '+' || b == '-') && state == 3:
			if b == '-' {
				expSign = -1
			}
		default:
			if state == 0 {
				return 0, 0, fmt.Errorf("unexpected %q at start of number", b)
			}
			v *= sign
			if state == 4 {
				e := exp * expSign
				if e >= 0 {
					for k := 0; k < e; k++ {
						v *= 10
					}
				} else {
					for k := 0; k < -e; k++ {
						v *= 0.1
					}
				}
			}
			return float32(v), b, nil
		}
	}
}

// readLabel reads a 5-char label ("fraud" or "legit") followed by '"'.
func readLabel(br *bufio.Reader) (bool, error) {
	var buf [5]byte
	if _, err := io.ReadFull(br, buf[:]); err != nil {
		return false, err
	}
	next, err := br.ReadByte()
	if err != nil {
		return false, err
	}
	if next != '"' {
		return false, fmt.Errorf("expected '\"' after label, got %q", next)
	}
	switch buf {
	case [5]byte{'f', 'r', 'a', 'u', 'd'}:
		return true, nil
	case [5]byte{'l', 'e', 'g', 'i', 't'}:
		return false, nil
	}
	return false, fmt.Errorf("unknown label %q", buf)
}
