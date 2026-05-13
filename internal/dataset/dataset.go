// Package dataset is the build-time loader for references.json.gz.
//
// Runtime search uses internal/ivf for the IVF index. Dataset is only used
// at index-build time by cmd/build-index (and by tests/benches that need
// raw vectors).
//
// The on-disk index format produced by the builder is owned by internal/ivf,
// not by this package — see lecture 09.
package dataset

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"os"
)

const (
	// Dims is the vector arity used everywhere on the hot path.
	Dims = 14
	// Stride is the per-vector stride in the flat int16 slab.
	Stride       = 14
	QuantScale   = 32000
	SentinelInt  = -32000
	SentinelReal = -1.0
)

// Dataset is the in-memory representation of references.json.gz after
// loading and quantization. The IVF builder consumes Vectors + Labels.
type Dataset struct {
	Vectors []int16 // length = Count * Stride
	Labels  []uint8 // length = Count
	Count   int
}

// LoadFromGzipJSON streams `references.json.gz`, quantizes floats → int16,
// and returns a Dataset with Vectors and Labels filled in source order.
func LoadFromGzipJSON(path string) (*Dataset, error) {
	count, err := scan(path, nil)
	if err != nil {
		return nil, fmt.Errorf("count: %w", err)
	}

	ds := &Dataset{
		Vectors: make([]int16, count*Stride),
		Labels:  make([]uint8, count),
		Count:   count,
	}

	idx := 0
	got, err := scan(path, func(vec [Dims]float32, isFraud bool) {
		base := idx * Stride
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
		if err := readUntil(br, vectorKey); err != nil {
			if err == io.EOF {
				return count, nil
			}
			return count, fmt.Errorf("scan vector key (entry %d): %w", count, err)
		}

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
			case 3:
				expSign = 1
				exp = int(b - '0')
				state = 4
			case 4:
				exp = exp*10 + int(b-'0')
			}
		case b == '.':
			state = 2
		case b == 'e' || b == 'E':
			state = 3
		case b == '+' && state == 3:
			expSign = 1
		case b == '-' && state == 3:
			expSign = -1
		default:
			if exp != 0 {
				pow := 1.0
				for i := 0; i < exp; i++ {
					pow *= 10
				}
				if expSign < 0 {
					v /= pow
				} else {
					v *= pow
				}
			}
			return float32(v * sign), b, nil
		}
	}
}

// readLabel parses a label value (after the opening quote): either "fraud"
// (5 bytes + closing quote) or "legit" (5 bytes + closing quote). Returns
// true if fraud.
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
