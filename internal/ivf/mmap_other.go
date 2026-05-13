//go:build !linux

package ivf

import "os"

// LoadMmap on non-linux falls back to read-into-heap. Used for dev on
// darwin/arm64 only — production runs linux/amd64.
func LoadMmap(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Load(f)
}

func (idx *IVFIndex) Munmap() error { return nil }
