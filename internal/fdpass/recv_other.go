//go:build !linux

package fdpass

// Listen on non-Linux is a stub returning a closed channel. We only run in
// production on Linux; this branch only exists so the package compiles for
// darwin/arm64 development.
func Listen(ctrlPath string) (<-chan int, int, error) {
	ch := make(chan int)
	close(ch)
	return ch, -1, nil
}
