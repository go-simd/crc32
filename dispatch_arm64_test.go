//go:build arm64

package crc32

import (
	stdcrc32 "hash/crc32"
	"math/rand"
	"testing"
)

// TestDispatchARM64 drives both update() branches — the PMULL kernel and the
// standard-library fallback — and additionally compares the assembly kernel
// against the portable Go reference bit for bit. The PMULL instruction is always
// decodable on the arm64 targets this package runs on; cpu.ARM64.HasPMULL is
// merely unset on darwin (the cpu package reads Linux HWCAP), so we force the
// kernel on to exercise the real assembly on every host, including the macOS dev
// machine.
func TestDispatchARM64(t *testing.T) {
	saved := hasKernel
	defer func() { hasKernel = saved }()

	rng := rand.New(rand.NewSource(12))
	check := func(label string) {
		for _, n := range []int{64, 128, 129, 160, 511, 512, 513, 1000, 4096, 70000} {
			data := make([]byte, n)
			rng.Read(data)
			if got, want := ChecksumIEEE(data), stdcrc32.ChecksumIEEE(data); got != want {
				t.Fatalf("%s n=%d: got %08x want %08x", label, n, got, want)
			}
		}
	}

	hasKernel = false
	check("fallback")
	hasKernel = true
	check("pmull")

	// Compare the assembly kernel directly against the Go reference, bit for bit.
	rng2 := rand.New(rand.NewSource(13))
	for _, n := range []int{128, 144, 160, 256, 4096, 65536} {
		data := make([]byte, n)
		rng2.Read(data)
		init := rng2.Uint64()
		gh, gl := foldKernelGo(data, init, &foldConsts)
		kh, kl := foldKernel(data, init, &foldConsts)
		if gh != kh || gl != kl {
			t.Fatalf("n=%d: pmull=(%016x,%016x) go=(%016x,%016x)", n, kh, kl, gh, gl)
		}
	}
}
