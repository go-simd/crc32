//go:build arm64

package crc32

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// foldKernel is the PMULL fold-by-eight, structurally identical to
// go-simd/crc64's single-lane fold widened to eight lanes (kernel_arm64.s). It
// requires len(p) >= 128 and len(p) % 16 == 0.
func foldKernel(p []byte, init uint64, c *foldConstants) (hi, lo uint64)

// hasKernel gates update onto the PMULL kernel. It is a var so tests can force it
// off to exercise the standard-library fallback. cpu.ARM64.HasPMULL reads Linux
// HWCAP and is unset on darwin, but every arm64 Mac is Apple Silicon with
// FEAT_PMULL, so the kernel is enabled there unconditionally.
var hasKernel = cpu.ARM64.HasPMULL || runtime.GOOS == "darwin"
