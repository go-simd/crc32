<p align="center"><img src="https://raw.githubusercontent.com/go-simd/brand/main/social/go-simd.png" alt="go-simd/crc32" width="720"></p>

# crc32

[![ci](https://github.com/go-simd/crc32/actions/workflows/ci.yml/badge.svg)](https://github.com/go-simd/crc32/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/go-simd/crc32/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/go-simd/crc32.svg)](https://pkg.go.dev/github.com/go-simd/crc32)

A pure-Go, SIMD-accelerated **drop-in replacement for the standard library's
`hash/crc32`**. It produces bit-identical CRC-32 checksums — for the `IEEE`,
`Castagnoli` and `Koopman` polynomials and for any custom polynomial — but, for
the ubiquitous **IEEE** polynomial, folds the bulk of the input with the host's
carryless-multiply unit instead of the scalar table.

No cgo, no `GOEXPERIMENT`, no assembler intrinsics required: a plain `go build`
produces the accelerated binary.

## Why arm64 only

`hash/crc32`'s IEEE fast path is already an excellent PCLMULQDQ fold on amd64
(AVX-512 `VPCLMULQDQ` fold-by-sixteen where available) and is hardware-assisted
on ppc64le and s390x, so there is nothing to beat there. The one 64-bit target
whose standard-library IEEE path is a latency-bound serial `CRC32X` is **arm64**;
this package folds it with an eight-lane `PMULL` / `PMULL2` kernel and defers to
the standard library everywhere else. The result is always exactly `hash/crc32`.

| Arch    | IEEE bulk path                              | Gate                                   |
|---------|---------------------------------------------|----------------------------------------|
| arm64   | `PMULL` / `PMULL2` fold-by-eight (this pkg) | `cpu.ARM64.HasPMULL`, always on darwin |
| amd64   | stdlib `PCLMULQDQ` / AVX-512 fold           | standard library                       |
| ppc64le | stdlib `VPMSUMD`                            | standard library                       |
| s390x   | stdlib vector-galois                        | standard library                       |
| riscv64 | stdlib scalar table                         | standard library                       |
| loong64 | stdlib scalar table                         | standard library                       |

Non-IEEE polynomials (Castagnoli, Koopman, custom) always use the standard
library on every architecture — the kernel constants are derived for IEEE.

## Drop-in usage

Change only the import path:

```go
import (
	"github.com/go-simd/crc32" // was "hash/crc32"
)

func main() {
	// Convenience helpers for the IEEE polynomial.
	sum := crc32.ChecksumIEEE(data)

	// Or the full table-based API, identical to hash/crc32.
	tab := crc32.MakeTable(crc32.IEEE)
	sum = crc32.Checksum(data, tab)
	crc := crc32.Update(0, tab, data)

	h := crc32.NewIEEE()
	h.Write(data)
	_ = h.Sum32()
}
```

The API matches `hash/crc32` exactly: `Checksum`, `ChecksumIEEE`, `Update`,
`New`, `NewIEEE`, `MakeTable`, `IEEETable`, the `IEEE`/`Castagnoli`/`Koopman`
constants, `Size`, the `Table` type (aliased to the stdlib type so tables are
interchangeable), and the `hash.Hash32` returned by `New` (including
`encoding.BinaryMarshaler`/`BinaryUnmarshaler`/`AppendBinary`).

## How it works

For the IEEE polynomial and inputs at or above `minBulk` (512 B), the data is
folded 16 bytes at a time into eight independent 128-bit reflected accumulators
using carryless multiplication — eight dependency chains hide the `PMULL`
latency, which is what lifts it past arm64's serial hardware `CRC32X`. The lanes
are then collapsed to one and reduced to the 32-bit CRC. The fold constants are
derived from the IEEE polynomial itself (`reflect(x^(d+63) mod P)` and
`reflect(x^(d-1) mod P)`) — there are no copied magic numbers. The short tail
(< 16 bytes) and every non-IEEE polynomial reuse the standard library, so the
result is guaranteed identical to `hash/crc32`.

The portable Go fold (`fold_go.go`) is the exact specification the arm64
assembly (`kernel_arm64.s`) reproduces bit for bit; a dispatch test compares the
two directly on every arm64 host.

## Testing

Correctness is gated by `FuzzChecksum`, which compares against `hash/crc32` for
both IEEE and Castagnoli on arbitrary inputs, plus exhaustive length sweeps
across every block boundary for IEEE, Castagnoli and Koopman. CI runs on native
amd64/arm64 (with `-race`) and under QEMU for riscv64, loong64, ppc64le (power9)
and s390x (big-endian), with a **100 %-statement-coverage** gate on every
architecture.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
