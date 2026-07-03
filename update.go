package crc32

import (
	stdcrc32 "hash/crc32"
	"math/bits"
)

// minBulk is the smallest input for which the CLMUL kernel is engaged. Below it
// the per-call setup (the 128->32 reduction and the scalar tail) outweighs the
// fold's throughput advantage, so the standard-library path is used instead. It
// must be at least 128: the fold-by-eight kernel seeds eight 128-bit lanes from
// the first 128 bytes. It is a var so tests can lower it to drive the kernel on
// small deterministic inputs.
var minBulk = 512

// polyNorm is the IEEE CRC-32 generator in normal (non-reflected) bit order,
// without the implicit x^32 term: reverse32(0xedb88320).
const (
	polyRef  = 0xedb88320 // reflected IEEE polynomial
	polyNorm = 0x04c11db7 // normal-order IEEE polynomial (x^32 implicit)
)

// polyOf recovers the reflected polynomial from a table. In hash/crc32's
// reflected table, entry 128 (0b1000_0000) shifts its single set bit down to
// bit 0 over the eight inner iterations and XORs the polynomial exactly once, so
// tab[128] == poly for every polynomial. The kernel is only valid for IEEE, so
// this is how update decides whether to fold or defer to the standard library.
func polyOf(tab *Table) uint32 { return tab[128] }

// xnModP returns x^n mod P in normal bit order (a value of degree < 32), where P
// is the degree-32 IEEE generator. Used to derive the reflected fold constants.
func xnModP(n int) uint32 {
	rem := uint32(1) // x^0
	for i := 0; i < n; i++ {
		msb := rem >> 31
		rem <<= 1
		if msb == 1 {
			rem ^= polyNorm
		}
	}
	return rem
}

// foldK returns the (lo, hi) constant pair that folds a 128-bit accumulator
// forward by dist bits. For a reflected fold the low word uses reflect(x^(dist+63)
// mod P) and the high word reflect(x^(dist-1) mod P); because each remainder is a
// degree-<32 value, the reflected constant is reverse32(rem) placed in the high
// half of the 64-bit lane the carryless multiply consumes.
func foldK(dist int) (lo, hi uint64) {
	lo = uint64(bits.Reverse32(xnModP(dist+63))) << 32
	hi = uint64(bits.Reverse32(xnModP(dist-1))) << 32
	return
}

// makeConstants builds the reflected fold constants for the IEEE polynomial, one
// (lo, hi) pair for every fold distance 128, 256, ... nLanes*128. The largest
// (nLanes*128 = 1024) drives the fold-by-eight main loop; the smaller distances
// collapse the eight lanes to one (128 also folds the single-lane 16-byte
// remainder).
func makeConstants() foldConstants {
	var c foldConstants
	for k := 0; k < nLanes; k++ {
		dist := (k + 1) * distStep
		c[off(dist)], c[off(dist)+1] = foldK(dist)
	}
	return c
}

// foldConsts is the precomputed constant pair for the IEEE polynomial. It is a
// package-level value so its address can be handed to the assembly kernel without
// escaping a fresh local to the heap on every call.
var foldConsts = makeConstants()

// reduce128 reduces the reflected 128-bit accumulator (hi:lo) to the reflected
// 32-bit CRC remainder. It runs once per update call on a fixed 128 bits, so the
// straightforward reflected bit-serial form is used: it is provably correct and
// entirely outside the hot fold loop. Bits are consumed LSB-first from lo then
// hi, exactly as the reflected CRC processes the input stream.
func reduce128(hi, lo uint64) uint32 {
	crc := uint32(0)
	for i := 0; i < 64; i++ {
		b := uint32((lo>>uint(i))&1) ^ (crc & 1)
		crc >>= 1
		if b == 1 {
			crc ^= polyRef
		}
	}
	for i := 0; i < 64; i++ {
		b := uint32((hi>>uint(i))&1) ^ (crc & 1)
		crc >>= 1
		if b == 1 {
			crc ^= polyRef
		}
	}
	return crc
}

// update returns the result of adding the bytes in p to the CRC-32 value crc,
// bit-identical to hash/crc32.Update(crc, tab, p). For the IEEE polynomial large
// inputs are folded through the CLMUL kernel; every other polynomial, small
// inputs and the sub-16-byte tail use the standard library.
func update(crc uint32, tab *Table, p []byte) uint32 {
	if polyOf(tab) != IEEE || len(p) < minBulk || !hasKernel {
		return stdcrc32.Update(crc, tab, p)
	}

	// Number of whole 16-byte blocks the kernel will consume.
	blocks := len(p) / 16
	bulkLen := blocks * 16

	// The kernel XORs init into the low word of the first 16-byte block. crc is
	// the finalized value; the running CRC state is its ones-complement.
	init := uint64(^crc)

	hi, lo := foldKernel(p[:bulkLen], init, &foldConsts)

	res := ^reduce128(hi, lo)

	if tail := p[bulkLen:]; len(tail) > 0 {
		res = stdcrc32.Update(res, tab, tail)
	}
	return res
}
