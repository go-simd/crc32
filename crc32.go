// Package crc32 is a pure-Go, SIMD-accelerated drop-in replacement for the
// standard library's hash/crc32. It computes bit-identical CRC-32 checksums but,
// for the IEEE polynomial, folds the bulk of the input with the host's
// carryless-multiply unit instead of the scalar table:
//
//	arm64    PMULL / PMULL2   (gated on cpu.ARM64.HasPMULL, and always on
//	                           GOOS=darwin, where every host is Apple Silicon
//	                           with FEAT_PMULL but the cpu package does not probe
//	                           the HWCAP)
//
// Every other GOARCH (amd64, ppc64le, s390x, riscv64, loong64, ...) defers to
// the standard library's own CRC-32 kernel, which is already hardware-assisted
// on several of those targets — the amd64 path is a PCLMULQDQ fold (AVX-512
// VPCLMULQDQ where available), and ppc64le/s390x are hardware-assisted too, so
// there is nothing to beat there. The hand-written kernel here targets arm64,
// whose standard-library IEEE path is a latency-bound serial CRC32X.
//
// Only the IEEE polynomial is folded by the kernel; Castagnoli, Koopman and any
// custom polynomial transparently use the standard library. The result always
// equals hash/crc32 exactly — bit for bit, on every architecture and for every
// polynomial. There is no cgo, no GOEXPERIMENT and no assembler intrinsic
// requirement: a plain `go build` produces the accelerated binary.
//
// The API mirrors hash/crc32 exactly, so it can be swapped in by changing only
// the import path.
package crc32

import (
	"hash"
	stdcrc32 "hash/crc32"
)

// The size of a CRC-32 checksum in bytes.
const Size = 4

// Predefined polynomials (identical to hash/crc32).
const (
	// IEEE is by far and away the most common CRC-32 polynomial. Used by
	// ethernet (IEEE 802.3), v.42, fddi, gzip, zip, png, ...
	IEEE = stdcrc32.IEEE
	// Castagnoli's polynomial, used in iSCSI. Has better error detection
	// characteristics than IEEE.
	Castagnoli = stdcrc32.Castagnoli
	// Koopman's polynomial. Also has better error detection characteristics
	// than IEEE.
	Koopman = stdcrc32.Koopman
)

// Table is a 256-word table representing the polynomial for efficient
// processing. It is the same type as hash/crc32.Table so tables are
// interchangeable between the two packages.
type Table = stdcrc32.Table

// IEEETable is the table for the IEEE polynomial.
var IEEETable = stdcrc32.IEEETable

// MakeTable returns a Table constructed from the specified polynomial. The
// contents of this Table must not be modified.
func MakeTable(poly uint32) *Table { return stdcrc32.MakeTable(poly) }

// New creates a new hash.Hash32 computing the CRC-32 checksum using the
// polynomial represented by the Table. Its Sum method lays the value out in
// big-endian byte order. The returned Hash32 also implements
// encoding.BinaryMarshaler and encoding.BinaryUnmarshaler.
//
// The hash uses the SIMD-accelerated kernel for its Write path when the table is
// the IEEE polynomial.
func New(tab *Table) hash.Hash32 { return &digest{0, tab} }

// NewIEEE creates a new hash.Hash32 computing the CRC-32 checksum using the IEEE
// polynomial.
func NewIEEE() hash.Hash32 { return New(IEEETable) }

// digest mirrors hash/crc32's digest but routes Write through the SIMD update.
type digest struct {
	crc uint32
	tab *Table
}

func (d *digest) Size() int      { return Size }
func (d *digest) BlockSize() int { return 1 }
func (d *digest) Reset()         { d.crc = 0 }

func (d *digest) Write(p []byte) (int, error) {
	d.crc = update(d.crc, d.tab, p)
	return len(p), nil
}

func (d *digest) Sum32() uint32 { return d.crc }

func (d *digest) Sum(in []byte) []byte {
	s := d.Sum32()
	return append(in, byte(s>>24), byte(s>>16), byte(s>>8), byte(s))
}

const (
	magic         = "crc\x01"
	marshaledSize = len(magic) + 4 + 4
)

func (d *digest) MarshalBinary() ([]byte, error) {
	b := make([]byte, 0, marshaledSize)
	b = append(b, magic...)
	b = appendUint32(b, tableSum(d.tab))
	b = appendUint32(b, d.crc)
	return b, nil
}

// AppendBinary implements encoding.BinaryAppender.
func (d *digest) AppendBinary(b []byte) ([]byte, error) {
	b = append(b, magic...)
	b = appendUint32(b, tableSum(d.tab))
	b = appendUint32(b, d.crc)
	return b, nil
}

func (d *digest) UnmarshalBinary(b []byte) error {
	if len(b) < len(magic) || string(b[:len(magic)]) != magic {
		return errInvalidIdentifier
	}
	if len(b) != marshaledSize {
		return errInvalidSize
	}
	if tableSum(d.tab) != beUint32(b[4:]) {
		return errTablesMismatch
	}
	d.crc = beUint32(b[8:])
	return nil
}

// Checksum returns the CRC-32 checksum of data using the polynomial represented
// by the Table.
func Checksum(data []byte, tab *Table) uint32 { return update(0, tab, data) }

// ChecksumIEEE returns the CRC-32 checksum of data using the IEEE polynomial.
func ChecksumIEEE(data []byte) uint32 { return update(0, IEEETable, data) }

// Update returns the result of adding the bytes in p to the crc.
func Update(crc uint32, tab *Table, p []byte) uint32 { return update(crc, tab, p) }

// tableSum returns the IEEE checksum of table t (matches hash/crc32.tableSum,
// used for the marshaled-state table fingerprint).
func tableSum(t *Table) uint32 {
	var a [1024]byte
	b := a[:0]
	if t != nil {
		for _, x := range t {
			b = appendUint32(b, x)
		}
	}
	return ChecksumIEEE(b)
}

func appendUint32(b []byte, v uint32) []byte {
	return append(b, byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

func beUint32(b []byte) uint32 {
	_ = b[3]
	return uint32(b[3]) | uint32(b[2])<<8 | uint32(b[1])<<16 | uint32(b[0])<<24
}
