package crc32

import (
	"bytes"
	"encoding"
	stdcrc32 "hash/crc32"
	"math/rand"
	"testing"
)

var polys = []struct {
	name string
	poly uint32
}{
	{"IEEE", IEEE},
	{"Castagnoli", Castagnoli},
	{"Koopman", Koopman},
}

func std(crc uint32, tab *Table, p []byte) uint32 {
	return stdcrc32.Update(crc, tab, p)
}

// TestChecksumMatchesStdlib is the core correctness contract: Checksum and
// Update must equal hash/crc32 for every length and polynomial, exercising both
// the CLMUL kernel (IEEE, default) and the standard-library fallback (IEEE below
// minBulk, and the non-IEEE polynomials which never fold).
func TestChecksumMatchesStdlib(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, p := range polys {
		tab := MakeTable(p.poly)
		stab := stdcrc32.MakeTable(p.poly)
		for n := 0; n <= 600; n++ {
			data := make([]byte, n)
			rng.Read(data)
			if got, want := Checksum(data, tab), stdcrc32.Checksum(data, stab); got != want {
				t.Fatalf("%s n=%d: got %08x want %08x", p.name, n, got, want)
			}
		}
		// Large random sizes spanning many fold iterations plus tails.
		for _, n := range []int{1024, 4096, 65537, 1<<16 + 13, 200000} {
			data := make([]byte, n)
			rng.Read(data)
			if got, want := Checksum(data, tab), stdcrc32.Checksum(data, stab); got != want {
				t.Fatalf("%s n=%d: got %08x want %08x", p.name, n, got, want)
			}
		}
	}
}

// TestChecksumIEEE checks the ChecksumIEEE convenience wrapper.
func TestChecksumIEEE(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	for _, n := range []int{0, 1, 15, 16, 511, 512, 4096} {
		data := make([]byte, n)
		rng.Read(data)
		if got, want := ChecksumIEEE(data), stdcrc32.ChecksumIEEE(data); got != want {
			t.Fatalf("n=%d: got %08x want %08x", n, got, want)
		}
	}
}

// TestUpdateSeeded covers non-zero seeds down both the kernel and the fallback
// path for the IEEE polynomial.
func TestUpdateSeeded(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	seeds := []uint32{0, 1, 0xffffffff, 0xdeadbeef, 12345}
	for _, n := range []int{0, 8, 16, 100, 511, 512, 513, 4096, 70000} {
		data := make([]byte, n)
		rng.Read(data)
		for _, seed := range seeds {
			if got, want := Update(seed, IEEETable, data), std(seed, IEEETable, data); got != want {
				t.Fatalf("n=%d seed=%08x: got %08x want %08x", n, seed, got, want)
			}
		}
	}
}

// TestUpdateChunked exercises incremental updates (the New/Write path) and
// confirms they match the stdlib for chunked input.
func TestUpdateChunked(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, p := range polys {
		tab := MakeTable(p.poly)
		stab := stdcrc32.MakeTable(p.poly)
		data := make([]byte, 5000)
		rng.Read(data)
		var crc uint32
		i := 0
		for i < len(data) {
			step := 1 + rng.Intn(257)
			if i+step > len(data) {
				step = len(data) - i
			}
			crc = Update(crc, tab, data[i:i+step])
			i += step
		}
		if want := stdcrc32.Checksum(data, stab); crc != want {
			t.Fatalf("%s chunked Update: got %08x want %08x", p.name, crc, want)
		}
	}
}

// TestForceKernelSmall drives the kernel path just above a lowered minBulk to
// cover the no-tail (exact multiple of 16) and with-tail branches
// deterministically, on every architecture (the generic build forwards
// foldKernel to the portable Go fold, so forcing hasKernel exercises it too).
func TestForceKernelSmall(t *testing.T) {
	savedMin, savedK := minBulk, hasKernel
	defer func() { minBulk = savedMin; hasKernel = savedK }()
	minBulk = 128
	hasKernel = true

	rng := rand.New(rand.NewSource(4))
	for _, n := range []int{128, 129, 143, 144, 160, 255, 256, 257, 4095, 4096} {
		data := make([]byte, n)
		rng.Read(data)
		for _, seed := range []uint32{0, 0xabcdef01} {
			if got, want := Update(seed, IEEETable, data), std(seed, IEEETable, data); got != want {
				t.Fatalf("n=%d seed=%08x: got %08x want %08x", n, seed, got, want)
			}
		}
	}
}

func TestNewHash(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	data := make([]byte, 3000)
	rng.Read(data)

	h := NewIEEE()
	if h.Size() != Size {
		t.Fatalf("Size = %d, want %d", h.Size(), Size)
	}
	if h.BlockSize() != 1 {
		t.Fatalf("BlockSize = %d, want 1", h.BlockSize())
	}
	h.Write(data[:1500])
	h.Write(data[1500:])
	if got, want := h.Sum32(), stdcrc32.ChecksumIEEE(data); got != want {
		t.Fatalf("Sum32 = %08x, want %08x", got, want)
	}

	// Sum appends big-endian.
	sum := h.Sum(nil)
	if len(sum) != 4 {
		t.Fatalf("Sum len = %d, want 4", len(sum))
	}
	var rebuilt uint32
	for _, b := range sum {
		rebuilt = rebuilt<<8 | uint32(b)
	}
	if rebuilt != h.Sum32() {
		t.Fatalf("Sum bytes %x decode to %08x, want %08x", sum, rebuilt, h.Sum32())
	}

	// Sum(prefix) preserves the prefix.
	pre := []byte("pre")
	s2 := h.Sum(pre)
	if !bytes.Equal(s2[:3], pre) || len(s2) != 7 {
		t.Fatalf("Sum(prefix) = %x", s2)
	}

	h.Reset()
	if h.Sum32() != 0 {
		t.Fatalf("after Reset Sum32 = %08x, want 0", h.Sum32())
	}
}

func TestMarshalRoundTrip(t *testing.T) {
	h := New(MakeTable(Castagnoli))
	h.Write([]byte("the quick brown fox jumps over the lazy dog, repeatedly!!"))
	want := h.Sum32()

	m := h.(encoding.BinaryMarshaler)
	state, err := m.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if len(state) != marshaledSize {
		t.Fatalf("marshaled size = %d, want %d", len(state), marshaledSize)
	}

	h2 := New(MakeTable(Castagnoli))
	u := h2.(encoding.BinaryUnmarshaler)
	if err := u.UnmarshalBinary(state); err != nil {
		t.Fatal(err)
	}
	if h2.Sum32() != want {
		t.Fatalf("after unmarshal Sum32 = %08x, want %08x", h2.Sum32(), want)
	}

	// AppendBinary produces the same bytes after the prefix.
	ab := h.(interface {
		AppendBinary([]byte) ([]byte, error)
	})
	appended, err := ab.AppendBinary([]byte("x"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(appended[1:], state) {
		t.Fatalf("AppendBinary mismatch")
	}
}

func TestUnmarshalErrors(t *testing.T) {
	u := NewIEEE().(encoding.BinaryUnmarshaler)

	if err := u.UnmarshalBinary([]byte("xx")); err != errInvalidIdentifier {
		t.Fatalf("short/bad magic: got %v", err)
	}
	if err := u.UnmarshalBinary([]byte(magic + "short")); err != errInvalidSize {
		t.Fatalf("bad size: got %v", err)
	}
	// Correct size & magic but wrong table fingerprint.
	good, _ := New(MakeTable(Castagnoli)).(encoding.BinaryMarshaler).MarshalBinary()
	if err := u.UnmarshalBinary(good); err != errTablesMismatch {
		t.Fatalf("table mismatch: got %v", err)
	}
}

func TestTableSum(t *testing.T) {
	// tableSum of a nil table is the IEEE checksum of an empty message.
	if got := tableSum(nil); got != stdcrc32.ChecksumIEEE(nil) {
		t.Fatalf("tableSum(nil) = %08x", got)
	}
}

// TestConstants pins the derived fold constants (regression guard on the
// polynomial math).
func TestConstants(t *testing.T) {
	c := makeConstants()
	if c != foldConsts {
		t.Fatalf("constants unstable: %v vs %v", c, foldConsts)
	}
	for i, v := range c {
		if v == 0 {
			t.Fatalf("degenerate constant at %d", i)
		}
	}
	// xnModP(0) == 1 (x^0), a cheap check of the reduction loop's identity path.
	if xnModP(0) != 1 {
		t.Fatalf("xnModP(0) = %d, want 1", xnModP(0))
	}
}

// TestClmul64 exercises clmul64 directly, including the bit-0 (i==0) path that
// the fold constants — which have a zero low word — never trigger. Values are
// carryless products checked by hand: (x+1)^2 = x^2+1 -> clmul(3,3) = 5.
func TestClmul64(t *testing.T) {
	cases := []struct{ a, b, hi, lo uint64 }{
		{3, 3, 0, 5},
		{0, 0xffffffffffffffff, 0, 0},
		{1, 1, 0, 1},
		{0xffffffffffffffff, 2, 1, 0xfffffffffffffffe},
	}
	for _, c := range cases {
		hi, lo := clmul64(c.a, c.b)
		if hi != c.hi || lo != c.lo {
			t.Fatalf("clmul64(%x,%x) = (%x,%x), want (%x,%x)", c.a, c.b, hi, lo, c.hi, c.lo)
		}
	}
	// high-bit carry: clmul(1<<63, 1<<1) sets hi bit 0.
	if hi, lo := clmul64(1<<63, 1<<1); hi != 1 || lo != 0 {
		t.Fatalf("clmul64 carry = (%x,%x), want (1,0)", hi, lo)
	}
}

// TestGoKernelReference exercises the portable Go fold (foldKernelGo, the shared
// constant derivation and the 128->32 reduction) directly on every arch,
// including those that route update() to the stdlib scalar path and so never
// reach the fold during normal operation. It reproduces a full IEEE checksum
// from the Go kernel and checks it against the standard library.
func TestGoKernelReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{128, 129, 144, 160, 256, 4096, 65536} {
		data := make([]byte, n)
		rng.Read(data)
		blocks := n / 16
		bulk := blocks * 16
		hi, lo := foldKernelGo(data[:bulk], uint64(^uint32(0)), &foldConsts)
		res := ^reduce128(hi, lo)
		if tail := data[bulk:]; len(tail) > 0 {
			res = stdcrc32.Update(res, IEEETable, tail)
		}
		if want := stdcrc32.ChecksumIEEE(data); res != want {
			t.Fatalf("n=%d: go-kernel %08x want %08x", n, res, want)
		}
	}
}

// FuzzChecksum is the authoritative correctness gate: for arbitrary inputs the
// SIMD checksum must equal hash/crc32 for the IEEE and Castagnoli polynomials.
func FuzzChecksum(f *testing.F) {
	for _, seed := range [][]byte{
		nil, {}, []byte("a"), []byte("123456789"),
		bytes.Repeat([]byte("z"), 511), bytes.Repeat([]byte("z"), 512),
		bytes.Repeat([]byte("Q"), 513), bytes.Repeat([]byte{0xff}, 4096),
	} {
		f.Add(seed)
	}
	ieeeT, castT := IEEETable, MakeTable(Castagnoli)
	castS := stdcrc32.MakeTable(Castagnoli)
	f.Fuzz(func(t *testing.T, data []byte) {
		if got, want := Checksum(data, ieeeT), stdcrc32.ChecksumIEEE(data); got != want {
			t.Fatalf("IEEE len=%d: got %08x want %08x", len(data), got, want)
		}
		if got, want := Checksum(data, castT), stdcrc32.Checksum(data, castS); got != want {
			t.Fatalf("Castagnoli len=%d: got %08x want %08x", len(data), got, want)
		}
	})
}
