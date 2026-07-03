package crc32

import "errors"

var (
	errInvalidIdentifier = errors.New("hash/crc32: invalid hash state identifier")
	errInvalidSize       = errors.New("hash/crc32: invalid hash state size")
	errTablesMismatch    = errors.New("hash/crc32: tables do not match")
)
