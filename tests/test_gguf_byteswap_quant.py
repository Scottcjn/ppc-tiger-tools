#!/usr/bin/env python3
"""
Regression tests for GGUF quantized-block byte swapping (tools/gguf_byteswap.py).

Every legacy quantized block starts with an f16 ``d`` (scale) value and, for the
*_1 variants, an f16 ``m``/``s`` value. On big-endian PowerPC those 16-bit fields
MUST be byte swapped, otherwise the inference engine reads a garbage scale for
every block and the whole tensor is corrupted.

swap_tensor_data() routes Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and Q8_1 to
swap_quantized_blocks(), so all six must actually swap their f16 scale bytes.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import gguf_byteswap as g  # noqa: E402


def _block(size, has_min):
    """Build one synthetic block: known f16 scale (and min) + filler weights."""
    b = bytearray(size)
    b[0], b[1] = 0xAB, 0xCD          # f16 scale/d
    if has_min:
        b[2], b[3] = 0x12, 0x34      # f16 min/sum
    for k in range(4 if has_min else 2, size):
        b[k] = 0xFF                  # weight bytes (must stay untouched)
    return bytes(b)


# (name, ggml type, block size in bytes, has second f16 field)
CASES = [
    ("Q4_0", g.GGML_TYPE_Q4_0, 18, False),
    ("Q4_1", g.GGML_TYPE_Q4_1, 20, True),
    ("Q5_0", g.GGML_TYPE_Q5_0, 22, False),
    ("Q5_1", g.GGML_TYPE_Q5_1, 24, True),
    ("Q8_0", g.GGML_TYPE_Q8_0, 34, False),
    ("Q8_1", g.GGML_TYPE_Q8_1, 36, True),
]


class TestQuantScaleSwap(unittest.TestCase):
    def setUp(self):
        self.sw = g.GGUFByteSwapper("in", "out")

    def test_scale_is_swapped_for_every_legacy_quant(self):
        for name, ttype, size, has_min in CASES:
            with self.subTest(quant=name):
                data = _block(size, has_min)
                # Two blocks back-to-back to catch off-by-one block striding.
                out = self.sw.swap_tensor_data(data + data, ttype)

                # f16 scale must be byte-swapped in both blocks.
                self.assertEqual(out[0], 0xCD, f"{name} block0 scale not swapped")
                self.assertEqual(out[1], 0xAB, f"{name} block0 scale not swapped")
                self.assertEqual(out[size], 0xCD, f"{name} block1 scale not swapped")
                self.assertEqual(out[size + 1], 0xAB, f"{name} block1 scale not swapped")

                if has_min:
                    self.assertEqual(out[2], 0x34, f"{name} min/sum not swapped")
                    self.assertEqual(out[3], 0x12, f"{name} min/sum not swapped")

                # Weight bytes must be left exactly as-is (byte arrays).
                weight_start = 4 if has_min else 2
                self.assertEqual(out[weight_start], 0xFF, f"{name} weights corrupted")
                self.assertEqual(len(out), 2 * size, f"{name} length changed")

    def test_q5_qh_bitmask_is_byte_swapped(self):
        """Q5_0/Q5_1's qh field is not a byte array like qs -- ggml reads it
        with memcpy(&qh, block->qh, sizeof(qh)) into a uint32_t and then
        bit-shifts per weight. On a big-endian target that memcpy
        reinterprets the 4 raw bytes as big-endian, so unless we swap them
        here too, every weight in the block picks up the wrong 5th bit.
        Q8_1 has no qh field (d + s + qs only) and is deliberately excluded.
        """
        cases = [
            ("Q5_0", g.GGML_TYPE_Q5_0, 22, 2),   # qh at offset 2 (no min field)
            ("Q5_1", g.GGML_TYPE_Q5_1, 24, 4),   # qh at offset 4 (after d + m)
        ]
        for name, ttype, size, qh_off in cases:
            with self.subTest(quant=name):
                b = bytearray(size)
                b[qh_off:qh_off + 4] = bytes([0x11, 0x22, 0x33, 0x44])
                qs_off = qh_off + 4
                for k in range(qs_off, size):
                    b[k] = 0xAA  # qs weight nibbles: must never move
                data = bytes(b)

                out = self.sw.swap_tensor_data(data, ttype)

                self.assertEqual(
                    out[qh_off:qh_off + 4], bytes([0x44, 0x33, 0x22, 0x11]),
                    f"{name} qh bitmask not byte-swapped",
                )
                self.assertEqual(
                    out[qs_off:size], b"\xaa" * (size - qs_off),
                    f"{name} qs weight bytes must stay untouched",
                )


if __name__ == "__main__":
    unittest.main()
