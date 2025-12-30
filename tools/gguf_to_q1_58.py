#!/usr/bin/env python3
"""
GGUF to Q1.58 (BitNet Ternary) Converter for PowerPC G4/G5

Converts standard GGUF models to ternary quantization {-1, 0, +1}.
Eliminates all floating-point multiplication in inference.

Usage:
    python3 gguf_to_q1_58.py input.gguf output-q1_58.gguf [--big-endian]
"""

import struct
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# GGML type constants
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_Q1_58 = 20  # Our new type

# Q1_58 block constants
QK_Q1_58 = 256  # Elements per block
BLOCK_SIZE_Q1_58 = 68  # 64 bytes packed + 2 bytes scale + 2 bytes zero_count

# GGUF value types
GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12


class Q158Converter:
    """Converts GGUF models to Q1.58 ternary format."""

    def __init__(self, input_path: str, output_path: str, big_endian: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.big_endian = big_endian

        # Endian format strings
        self.le = '<'  # Little endian for input (GGUF standard)
        self.out = '>' if big_endian else '<'  # Output format

    def read_u32(self, f) -> int:
        return struct.unpack(f'{self.le}I', f.read(4))[0]

    def read_u64(self, f) -> int:
        return struct.unpack(f'{self.le}Q', f.read(8))[0]

    def read_i32(self, f) -> int:
        return struct.unpack(f'{self.le}i', f.read(4))[0]

    def read_f32(self, f) -> float:
        return struct.unpack(f'{self.le}f', f.read(4))[0]

    def read_string(self, f) -> str:
        length = self.read_u64(f)
        return f.read(length).decode('utf-8')

    def write_u32(self, f, val: int):
        f.write(struct.pack(f'{self.out}I', val))

    def write_u64(self, f, val: int):
        f.write(struct.pack(f'{self.out}Q', val))

    def write_i32(self, f, val: int):
        f.write(struct.pack(f'{self.out}i', val))

    def write_f32(self, f, val: float):
        f.write(struct.pack(f'{self.out}f', val))

    def write_string(self, f, s: str):
        data = s.encode('utf-8')
        self.write_u64(f, len(data))
        f.write(data)

    def read_value(self, f, vtype: int):
        """Read a GGUF value of given type."""
        if vtype == GGUF_VALUE_TYPE_UINT8:
            return struct.unpack('B', f.read(1))[0]
        elif vtype == GGUF_VALUE_TYPE_INT8:
            return struct.unpack('b', f.read(1))[0]
        elif vtype == GGUF_VALUE_TYPE_UINT16:
            return struct.unpack(f'{self.le}H', f.read(2))[0]
        elif vtype == GGUF_VALUE_TYPE_INT16:
            return struct.unpack(f'{self.le}h', f.read(2))[0]
        elif vtype == GGUF_VALUE_TYPE_UINT32:
            return self.read_u32(f)
        elif vtype == GGUF_VALUE_TYPE_INT32:
            return self.read_i32(f)
        elif vtype == GGUF_VALUE_TYPE_FLOAT32:
            return self.read_f32(f)
        elif vtype == GGUF_VALUE_TYPE_BOOL:
            return struct.unpack('B', f.read(1))[0] != 0
        elif vtype == GGUF_VALUE_TYPE_STRING:
            return self.read_string(f)
        elif vtype == GGUF_VALUE_TYPE_UINT64:
            return self.read_u64(f)
        elif vtype == GGUF_VALUE_TYPE_INT64:
            return struct.unpack(f'{self.le}q', f.read(8))[0]
        elif vtype == GGUF_VALUE_TYPE_FLOAT64:
            return struct.unpack(f'{self.le}d', f.read(8))[0]
        elif vtype == GGUF_VALUE_TYPE_ARRAY:
            arr_type = self.read_u32(f)
            arr_len = self.read_u64(f)
            return [self.read_value(f, arr_type) for _ in range(arr_len)]
        else:
            raise ValueError(f"Unknown value type: {vtype}")

    def write_value(self, f, val, vtype: int):
        """Write a GGUF value of given type."""
        if vtype == GGUF_VALUE_TYPE_UINT8:
            f.write(struct.pack('B', val))
        elif vtype == GGUF_VALUE_TYPE_INT8:
            f.write(struct.pack('b', val))
        elif vtype == GGUF_VALUE_TYPE_UINT16:
            f.write(struct.pack(f'{self.out}H', val))
        elif vtype == GGUF_VALUE_TYPE_INT16:
            f.write(struct.pack(f'{self.out}h', val))
        elif vtype == GGUF_VALUE_TYPE_UINT32:
            self.write_u32(f, val)
        elif vtype == GGUF_VALUE_TYPE_INT32:
            self.write_i32(f, val)
        elif vtype == GGUF_VALUE_TYPE_FLOAT32:
            self.write_f32(f, val)
        elif vtype == GGUF_VALUE_TYPE_BOOL:
            f.write(struct.pack('B', 1 if val else 0))
        elif vtype == GGUF_VALUE_TYPE_STRING:
            self.write_string(f, val)
        elif vtype == GGUF_VALUE_TYPE_UINT64:
            self.write_u64(f, val)
        elif vtype == GGUF_VALUE_TYPE_INT64:
            f.write(struct.pack(f'{self.out}q', val))
        elif vtype == GGUF_VALUE_TYPE_FLOAT64:
            f.write(struct.pack(f'{self.out}d', val))
        elif vtype == GGUF_VALUE_TYPE_ARRAY:
            # For arrays, val should be (arr_type, items)
            arr_type, items = val
            self.write_u32(f, arr_type)
            self.write_u64(f, len(items))
            for item in items:
                self.write_value(f, item, arr_type)

    def dequantize_q4_k(self, data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q4_K data to float32."""
        block_size = 144  # Q4_K block size
        n_blocks = len(data) // block_size
        result = np.zeros(n_elements, dtype=np.float32)

        for b in range(n_blocks):
            offset = b * block_size
            block = data[offset:offset + block_size]

            # d and dmin are FP16 at bytes 0-1 and 2-3
            d = np.frombuffer(block[0:2], dtype=np.float16)[0]
            dmin = np.frombuffer(block[2:4], dtype=np.float16)[0]
            scales = block[4:16]  # 12 bytes of scales
            qs = block[16:144]    # 128 bytes of quantized values

            # Decode 6-bit scales
            sc = np.zeros(8, dtype=np.float32)
            m = np.zeros(8, dtype=np.float32)

            for i in range(8):
                sc[i] = (scales[i % 4] >> ((i // 4) * 4)) & 0x3f
                m[i] = (scales[4 + i % 4] >> ((i // 4) * 4)) & 0x3f

            # Dequantize 256 values
            for j in range(256):
                scale_idx = j // 32
                q_idx = j // 2
                shift = (j % 2) * 4

                q = (qs[q_idx] >> shift) & 0x0f
                result[b * 256 + j] = float(d) * (sc[scale_idx] * q) - float(dmin) * m[scale_idx]

        return result

    def dequantize_q6_k(self, data: bytes, n_elements: int) -> np.ndarray:
        """Dequantize Q6_K data to float32."""
        block_size = 210  # Q6_K block size
        n_blocks = len(data) // block_size
        result = np.zeros(n_elements, dtype=np.float32)

        for b in range(n_blocks):
            offset = b * block_size
            block = data[offset:offset + block_size]

            # Q6_K structure: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes
            ql = block[0:128]     # Low 4 bits
            qh = block[128:192]   # High 2 bits
            scales = block[192:208]  # 16 int8 scales
            d = np.frombuffer(block[208:210], dtype=np.float16)[0]

            # Dequantize 256 values
            for j in range(256):
                # Get 6-bit quantized value
                l_idx = j % 128
                h_idx = (j % 128) // 2
                h_shift = ((j % 128) % 2) * 2 + (j // 128) * 4

                q_l = ql[l_idx] if j < 128 else (ql[l_idx] >> 4) & 0x0f
                if j < 128:
                    q_l = ql[j] & 0x0f if j % 2 == 0 else (ql[j // 2] >> 4) & 0x0f
                    q_l = ql[j // 2]
                    q_l = (q_l >> ((j % 2) * 4)) & 0x0f
                else:
                    q_l = ql[(j - 128) // 2]
                    q_l = (q_l >> (((j - 128) % 2) * 4)) & 0x0f

                # Simplified: just use first 4 bits for now
                q_idx = j // 2
                q = ql[q_idx % 128]
                if j < 128:
                    q = (q >> ((j % 2) * 4)) & 0x0f
                else:
                    q = (q >> (((j - 128) % 2) * 4)) & 0x0f

                # Get scale (scales are signed int8)
                scale_idx = j // 16
                sc = scales[scale_idx]
                if sc > 127:
                    sc = sc - 256  # Convert to signed

                result[b * 256 + j] = float(d) * sc * (q - 8)

        return result

    def dequantize_f32(self, data: bytes) -> np.ndarray:
        """Interpret bytes as float32."""
        return np.frombuffer(data, dtype=np.float32)

    def dequantize_f16(self, data: bytes) -> np.ndarray:
        """Interpret bytes as float16, convert to float32."""
        return np.frombuffer(data, dtype=np.float16).astype(np.float32)

    def quantize_to_ternary(self, weights: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        Quantize float weights to ternary {-1, 0, +1}.
        Uses absmean quantization from BitNet b1.58.

        Returns: (ternary_weights, scale, zero_count)
        """
        # Compute scale factor (mean absolute value)
        scale = np.mean(np.abs(weights))
        if scale < 1e-10:
            scale = 1e-10

        # Normalize and round to nearest ternary
        normalized = weights / scale
        ternary = np.clip(np.round(normalized), -1, 1).astype(np.int8)

        # Count zeros for sparsity hint
        zero_count = int(np.sum(ternary == 0))

        return ternary, float(scale), zero_count

    def pack_ternary(self, ternary: np.ndarray) -> bytes:
        """
        Pack ternary values {-1, 0, +1} into 2-bit encoding.
        -1 -> 00, 0 -> 01, +1 -> 10
        """
        # Map {-1, 0, 1} to {0, 1, 2}
        encoded = (ternary + 1).astype(np.uint8)

        # Pack 4 values per byte
        packed = bytearray()
        for i in range(0, len(encoded), 4):
            if i + 3 < len(encoded):
                byte = (encoded[i] |
                        (encoded[i+1] << 2) |
                        (encoded[i+2] << 4) |
                        (encoded[i+3] << 6))
            else:
                # Handle partial block at end
                byte = encoded[i]
                if i + 1 < len(encoded):
                    byte |= encoded[i+1] << 2
                if i + 2 < len(encoded):
                    byte |= encoded[i+2] << 4
                if i + 3 < len(encoded):
                    byte |= encoded[i+3] << 6
            packed.append(byte)

        return bytes(packed)

    def float32_to_float16(self, val: float) -> bytes:
        """Convert float32 to float16 bytes."""
        f16 = np.float16(val)
        data = f16.tobytes()
        if self.big_endian:
            data = data[::-1]  # Reverse for big endian
        return data

    def convert_tensor_to_q1_58(self, data: bytes, tensor_type: int, n_elements: int) -> bytes:
        """Convert tensor data to Q1_58 format."""

        # First, dequantize to float32
        if tensor_type == GGML_TYPE_F32:
            floats = self.dequantize_f32(data)
        elif tensor_type == GGML_TYPE_F16:
            floats = self.dequantize_f16(data)
        elif tensor_type == GGML_TYPE_Q4_K:
            floats = self.dequantize_q4_k(data, n_elements)
        elif tensor_type == GGML_TYPE_Q6_K:
            floats = self.dequantize_q6_k(data, n_elements)
        else:
            print(f"  Warning: Unsupported type {tensor_type}, keeping as-is")
            return data

        # Pad to multiple of 256
        if len(floats) % QK_Q1_58 != 0:
            pad_size = QK_Q1_58 - (len(floats) % QK_Q1_58)
            floats = np.concatenate([floats, np.zeros(pad_size, dtype=np.float32)])

        # Quantize each block
        n_blocks = len(floats) // QK_Q1_58
        output = bytearray()

        for i in range(n_blocks):
            block = floats[i * QK_Q1_58:(i + 1) * QK_Q1_58]
            ternary, scale, zero_count = self.quantize_to_ternary(block)

            # Pack ternary weights (64 bytes for 256 weights)
            packed = self.pack_ternary(ternary)
            output.extend(packed)

            # Scale as FP16 (2 bytes)
            output.extend(self.float32_to_float16(scale))

            # Zero count as uint16 (2 bytes)
            output.extend(struct.pack(f'{self.out}H', min(zero_count, 65535)))

        return bytes(output)

    def convert(self):
        """Main conversion function."""
        print(f"Converting {self.input_path} to Q1.58 format")
        print(f"Output: {self.output_path}")
        print(f"Big-endian: {self.big_endian}")

        with open(self.input_path, 'rb') as fin:
            # Read header
            magic = fin.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Invalid GGUF magic: {magic}")

            version = self.read_u32(fin)
            tensor_count = self.read_u64(fin)
            metadata_count = self.read_u64(fin)

            print(f"Version: {version}, Tensors: {tensor_count}, Metadata: {metadata_count}")

            # Read metadata
            metadata = {}
            metadata_raw = []  # Store raw (key, type, value) for writing
            for _ in range(metadata_count):
                key = self.read_string(fin)
                vtype = self.read_u32(fin)
                value = self.read_value(fin, vtype)
                metadata[key] = value
                metadata_raw.append((key, vtype, value))

            # Read tensor info
            tensors = []
            for _ in range(tensor_count):
                name = self.read_string(fin)
                n_dims = self.read_u32(fin)
                dims = [self.read_u64(fin) for _ in range(n_dims)]
                ttype = self.read_u32(fin)
                offset = self.read_u64(fin)
                n_elements = 1
                for d in dims:
                    n_elements *= d
                tensors.append({
                    'name': name,
                    'n_dims': n_dims,
                    'dims': dims,
                    'type': ttype,
                    'offset': offset,
                    'n_elements': n_elements
                })

            # Get data start position (aligned to 32 bytes)
            current_pos = fin.tell()
            padding = (32 - (current_pos % 32)) % 32
            fin.read(padding)
            data_start = fin.tell()

            print(f"Data starts at offset {data_start}")

            # Sort tensors by offset for sequential reading
            tensors_sorted = sorted(tensors, key=lambda t: t['offset'])

        # Now write the output file
        with open(self.output_path, 'wb') as fout:
            # Write header
            fout.write(b'GGUF')
            self.write_u32(fout, version)
            self.write_u64(fout, tensor_count)
            self.write_u64(fout, metadata_count)

            # Write metadata
            for key, vtype, value in metadata_raw:
                self.write_string(fout, key)
                self.write_u32(fout, vtype)
                if vtype == GGUF_VALUE_TYPE_ARRAY:
                    # Need to determine array type
                    if value and len(value) > 0:
                        if isinstance(value[0], float):
                            arr_type = GGUF_VALUE_TYPE_FLOAT32
                        elif isinstance(value[0], int):
                            arr_type = GGUF_VALUE_TYPE_INT32
                        elif isinstance(value[0], str):
                            arr_type = GGUF_VALUE_TYPE_STRING
                        else:
                            arr_type = GGUF_VALUE_TYPE_UINT8
                        self.write_value(fout, (arr_type, value), vtype)
                    else:
                        self.write_u32(fout, GGUF_VALUE_TYPE_UINT8)
                        self.write_u64(fout, 0)
                else:
                    self.write_value(fout, value, vtype)

            # Calculate new tensor info (with updated types and offsets)
            new_tensors = []
            new_offset = 0
            for t in tensors:
                new_type = t['type']
                n_elements = t['n_elements']

                # Determine new size
                if t['type'] in [GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                                 GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_F32, GGML_TYPE_F16,
                                 14]:  # 14 = Q6_K
                    # Convert to Q1_58
                    new_type = GGML_TYPE_Q1_58
                    # Pad to multiple of 256
                    padded_elements = ((n_elements + QK_Q1_58 - 1) // QK_Q1_58) * QK_Q1_58
                    new_size = (padded_elements // QK_Q1_58) * BLOCK_SIZE_Q1_58
                else:
                    # Keep original size (for Q8 activations, etc.)
                    new_size = self.get_tensor_size(t['type'], n_elements)

                new_tensors.append({
                    'name': t['name'],
                    'n_dims': t['n_dims'],
                    'dims': t['dims'],
                    'type': new_type,
                    'offset': new_offset,
                    'n_elements': n_elements,
                    'orig_type': t['type'],
                    'orig_offset': t['offset']
                })
                new_offset += new_size

            # Write tensor info
            for t in new_tensors:
                self.write_string(fout, t['name'])
                self.write_u32(fout, t['n_dims'])
                for d in t['dims']:
                    self.write_u64(fout, d)
                self.write_u32(fout, t['type'])
                self.write_u64(fout, t['offset'])

            # Align to 32 bytes
            current_pos = fout.tell()
            padding = (32 - (current_pos % 32)) % 32
            fout.write(b'\x00' * padding)

            print(f"\nConverting {len(new_tensors)} tensors...")

            # Convert and write tensor data
            with open(self.input_path, 'rb') as fin:
                for i, t in enumerate(new_tensors):
                    name = t['name']
                    orig_type = t['orig_type']
                    orig_offset = t['orig_offset']
                    n_elements = t['n_elements']

                    # Read original data
                    orig_size = self.get_tensor_size(orig_type, n_elements)
                    fin.seek(data_start + orig_offset)
                    orig_data = fin.read(orig_size)

                    # Convert if needed
                    if t['type'] == GGML_TYPE_Q1_58:
                        new_data = self.convert_tensor_to_q1_58(orig_data, orig_type, n_elements)
                        print(f"  [{i+1}/{len(new_tensors)}] {name}: {self.type_name(orig_type)} -> Q1_58 "
                              f"({len(orig_data)} -> {len(new_data)} bytes)")
                    else:
                        new_data = orig_data
                        if self.big_endian:
                            # Byte swap if needed
                            new_data = self.byteswap_tensor(new_data, orig_type)
                        print(f"  [{i+1}/{len(new_tensors)}] {name}: keeping as {self.type_name(orig_type)}")

                    fout.write(new_data)

        # Print summary
        orig_size = os.path.getsize(self.input_path)
        new_size = os.path.getsize(self.output_path)
        print(f"\nConversion complete!")
        print(f"Original size: {orig_size / 1024 / 1024:.1f} MB")
        print(f"Q1.58 size:    {new_size / 1024 / 1024:.1f} MB")
        print(f"Compression:   {orig_size / new_size:.2f}x")

    def get_tensor_size(self, ttype: int, n_elements: int) -> int:
        """Calculate tensor size in bytes."""
        if ttype == GGML_TYPE_F32:
            return n_elements * 4
        elif ttype == GGML_TYPE_F16:
            return n_elements * 2
        elif ttype == GGML_TYPE_Q4_K:
            return (n_elements // 256) * 144
        elif ttype == GGML_TYPE_Q6_K:
            return (n_elements // 256) * 210
        elif ttype == GGML_TYPE_Q8_K:
            return (n_elements // 256) * 292
        elif ttype == GGML_TYPE_Q8_0:
            return (n_elements // 32) * 34
        elif ttype == GGML_TYPE_Q1_58:
            return (n_elements // QK_Q1_58) * BLOCK_SIZE_Q1_58
        else:
            # Default estimate
            return n_elements * 2

    def type_name(self, ttype: int) -> str:
        """Get human-readable type name."""
        names = {
            GGML_TYPE_F32: "F32",
            GGML_TYPE_F16: "F16",
            GGML_TYPE_Q4_0: "Q4_0",
            GGML_TYPE_Q4_1: "Q4_1",
            GGML_TYPE_Q4_K: "Q4_K",
            GGML_TYPE_Q5_K: "Q5_K",
            GGML_TYPE_Q6_K: "Q6_K",
            GGML_TYPE_Q8_K: "Q8_K",
            GGML_TYPE_Q8_0: "Q8_0",
            GGML_TYPE_Q1_58: "Q1_58",
        }
        return names.get(ttype, f"TYPE_{ttype}")

    def byteswap_tensor(self, data: bytes, ttype: int) -> bytes:
        """Byte-swap tensor data for big-endian output."""
        if ttype == GGML_TYPE_F32:
            arr = np.frombuffer(data, dtype=np.float32)
            return arr.byteswap().tobytes()
        elif ttype == GGML_TYPE_F16:
            arr = np.frombuffer(data, dtype=np.float16)
            return arr.byteswap().tobytes()
        # For quantized types, byte swap is more complex
        # (handled in original gguf_byteswap.py)
        return data


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: python3 gguf_to_q1_58.py input.gguf output-q1_58.gguf [--big-endian]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    big_endian = '--big-endian' in sys.argv

    converter = Q158Converter(input_path, output_path, big_endian)
    converter.convert()


if __name__ == "__main__":
    main()
