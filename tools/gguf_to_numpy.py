#!/usr/bin/env python3
"""
GGUF to NumPy Weight Extractor

Extracts weights from a GGUF model file and saves them as NumPy .npz format.
NumPy handles endianness automatically, so the extracted weights can be loaded
on any architecture (x86, ARM, PowerPC).

Usage:
    python3 gguf_to_numpy.py tinyllama-1.1b-q4.gguf tinyllama_weights.npz

Author: ppc-tiger-tools project
License: MIT
December 2025
"""

import struct
import sys
import os
import numpy as np
from pathlib import Path

# GGUF constants
GGUF_MAGIC = b'GGUF'

# GGUF value types
GGUF_TYPES = {
    0: ('uint8', 'B', 1),
    1: ('int8', 'b', 1),
    2: ('uint16', 'H', 2),
    3: ('int16', 'h', 2),
    4: ('uint32', 'I', 4),
    5: ('int32', 'i', 4),
    6: ('float32', 'f', 4),
    7: ('bool', '?', 1),
    8: ('string', None, None),
    9: ('array', None, None),
    10: ('uint64', 'Q', 8),
    11: ('int64', 'q', 8),
    12: ('float64', 'd', 8),
}

# GGML tensor types
GGML_TYPES = {
    0: ('F32', 4, 1),      # 4 bytes per element
    1: ('F16', 2, 1),      # 2 bytes per element
    2: ('Q4_0', 18, 32),   # 18 bytes per 32 elements
    3: ('Q4_1', 20, 32),
    6: ('Q5_0', 22, 32),
    7: ('Q5_1', 24, 32),
    8: ('Q8_0', 34, 32),
    9: ('Q8_1', 36, 32),
    10: ('Q2_K', 84, 256),
    11: ('Q3_K', 110, 256),
    12: ('Q4_K', 144, 256),
    13: ('Q5_K', 176, 256),
    14: ('Q6_K', 210, 256),
    15: ('Q8_K', 292, 256),
    16: ('I8', 1, 1),
    17: ('I16', 2, 1),
    18: ('I32', 4, 1),
}


def dequantize_q4_0(data, n_elements):
    """Dequantize Q4_0 format to float32"""
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    offset = 0
    for block in range(n_blocks):
        if offset + 18 > len(data):
            break

        # Read scale (f16)
        scale_bytes = data[offset:offset+2]
        scale = np.frombuffer(scale_bytes, dtype=np.float16)[0]
        offset += 2

        # Read 16 bytes of weights (32 x 4-bit values)
        weights = data[offset:offset+16]
        offset += 16

        for i in range(16):
            if block * 32 + i * 2 >= n_elements:
                break
            byte = weights[i]
            v0 = (byte & 0x0F) - 8  # Low nibble
            v1 = ((byte >> 4) & 0x0F) - 8  # High nibble

            idx0 = block * 32 + i * 2
            idx1 = block * 32 + i * 2 + 1

            if idx0 < n_elements:
                result[idx0] = float(scale) * v0
            if idx1 < n_elements:
                result[idx1] = float(scale) * v1

    return result


def dequantize_q4_k(data, n_elements):
    """Dequantize Q4_K format to float32 (simplified)"""
    block_size = 256
    n_blocks = (n_elements + block_size - 1) // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    bytes_per_block = 144  # Q4_K block size
    offset = 0

    for block in range(n_blocks):
        if offset + bytes_per_block > len(data):
            break

        # Read d and dmin (f16)
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        dmin = np.frombuffer(data[offset+2:offset+4], dtype=np.float16)[0]
        offset += 4

        # Read scales (12 bytes for 8 sub-blocks)
        scales_raw = data[offset:offset+12]
        offset += 12

        # Read quantized values (128 bytes = 256 x 4-bit)
        qs = data[offset:offset+128]
        offset += 128

        # Simplified dequantization (approximate)
        for i in range(128):
            if block * 256 + i * 2 >= n_elements:
                break
            byte = qs[i]
            v0 = (byte & 0x0F)
            v1 = ((byte >> 4) & 0x0F)

            idx0 = block * 256 + i * 2
            idx1 = block * 256 + i * 2 + 1

            # Simplified: use global scale
            if idx0 < n_elements:
                result[idx0] = float(d) * (v0 - 8)
            if idx1 < n_elements:
                result[idx1] = float(d) * (v1 - 8)

    return result


def dequantize_q6_k(data, n_elements):
    """Dequantize Q6_K format to float32 (simplified)"""
    block_size = 256
    n_blocks = (n_elements + block_size - 1) // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    bytes_per_block = 210
    offset = 0

    for block in range(n_blocks):
        if offset + bytes_per_block > len(data):
            break

        # Q6_K structure: ql(128) + qh(64) + scales(16) + d(2)
        ql = data[offset:offset+128]
        offset += 128
        qh = data[offset:offset+64]
        offset += 64
        scales = data[offset:offset+16]
        offset += 16
        d = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        offset += 2

        # Simplified dequantization
        for i in range(128):
            if block * 256 + i * 2 >= n_elements:
                break

            idx0 = block * 256 + i * 2
            idx1 = block * 256 + i * 2 + 1

            # Get 6-bit values (simplified)
            q_lo = ql[i]
            v0 = (q_lo & 0x3F) - 32
            v1 = ((q_lo >> 6) | ((qh[i // 2] >> ((i % 2) * 4)) & 0x03) << 2) - 32

            if idx0 < n_elements:
                result[idx0] = float(d) * v0
            if idx1 < n_elements:
                result[idx1] = float(d) * v1

    return result


class GGUFReader:
    def __init__(self, path):
        self.path = Path(path)
        self.file = open(path, 'rb')
        self.metadata = {}
        self.tensors = []

    def read_u32(self):
        return struct.unpack('<I', self.file.read(4))[0]

    def read_u64(self):
        return struct.unpack('<Q', self.file.read(8))[0]

    def read_i32(self):
        return struct.unpack('<i', self.file.read(4))[0]

    def read_f32(self):
        return struct.unpack('<f', self.file.read(4))[0]

    def read_string(self):
        length = self.read_u64()
        return self.file.read(length).decode('utf-8')

    def read_value(self, vtype):
        if vtype == 0:  # uint8
            return struct.unpack('B', self.file.read(1))[0]
        elif vtype == 1:  # int8
            return struct.unpack('b', self.file.read(1))[0]
        elif vtype == 2:  # uint16
            return struct.unpack('<H', self.file.read(2))[0]
        elif vtype == 3:  # int16
            return struct.unpack('<h', self.file.read(2))[0]
        elif vtype == 4:  # uint32
            return self.read_u32()
        elif vtype == 5:  # int32
            return self.read_i32()
        elif vtype == 6:  # float32
            return self.read_f32()
        elif vtype == 7:  # bool
            return struct.unpack('?', self.file.read(1))[0]
        elif vtype == 8:  # string
            return self.read_string()
        elif vtype == 9:  # array
            arr_type = self.read_u32()
            arr_len = self.read_u64()
            return [self.read_value(arr_type) for _ in range(arr_len)]
        elif vtype == 10:  # uint64
            return self.read_u64()
        elif vtype == 11:  # int64
            return struct.unpack('<q', self.file.read(8))[0]
        elif vtype == 12:  # float64
            return struct.unpack('<d', self.file.read(8))[0]
        else:
            raise ValueError(f"Unknown type: {vtype}")

    def read_header(self):
        magic = self.file.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        version = self.read_u32()
        n_tensors = self.read_u64()
        n_kv = self.read_u64()

        print(f"GGUF Version: {version}")
        print(f"Tensors: {n_tensors}")
        print(f"Metadata: {n_kv}")

        # Read metadata
        for _ in range(n_kv):
            key = self.read_string()
            vtype = self.read_u32()
            value = self.read_value(vtype)
            self.metadata[key] = value

        # Read tensor info
        for _ in range(n_tensors):
            name = self.read_string()
            n_dims = self.read_u32()
            dims = [self.read_u64() for _ in range(n_dims)]
            ttype = self.read_u32()
            offset = self.read_u64()

            self.tensors.append({
                'name': name,
                'dims': dims,
                'type': ttype,
                'offset': offset
            })

        # Align to 32 bytes
        pos = self.file.tell()
        padding = (32 - (pos % 32)) % 32
        self.file.read(padding)

        self.data_start = self.file.tell()

    def extract_tensor(self, tensor):
        """Extract and dequantize a tensor"""
        name = tensor['name']
        dims = tensor['dims']
        ttype = tensor['type']
        offset = tensor['offset']

        n_elements = 1
        for d in dims:
            n_elements *= d

        # Calculate size
        if ttype not in GGML_TYPES:
            print(f"  Warning: Unknown type {ttype} for {name}, skipping")
            return None

        type_name, bytes_per_block, elements_per_block = GGML_TYPES[ttype]
        n_blocks = (n_elements + elements_per_block - 1) // elements_per_block
        data_size = n_blocks * bytes_per_block

        # Read data
        self.file.seek(self.data_start + offset)
        data = self.file.read(data_size)

        # Dequantize based on type
        if ttype == 0:  # F32
            result = np.frombuffer(data, dtype=np.float32)
        elif ttype == 1:  # F16
            result = np.frombuffer(data, dtype=np.float16).astype(np.float32)
        elif ttype == 2:  # Q4_0
            result = dequantize_q4_0(data, n_elements)
        elif ttype == 12:  # Q4_K
            result = dequantize_q4_k(data, n_elements)
        elif ttype == 14:  # Q6_K
            result = dequantize_q6_k(data, n_elements)
        else:
            print(f"  Warning: Unsupported quant type {type_name} for {name}, using zeros")
            result = np.zeros(n_elements, dtype=np.float32)

        # Reshape
        if len(dims) > 1:
            result = result.reshape(dims[::-1])  # GGUF uses reverse dim order

        return result

    def close(self):
        self.file.close()


def extract_weights(input_path, output_path):
    """Extract all weights from GGUF to NumPy format"""
    print(f"Reading: {input_path}")

    reader = GGUFReader(input_path)
    reader.read_header()

    # Print key metadata
    print("\nModel Info:")
    for key in ['general.architecture', 'general.name', 'llama.block_count',
                'llama.embedding_length', 'llama.attention.head_count']:
        if key in reader.metadata:
            print(f"  {key}: {reader.metadata[key]}")

    # Extract tensors
    weights = {}
    print(f"\nExtracting {len(reader.tensors)} tensors...")

    for i, tensor in enumerate(reader.tensors):
        name = tensor['name']
        ttype = tensor['type']
        type_name = GGML_TYPES.get(ttype, ('?', 0, 0))[0]

        print(f"  [{i+1}/{len(reader.tensors)}] {name} ({type_name})", end='')

        data = reader.extract_tensor(tensor)
        if data is not None:
            weights[name] = data
            print(f" -> {data.shape}")
        else:
            print(" -> SKIPPED")

    reader.close()

    # Convert to float16 to reduce memory footprint
    print("\nConverting to float16...")
    for name, arr in weights.items():
        if arr.dtype == np.float32:
            weights[name] = arr.astype(np.float16)

    # Save as .npz
    print(f"Saving to: {output_path}")
    # Use uncompressed format for mmap compatibility on target systems
    np.savez(output_path, **weights, **{'__metadata__': str(reader.metadata)})

    output_size = os.path.getsize(output_path)
    print(f"Output size: {output_size / (1024*1024):.1f} MB")
    print("Done!")


def main():
    if len(sys.argv) != 3:
        print("GGUF to NumPy Weight Extractor")
        print()
        print("Usage: python3 gguf_to_numpy.py <input.gguf> <output.npz>")
        print()
        print("Extracts and dequantizes weights from GGUF format to NumPy.")
        print("NumPy .npz files are endian-safe and work on any architecture.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    extract_weights(input_path, output_path)


if __name__ == "__main__":
    main()
