#!/usr/bin/env python3
"""
GGUF Tensor Comparator - Compare tensors between LE and BE GGUF files.
Useful for debugging byte-swap issues.
"""

import struct
import sys
from pathlib import Path

def read_le_u32(f):
    return struct.unpack('<I', f.read(4))[0]

def read_le_u64(f):
    return struct.unpack('<Q', f.read(8))[0]

def read_be_u32(f):
    return struct.unpack('>I', f.read(4))[0]

def read_be_u64(f):
    return struct.unpack('>Q', f.read(8))[0]

def read_string_le(f):
    length = read_le_u64(f)
    return f.read(length).decode('utf-8')

def read_string_be(f):
    length = read_be_u64(f)
    return f.read(length).decode('utf-8')

def skip_value(f, vtype, is_be=False):
    """Skip over a GGUF value"""
    read_u32 = read_be_u32 if is_be else read_le_u32
    read_u64 = read_be_u64 if is_be else read_le_u64
    read_string = read_string_be if is_be else read_string_le

    if vtype in (0, 1, 7):  # uint8, int8, bool
        f.read(1)
    elif vtype in (2, 3):  # uint16, int16
        f.read(2)
    elif vtype in (4, 5, 6):  # uint32, int32, float32
        f.read(4)
    elif vtype in (10, 11, 12):  # uint64, int64, float64
        f.read(8)
    elif vtype == 8:  # string
        read_string(f)
    elif vtype == 9:  # array
        arr_type = read_u32(f)
        arr_len = read_u64(f)
        for _ in range(arr_len):
            skip_value(f, arr_type, is_be)

def parse_gguf_header(path, is_be=False):
    """Parse GGUF header and return tensor info"""
    read_u32 = read_be_u32 if is_be else read_le_u32
    read_u64 = read_be_u64 if is_be else read_le_u64
    read_string = read_string_be if is_be else read_string_le

    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid magic: {magic}")

        version = read_u32(f)
        tensor_count = read_u64(f)
        metadata_count = read_u64(f)

        print(f"Version: {version}, Tensors: {tensor_count}, Metadata: {metadata_count}")

        # Skip metadata
        for _ in range(metadata_count):
            key = read_string(f)
            vtype = read_u32(f)
            skip_value(f, vtype, is_be)

        # Read tensor info
        tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = read_u32(f)
            dims = [read_u64(f) for _ in range(n_dims)]
            ttype = read_u32(f)
            offset = read_u64(f)
            tensors.append({
                'name': name,
                'n_dims': n_dims,
                'dims': dims,
                'type': ttype,
                'offset': offset
            })

        # Get data start
        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        f.read(padding)
        data_start = f.tell()

        return tensors, data_start

def compare_tensor_bytes(le_file, be_file, tensor_name, is_f32=False, is_f16=False, count=16):
    """Compare bytes of a specific tensor between LE and BE files"""
    le_tensors, le_data_start = parse_gguf_header(le_file, is_be=False)
    be_tensors, be_data_start = parse_gguf_header(be_file, is_be=True)

    le_tensor = next((t for t in le_tensors if t['name'] == tensor_name), None)
    be_tensor = next((t for t in be_tensors if t['name'] == tensor_name), None)

    if not le_tensor or not be_tensor:
        print(f"Tensor {tensor_name} not found")
        return

    print(f"\nTensor: {tensor_name}")
    print(f"  LE offset: {le_tensor['offset']}, BE offset: {be_tensor['offset']}")
    print(f"  Type: {le_tensor['type']}, Dims: {le_tensor['dims']}")

    with open(le_file, 'rb') as f:
        f.seek(le_data_start + le_tensor['offset'])
        le_bytes = f.read(count * 4 if is_f32 else count * 2 if is_f16 else count)

    with open(be_file, 'rb') as f:
        f.seek(be_data_start + be_tensor['offset'])
        be_bytes = f.read(count * 4 if is_f32 else count * 2 if is_f16 else count)

    print(f"\n  LE bytes (first {len(le_bytes)}): {le_bytes.hex()}")
    print(f"  BE bytes (first {len(be_bytes)}): {be_bytes.hex()}")

    if is_f32:
        le_floats = struct.unpack(f'<{count}f', le_bytes)
        be_floats = struct.unpack(f'>{count}f', be_bytes)
        print(f"\n  LE floats: {le_floats[:8]}...")
        print(f"  BE floats: {be_floats[:8]}...")
        # Check if they match
        match = all(abs(a - b) < 1e-6 for a, b in zip(le_floats, be_floats))
        print(f"  Match: {match}")

    if is_f16:
        # Decode f16 values
        le_f16 = []
        for i in range(0, len(le_bytes), 2):
            val = struct.unpack('<e', le_bytes[i:i+2])[0]
            le_f16.append(val)
        be_f16 = []
        for i in range(0, len(be_bytes), 2):
            val = struct.unpack('>e', be_bytes[i:i+2])[0]
            be_f16.append(val)
        print(f"\n  LE f16 values: {le_f16[:8]}...")
        print(f"  BE f16 values: {be_f16[:8]}...")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 gguf_compare.py <le.gguf> <be.gguf> [tensor_name]")
        print("\nCompares tensor data between little-endian and big-endian GGUF files.")
        sys.exit(1)

    le_file = sys.argv[1]
    be_file = sys.argv[2]
    tensor_name = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"LE file: {le_file}")
    le_tensors, le_data_start = parse_gguf_header(le_file, is_be=False)
    print(f"Data starts at: {le_data_start}\n")

    print(f"BE file: {be_file}")
    be_tensors, be_data_start = parse_gguf_header(be_file, is_be=True)
    print(f"Data starts at: {be_data_start}\n")

    # Find F32 tensors (type 0)
    f32_tensors = [t for t in le_tensors if t['type'] == 0]
    print(f"F32 tensors: {len(f32_tensors)}")

    # Compare first F32 tensor
    if f32_tensors:
        first_f32 = f32_tensors[0]['name']
        compare_tensor_bytes(le_file, be_file, first_f32, is_f32=True, count=16)

    # Find Q4_K tensors (type 12)
    q4k_tensors = [t for t in le_tensors if t['type'] == 12]
    print(f"\nQ4_K tensors: {len(q4k_tensors)}")

    # Compare first Q4_K tensor (check d/dmin f16 values)
    if q4k_tensors:
        first_q4k = q4k_tensors[0]['name']
        print(f"\nChecking Q4_K tensor: {first_q4k}")
        with open(le_file, 'rb') as f:
            le_t = next(t for t in le_tensors if t['name'] == first_q4k)
            f.seek(le_data_start + le_t['offset'])
            le_block = f.read(144)  # First Q4_K block

        with open(be_file, 'rb') as f:
            be_t = next(t for t in be_tensors if t['name'] == first_q4k)
            f.seek(be_data_start + be_t['offset'])
            be_block = f.read(144)

        print(f"  LE block first 16 bytes: {le_block[:16].hex()}")
        print(f"  BE block first 16 bytes: {be_block[:16].hex()}")

        # Decode d and dmin
        le_d = struct.unpack('<e', le_block[0:2])[0]
        le_dmin = struct.unpack('<e', le_block[2:4])[0]
        be_d = struct.unpack('>e', be_block[0:2])[0]
        be_dmin = struct.unpack('>e', be_block[2:4])[0]

        print(f"  LE: d={le_d:.6f}, dmin={le_dmin:.6f}")
        print(f"  BE: d={be_d:.6f}, dmin={be_dmin:.6f}")
        print(f"  Match: d={abs(le_d-be_d)<1e-4}, dmin={abs(le_dmin-be_dmin)<1e-4}")

if __name__ == "__main__":
    main()
