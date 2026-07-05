#!/usr/bin/env python3
"""
Analyze Microsoft BitNet I2_S (type 36) format.
Reverse-engineer the block structure from raw bytes.
"""

import struct
import numpy as np
import sys

def read_gguf_tensors(path):
    """Read tensor info from GGUF."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        # Skip metadata
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            vtype = struct.unpack('<I', f.read(4))[0]
            skip_value(f, vtype)

        # Read tensor info
        tensors = []
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors.append({
                'name': name, 'dims': dims, 'type': ttype, 'offset': offset
            })

        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        f.read(padding)
        data_start = f.tell()

        return tensors, data_start

def skip_value(f, vtype):
    if vtype in (0, 1, 7): f.read(1)
    elif vtype in (2, 3): f.read(2)
    elif vtype in (4, 5, 6): f.read(4)
    elif vtype in (10, 11, 12): f.read(8)
    elif vtype == 8:
        length = struct.unpack('<Q', f.read(8))[0]
        f.read(length)
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len): skip_value(f, arr_type)

def analyze_i2s_tensor(path, tensor_info, data_start, max_bytes=1024):
    """Analyze raw bytes of an I2_S tensor."""
    with open(path, 'rb') as f:
        f.seek(data_start + tensor_info['offset'])
        raw = f.read(max_bytes)

    n_elements = 1
    for d in tensor_info['dims']:
        n_elements *= d

    print(f"\nTensor: {tensor_info['name']}")
    print(f"  Dims: {tensor_info['dims']}")
    print(f"  Elements: {n_elements}")

    # Calculate bytes per element
    # I2_S = 2-bit integer = 4 values per byte
    expected_size_2bit = n_elements // 4
    print(f"  Expected size at 2-bit: {expected_size_2bit} bytes")

    # Check for block structure
    # Common block sizes: 32, 64, 128, 256 elements
    print(f"\n  First 128 bytes (hex):")
    for i in range(0, min(128, len(raw)), 16):
        hex_str = raw[i:i+16].hex()
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw[i:i+16])
        print(f"    {i:04x}: {hex_str}  {ascii_str}")

    # Look for repeating patterns (scale factors at block boundaries)
    print(f"\n  Byte value histogram (first 256 bytes):")
    hist = {}
    for b in raw[:256]:
        hist[b] = hist.get(b, 0) + 1
    # Show most common
    common = sorted(hist.items(), key=lambda x: -x[1])[:10]
    for val, count in common:
        print(f"    0x{val:02x} ({val:3d}): {count} times")

    # Try to decode as 2-bit ternary
    print(f"\n  Decoding as 2-bit ternary (first 64 values):")
    decoded = []
    for byte in raw[:16]:  # 16 bytes = 64 values
        for shift in range(0, 8, 2):
            val = (byte >> shift) & 0x03
            # Possible mappings: 0=-1, 1=0, 2=+1 or 0=0, 1=-1, 2=+1
            decoded.append(val)

    print(f"    Raw 2-bit values: {decoded}")

    # Try different ternary mappings
    mappings = [
        {0: -1, 1: 0, 2: 1, 3: 0},  # Our mapping
        {0: 0, 1: -1, 2: 1, 3: 0},  # Alternative
        {0: 0, 1: 1, 2: -1, 3: 0},  # Another
        {0: -1, 1: 1, 2: 0, 3: 0},  # Yet another
    ]

    for i, mapping in enumerate(mappings):
        ternary = [mapping[v] for v in decoded]
        neg = sum(1 for v in ternary if v == -1)
        zero = sum(1 for v in ternary if v == 0)
        pos = sum(1 for v in ternary if v == 1)
        print(f"    Mapping {i}: -1:{neg} 0:{zero} +1:{pos}")

    # Check if there's a scale factor
    print(f"\n  Looking for FP16 scale factors:")
    for offset in [0, 2, 4, 62, 64, 126, 128, 254, 256]:
        if offset + 2 <= len(raw):
            f16_bytes = raw[offset:offset+2]
            try:
                f16_val = np.frombuffer(f16_bytes, dtype=np.float16)[0]
                if 0.0001 < abs(f16_val) < 100:  # Reasonable scale range
                    print(f"    Offset {offset}: {f16_val:.6f} (0x{f16_bytes.hex()})")
            except:
                pass

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/Xilinx/models/bitnet/ggml-model-i2_s.gguf"

    print(f"Analyzing: {path}")
    tensors, data_start = read_gguf_tensors(path)

    # Find I2_S tensors (type 36)
    i2s_tensors = [t for t in tensors if t['type'] == 36]
    print(f"\nFound {len(i2s_tensors)} I2_S tensors (type 36)")

    if i2s_tensors:
        # Analyze first I2_S tensor
        analyze_i2s_tensor(path, i2s_tensors[0], data_start)

        # Also analyze a smaller one if available
        small_tensors = [t for t in i2s_tensors if t['dims'][0] * t['dims'][1] < 1000000]
        if small_tensors and small_tensors[0] != i2s_tensors[0]:
            analyze_i2s_tensor(path, small_tensors[0], data_start)

if __name__ == "__main__":
    main()
