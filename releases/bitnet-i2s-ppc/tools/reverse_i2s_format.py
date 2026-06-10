#!/usr/bin/env python3
"""
Reverse-engineer Microsoft BitNet I2_S (type 36) format from raw bytes.
Determine block structure, scale position, and ternary encoding.
"""

import struct
import numpy as np
import sys

def read_gguf_full(path):
    """Read GGUF header, metadata, and tensor info."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        # Read metadata
        metadata = {}
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            metadata[key] = value

        # Read tensor info
        tensors = []
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]

            n_elements = 1
            for d in dims:
                n_elements *= d

            tensors.append({
                'name': name, 'dims': dims, 'type': ttype,
                'offset': offset, 'n_elements': n_elements
            })

        # Get data start with alignment
        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        f.read(padding)
        data_start = f.tell()

        return metadata, tensors, data_start

def read_value(f, vtype):
    """Read GGUF value by type."""
    if vtype == 0: return struct.unpack('B', f.read(1))[0]
    elif vtype == 1: return struct.unpack('b', f.read(1))[0]
    elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
    elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7: return struct.unpack('B', f.read(1))[0] != 0
    elif vtype == 8:
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
    return None

def analyze_i2s_blocks(path, tensor_info, data_start):
    """Deep analysis of I2_S tensor to determine block structure."""
    n_elements = tensor_info['n_elements']

    # I2_S = 2 bits per weight = 4 weights per byte
    bytes_for_data = n_elements // 4

    print(f"\n{'='*60}")
    print(f"Tensor: {tensor_info['name']}")
    print(f"  Dims: {tensor_info['dims']}")
    print(f"  Elements: {n_elements:,}")
    print(f"  Raw data bytes (at 2-bit): {bytes_for_data:,}")

    with open(path, 'rb') as f:
        f.seek(data_start + tensor_info['offset'])

        # Read more data for analysis
        read_size = min(bytes_for_data + 1024, 10000)  # Extra for scale factors
        raw = f.read(read_size)

    # Try to find scale factor pattern
    # Common block sizes: 32, 64, 128, 256 elements
    # At 2-bit: 8, 16, 32, 64 bytes for data

    print(f"\n  === Block Size Detection ===")

    # Test different block sizes
    for block_weights in [32, 64, 128, 256]:
        data_bytes = block_weights // 4

        # Scale at beginning?
        scale_begin = try_detect_scale(raw, data_bytes, scale_first=True, n_elements=n_elements)
        # Scale at end?
        scale_end = try_detect_scale(raw, data_bytes, scale_first=False, n_elements=n_elements)

        if scale_begin > 0.5:
            print(f"  {block_weights} weights: scale FIRST (confidence: {scale_begin:.2f})")
        if scale_end > 0.5:
            print(f"  {block_weights} weights: scale LAST (confidence: {scale_end:.2f})")

    # Decode first few blocks with different assumptions
    print(f"\n  === Decoding Tests ===")

    # Test: 256 weights per block, FP16 scale at end (like our Q1_58)
    test_block_structure(raw, block_weights=256, scale_bytes=2, scale_first=False)

    # Test: 256 weights per block, FP16 scale at beginning
    test_block_structure(raw, block_weights=256, scale_bytes=2, scale_first=True)

    # Test: 32 weights per block (super-block like Q4_K)
    test_block_structure(raw, block_weights=32, scale_bytes=2, scale_first=True)

    # Analyze actual byte patterns
    print(f"\n  === Byte Pattern Analysis ===")

    # Look for repeating patterns that might indicate block boundaries
    pattern_analysis(raw[:1024])

    return raw

def try_detect_scale(data, data_bytes_per_block, scale_first, n_elements):
    """Try to detect if there's a scale factor pattern."""
    # FP16 scale should be reasonable values (0.001 to 10.0 typically)
    scale_bytes = 2  # FP16

    if scale_first:
        block_size = scale_bytes + data_bytes_per_block
    else:
        block_size = data_bytes_per_block + scale_bytes

    n_blocks = min(100, len(data) // block_size)
    if n_blocks < 5:
        return 0.0

    valid_scales = 0
    for i in range(n_blocks):
        if scale_first:
            scale_offset = i * block_size
        else:
            scale_offset = i * block_size + data_bytes_per_block

        if scale_offset + 2 > len(data):
            break

        try:
            scale = np.frombuffer(data[scale_offset:scale_offset+2], dtype=np.float16)[0]
            # Check if it looks like a valid scale
            if not np.isnan(scale) and not np.isinf(scale):
                if 0.0001 < abs(scale) < 100:
                    valid_scales += 1
        except:
            pass

    return valid_scales / n_blocks if n_blocks > 0 else 0.0

def test_block_structure(data, block_weights, scale_bytes, scale_first):
    """Test decoding with a specific block structure."""
    data_bytes = block_weights // 4

    if scale_first:
        block_size = scale_bytes + data_bytes
        data_offset = scale_bytes
        scale_offset = 0
    else:
        block_size = data_bytes + scale_bytes
        data_offset = 0
        scale_offset = data_bytes

    print(f"\n  Block: {block_weights} weights, scale {'first' if scale_first else 'last'}")
    print(f"  Block size: {block_size} bytes ({data_bytes} data + {scale_bytes} scale)")

    # Decode first block
    if len(data) < block_size:
        print(f"    Not enough data")
        return

    # Get scale
    scale = np.frombuffer(data[scale_offset:scale_offset+2], dtype=np.float16)[0]
    print(f"    Scale (FP16): {scale}")

    # Decode weights
    weights = []
    packed = data[data_offset:data_offset + data_bytes]

    for byte in packed:
        for shift in range(0, 8, 2):
            val = (byte >> shift) & 0x03
            weights.append(val)

    weights = np.array(weights[:block_weights])

    # Try different ternary mappings
    mappings = {
        '0=-1,1=0,2=1': {0: -1, 1: 0, 2: 1, 3: 0},
        '0=0,1=-1,2=1': {0: 0, 1: -1, 2: 1, 3: 0},
        '0=0,1=1,2=-1': {0: 0, 1: 1, 2: -1, 3: 0},
        '0=-1,1=1,2=0': {0: -1, 1: 1, 2: 0, 3: 0},
    }

    print(f"    Raw 2-bit histogram: 0:{np.sum(weights==0)} 1:{np.sum(weights==1)} 2:{np.sum(weights==2)} 3:{np.sum(weights==3)}")

    for name, mapping in mappings.items():
        ternary = np.array([mapping[v] for v in weights])
        neg = np.sum(ternary == -1)
        zero = np.sum(ternary == 0)
        pos = np.sum(ternary == 1)

        # Apply scale
        scaled = ternary.astype(np.float32) * float(scale)

        print(f"    {name}: -1:{neg} 0:{zero} +1:{pos} | range: [{scaled.min():.4f}, {scaled.max():.4f}]")

def pattern_analysis(data):
    """Look for repeating patterns in byte data."""
    # Check for common delimiters or headers
    print(f"  First 64 bytes:")
    for i in range(0, min(64, len(data)), 16):
        hex_str = data[i:i+16].hex()
        print(f"    {i:04x}: {hex_str}")

    # Check byte value distribution
    unique, counts = np.unique(np.frombuffer(data[:256], dtype=np.uint8), return_counts=True)
    top_bytes = sorted(zip(unique, counts), key=lambda x: -x[1])[:5]
    print(f"\n  Most common bytes: {[(f'0x{b:02x}', c) for b, c in top_bytes]}")

    # Check for pattern at offsets 64, 66, 128, 130, etc.
    print(f"\n  Checking FP16 values at regular intervals:")
    for offset in [64, 66, 128, 130, 256, 258]:
        if offset + 2 <= len(data):
            val = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
            if not np.isnan(val) and 0.0001 < abs(val) < 100:
                print(f"    Offset {offset}: {val:.6f}")

def compute_theoretical_size(n_elements, block_weights, scale_bytes):
    """Calculate expected tensor size."""
    data_bytes = block_weights // 4
    block_size = data_bytes + scale_bytes
    n_blocks = (n_elements + block_weights - 1) // block_weights
    return n_blocks * block_size

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/Xilinx/models/bitnet/ggml-model-i2_s.gguf"

    print(f"Reverse-engineering I2_S format: {path}")

    metadata, tensors, data_start = read_gguf_full(path)

    # Print key metadata
    print("\n=== Model Metadata ===")
    for key in ['general.architecture', 'general.name', 'llama.block_count',
                'llama.embedding_length', 'llama.vocab_size']:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")

    # Find I2_S tensors (type 36)
    i2s_tensors = [t for t in tensors if t['type'] == 36]
    print(f"\n=== Found {len(i2s_tensors)} I2_S tensors ===")

    # Analyze sizes to find block structure
    print("\n=== Size Analysis ===")

    # Get actual file positions and sizes
    with open(path, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()

    print(f"  File size: {file_size:,} bytes")
    print(f"  Data start: {data_start:,} bytes")

    # Calculate total I2_S data
    total_elements = sum(t['n_elements'] for t in i2s_tensors)
    theoretical_2bit = total_elements // 4

    print(f"  Total I2_S elements: {total_elements:,}")
    print(f"  Theoretical size (pure 2-bit): {theoretical_2bit:,} bytes")

    # Calculate actual I2_S data size from offsets
    sorted_tensors = sorted(tensors, key=lambda t: t['offset'])

    # Find the actual I2_S data range
    for block_size in [256, 128, 64, 32]:
        scale_bytes = 2
        data_bytes = block_size // 4
        total_block_size = data_bytes + scale_bytes
        n_blocks = total_elements // block_size
        expected_size = n_blocks * total_block_size

        print(f"\n  If block_size={block_size}: expected {expected_size:,} bytes ({expected_size/theoretical_2bit:.3f}x pure 2-bit)")

    # Analyze a few representative tensors
    if i2s_tensors:
        # Analyze first tensor
        analyze_i2s_blocks(path, i2s_tensors[0], data_start)

        # Analyze a smaller tensor if available
        small = [t for t in i2s_tensors if t['n_elements'] < 100000]
        if small:
            analyze_i2s_blocks(path, small[0], data_start)

if __name__ == "__main__":
    main()
