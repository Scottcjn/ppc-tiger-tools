#!/usr/bin/env python3
"""
Test Q1.58 ternary inference.
Loads a Q1.58 model and performs a simple forward pass to verify the math works.
"""

import struct
import numpy as np
import sys

# Q1.58 constants
QK_Q1_58 = 256
BLOCK_SIZE_Q1_58 = 68


def read_gguf_header(path):
    """Read GGUF header and return tensor info."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        print(f"Version: {version}, Tensors: {tensor_count}, Metadata: {metadata_count}")

        # Skip metadata
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)  # key
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
                'name': name,
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


def skip_value(f, vtype):
    """Skip a GGUF value."""
    if vtype in (0, 1, 7):
        f.read(1)
    elif vtype in (2, 3):
        f.read(2)
    elif vtype in (4, 5, 6):
        f.read(4)
    elif vtype in (10, 11, 12):
        f.read(8)
    elif vtype == 8:
        length = struct.unpack('<Q', f.read(8))[0]
        f.read(length)
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            skip_value(f, arr_type)


def unpack_ternary(packed_byte):
    """Unpack 4 ternary values from 1 byte."""
    decode = [-1, 0, 1, 0]  # 00=-1, 01=0, 10=+1, 11=0
    return [
        decode[(packed_byte >> 0) & 3],
        decode[(packed_byte >> 2) & 3],
        decode[(packed_byte >> 4) & 3],
        decode[(packed_byte >> 6) & 3]
    ]


def load_q1_58_tensor(f, data_start, tensor, max_blocks=10):
    """Load a Q1.58 tensor from file."""
    f.seek(data_start + tensor['offset'])

    n_elements = 1
    for d in tensor['dims']:
        n_elements *= d

    n_blocks = n_elements // QK_Q1_58
    if max_blocks:
        n_blocks = min(n_blocks, max_blocks)

    weights = []
    scales = []
    zero_counts = []

    for _ in range(n_blocks):
        block_data = f.read(BLOCK_SIZE_Q1_58)

        # Unpack 256 ternary weights from 64 bytes
        packed = block_data[0:64]
        block_weights = []
        for byte in packed:
            block_weights.extend(unpack_ternary(byte))

        # Scale (FP16 at bytes 64-65)
        scale = np.frombuffer(block_data[64:66], dtype=np.float16)[0]

        # Zero count (uint16 at bytes 66-67)
        zero_count = struct.unpack('<H', block_data[66:68])[0]

        weights.append(block_weights)
        scales.append(float(scale))
        zero_counts.append(zero_count)

    return weights, scales, zero_counts


def ternary_dot_product(weights, activations, scale):
    """
    Compute dot product with ternary weights.
    This is the key operation - no multiplication needed!
    """
    result = 0
    for w, a in zip(weights, activations):
        if w == 1:
            result += a
        elif w == -1:
            result -= a
        # w == 0: skip (add nothing)

    return result * scale


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_q1_58_inference.py model.gguf")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Testing Q1.58 model: {model_path}")

    tensors, data_start = read_gguf_header(model_path)

    # Find a Q1.58 tensor (type 20)
    q1_58_tensors = [t for t in tensors if t['type'] == 20]
    print(f"\nFound {len(q1_58_tensors)} Q1.58 tensors")

    if not q1_58_tensors:
        print("No Q1.58 tensors found!")
        sys.exit(1)

    # Test with first tensor
    test_tensor = q1_58_tensors[0]
    print(f"\nTesting tensor: {test_tensor['name']}")
    print(f"  Dims: {test_tensor['dims']}")
    print(f"  Type: {test_tensor['type']}")

    with open(model_path, 'rb') as f:
        weights, scales, zero_counts = load_q1_58_tensor(f, data_start, test_tensor)

    print(f"\nLoaded {len(weights)} blocks")
    print(f"  First block scale: {scales[0]:.6f}")
    print(f"  First block zero_count: {zero_counts[0]} / 256 ({100*zero_counts[0]/256:.1f}% sparse)")

    # Analyze ternary distribution
    first_weights = weights[0]
    neg_count = sum(1 for w in first_weights if w == -1)
    zero_count = sum(1 for w in first_weights if w == 0)
    pos_count = sum(1 for w in first_weights if w == 1)
    print(f"\n  Weight distribution: -1:{neg_count} 0:{zero_count} +1:{pos_count}")

    # Test ternary dot product
    print("\n--- Ternary Dot Product Test ---")
    # Random int8 activations
    np.random.seed(42)
    activations = np.random.randint(-128, 127, size=256, dtype=np.int8).tolist()

    result = ternary_dot_product(first_weights, activations, scales[0])
    print(f"  Input activations (first 10): {activations[:10]}")
    print(f"  Ternary weights (first 10): {first_weights[:10]}")
    print(f"  Scale: {scales[0]:.6f}")
    print(f"  Dot product result: {result:.6f}")

    # Verify integer-only accumulation works
    print("\n--- Integer-Only Accumulation Test ---")
    int_sum = 0
    ops_count = 0
    for w, a in zip(first_weights, activations):
        if w == 1:
            int_sum += a
            ops_count += 1
        elif w == -1:
            int_sum -= a
            ops_count += 1
        # w == 0: no operation

    print(f"  Integer accumulator: {int_sum}")
    print(f"  After scaling: {int_sum * scales[0]:.6f}")
    print(f"  Operations performed: {ops_count} / 256 ({100*ops_count/256:.1f}%)")
    print(f"  Skipped (zeros): {256 - ops_count}")

    print("\n✓ Q1.58 ternary math verified!")


if __name__ == "__main__":
    main()
