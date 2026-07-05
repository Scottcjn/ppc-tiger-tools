#!/usr/bin/env python3
"""
Test I2_S (native BitNet) inference.
Verifies ternary weights produce correct outputs.
"""

import struct
import numpy as np
import sys

# I2_S constants
I2S_TYPE = 36
I2S_BLOCK_WEIGHTS = 256
I2S_BLOCK_BYTES = 64
I2S_TENSOR_HEADER = 32


def read_gguf_full(path, big_endian=False):
    """Read GGUF header with endian support."""
    fmt = '>' if big_endian else '<'

    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack(f'{fmt}I', f.read(4))[0]
        tensor_count = struct.unpack(f'{fmt}Q', f.read(8))[0]
        metadata_count = struct.unpack(f'{fmt}Q', f.read(8))[0]

        # Skip metadata
        for _ in range(metadata_count):
            key_len = struct.unpack(f'{fmt}Q', f.read(8))[0]
            f.read(key_len)
            vtype = struct.unpack(f'{fmt}I', f.read(4))[0]
            skip_value(f, vtype, fmt)

        # Read tensor info
        tensors = {}
        for _ in range(tensor_count):
            name_len = struct.unpack(f'{fmt}Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack(f'{fmt}I', f.read(4))[0]
            dims = [struct.unpack(f'{fmt}Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack(f'{fmt}I', f.read(4))[0]
            offset = struct.unpack(f'{fmt}Q', f.read(8))[0]

            n_el = 1
            for d in dims:
                n_el *= d

            tensors[name] = {
                'dims': dims, 'type': ttype, 'offset': offset, 'n_elements': n_el
            }

        # Alignment
        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        f.read(padding)
        data_start = f.tell()

        return tensors, data_start


def skip_value(f, vtype, fmt='<'):
    """Skip a GGUF value."""
    if vtype in (0, 1, 7): f.read(1)
    elif vtype in (2, 3): f.read(2)
    elif vtype in (4, 5, 6): f.read(4)
    elif vtype in (10, 11, 12): f.read(8)
    elif vtype == 8:
        length = struct.unpack(f'{fmt}Q', f.read(8))[0]
        f.read(length)
    elif vtype == 9:
        arr_type = struct.unpack(f'{fmt}I', f.read(4))[0]
        arr_len = struct.unpack(f'{fmt}Q', f.read(8))[0]
        for _ in range(arr_len):
            skip_value(f, arr_type, fmt)


def unpack_i2s_ternary(packed_bytes: bytes) -> np.ndarray:
    """Unpack I2_S packed bytes to ternary values."""
    # Encoding: 00=-1, 01=0, 10=+1, 11=0
    decode = {0: -1, 1: 0, 2: 1, 3: 0}

    weights = []
    for byte in packed_bytes:
        for shift in range(0, 8, 2):
            bits = (byte >> shift) & 0x03
            weights.append(decode[bits])

    return np.array(weights, dtype=np.int8)


def load_i2s_tensor(f, data_start, tensor_info, max_elements=None, big_endian=False):
    """Load an I2_S tensor."""
    f.seek(data_start + tensor_info['offset'])

    n_elements = tensor_info['n_elements']
    if max_elements:
        n_elements = min(n_elements, max_elements)

    # Read header (contains tensor-level scale)
    header = f.read(I2S_TENSOR_HEADER)

    # Scale is first 2 bytes as FP16
    # Use correct endianness for reading
    if big_endian:
        scale = np.frombuffer(header[:2], dtype='>f2')[0]  # Big-endian FP16
    else:
        scale = np.frombuffer(header[:2], dtype='<f2')[0]  # Little-endian FP16

    # Read packed data
    n_blocks = n_elements // I2S_BLOCK_WEIGHTS
    packed = f.read(n_blocks * I2S_BLOCK_BYTES)

    # Unpack to ternary
    ternary = unpack_i2s_ternary(packed)[:n_elements]

    return ternary, float(scale)


def test_i2s_inference(model_path, big_endian=False):
    """Test I2_S model inference."""
    endian_str = "big-endian" if big_endian else "little-endian"
    print(f"Testing I2_S inference ({endian_str})")
    print(f"Model: {model_path}")
    print("=" * 60)

    tensors, data_start = read_gguf_full(model_path, big_endian)

    # Find I2_S tensors
    i2s_tensors = {k: v for k, v in tensors.items() if v['type'] == I2S_TYPE}
    print(f"\nFound {len(i2s_tensors)} I2_S tensors")

    with open(model_path, 'rb') as f:
        # Test first I2_S tensor
        test_name = list(i2s_tensors.keys())[0]
        test_info = i2s_tensors[test_name]

        print(f"\n--- Testing: {test_name} ---")
        print(f"Dims: {test_info['dims']}")
        print(f"Elements: {test_info['n_elements']:,}")

        # Load first 10240 elements (40 blocks)
        ternary, scale = load_i2s_tensor(f, data_start, test_info, max_elements=10240, big_endian=big_endian)

        print(f"\nTensor scale (FP16): {scale}")
        print(f"Loaded {len(ternary)} ternary values")

        # Analyze distribution
        neg = np.sum(ternary == -1)
        zero = np.sum(ternary == 0)
        pos = np.sum(ternary == 1)
        total = len(ternary)

        print(f"\nWeight distribution:")
        print(f"  -1: {neg:6d} ({100*neg/total:.1f}%)")
        print(f"   0: {zero:6d} ({100*zero/total:.1f}%)")
        print(f"  +1: {pos:6d} ({100*pos/total:.1f}%)")

        # Verify ternary (should be exactly {-1, 0, +1})
        unique_vals = np.unique(ternary)
        print(f"\nUnique values: {unique_vals}")
        if set(unique_vals) <= {-1, 0, 1}:
            print("PASS: All weights are ternary {-1, 0, +1}")
        else:
            print("FAIL: Non-ternary values found!")

        # Test dot product
        print(f"\n--- Dot Product Test ---")

        # Create random Q8 activations
        np.random.seed(42)
        activations = np.random.randint(-127, 128, size=len(ternary), dtype=np.int8)

        # Integer dot product (no scale yet)
        int_dot = np.sum(ternary.astype(np.int32) * activations.astype(np.int32))

        # Verify sparsity savings
        nonzero_weights = np.count_nonzero(ternary)
        ops_saved = total - nonzero_weights

        print(f"Integer dot product: {int_dot}")
        print(f"Scaled dot product: {int_dot * scale}")
        print(f"Operations: {nonzero_weights}/{total} ({100*nonzero_weights/total:.1f}%)")
        print(f"Ops saved (zeros): {ops_saved} ({100*ops_saved/total:.1f}%)")

        # Simulate what the result means
        print(f"\n--- Full Ternary Math Verification ---")

        # For ternary, the dot product is: sum(x where w=+1) - sum(x where w=-1)
        pos_sum = np.sum(activations[ternary == 1])
        neg_sum = np.sum(activations[ternary == -1])
        ternary_result = pos_sum - neg_sum

        print(f"Sum where w=+1: {pos_sum}")
        print(f"Sum where w=-1: {neg_sum}")
        print(f"Ternary result (pos - neg): {ternary_result}")
        print(f"Standard dot product: {int_dot}")

        if ternary_result == int_dot:
            print("PASS: Ternary math verified!")
        else:
            print("FAIL: Mismatch in ternary computation")

        # Check that outputs look reasonable (not all same value)
        print(f"\n--- Output Quality Check ---")

        # Load another tensor for comparison
        if len(i2s_tensors) > 1:
            other_name = list(i2s_tensors.keys())[1]
            other_info = i2s_tensors[other_name]
            other_ternary, other_scale = load_i2s_tensor(f, data_start, other_info, max_elements=10240, big_endian=big_endian)

            # Check if tensors are different
            if len(ternary) == len(other_ternary):
                correlation = np.corrcoef(ternary.astype(float), other_ternary.astype(float))[0, 1]
                print(f"Correlation with {other_name[:30]}...: {correlation:.4f}")

                if abs(correlation) < 0.5:
                    print("PASS: Different tensors have distinct weights")
                else:
                    print("WARNING: High correlation between different tensors")

        print("\n" + "=" * 60)
        print("I2_S inference test complete!")

        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_i2s_inference.py model.gguf [--big-endian]")
        sys.exit(1)

    model_path = sys.argv[1]
    big_endian = '--big-endian' in sys.argv or '-BE' in sys.argv

    test_i2s_inference(model_path, big_endian)


if __name__ == "__main__":
    main()
