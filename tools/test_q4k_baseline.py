#!/usr/bin/env python3
"""Quick test of Q4_K embedding quality for comparison."""

import struct
import numpy as np
import sys

def read_gguf_header(path):
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
        tensors = {}
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {'dims': dims, 'type': ttype, 'offset': offset}

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

def dequantize_q4_k_block(block):
    """Dequantize one Q4_K block (256 values from 144 bytes)."""
    d = np.frombuffer(block[0:2], dtype=np.float16)[0]
    dmin = np.frombuffer(block[2:4], dtype=np.float16)[0]
    scales_raw = block[4:16]
    qs = block[16:144]

    result = np.zeros(256, dtype=np.float32)

    for j in range(256):
        is_high = j >= 128
        j_adj = j - 128 if is_high else j
        scale_idx = j_adj // 32
        q_idx = j_adj // 2
        shift = (j_adj % 2) * 4

        # Get scale and min for this group
        sc = scales_raw[scale_idx % 4] & 0x3f
        m = scales_raw[4 + scale_idx % 4] & 0x3f

        q = (qs[q_idx + (64 if is_high else 0)] >> shift) & 0x0f
        result[j] = float(d) * sc * q - float(dmin) * m

    return result

def load_q4k_tensor_partial(f, data_start, tensor_info, max_blocks=100):
    """Load first few blocks of a Q4_K tensor."""
    f.seek(data_start + tensor_info['offset'])

    n_elements = 1
    for d in tensor_info['dims']:
        n_elements *= d

    n_blocks = min(n_elements // 256, max_blocks)

    all_weights = []
    for _ in range(n_blocks):
        block = f.read(144)
        weights = dequantize_q4_k_block(block)
        all_weights.extend(weights)

    return np.array(all_weights)

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/scott/models/tinyllama-1.1b-q4.gguf"
    print(f"Testing Q4_K model: {model_path}")

    tensors, data_start = read_gguf_header(model_path)

    with open(model_path, 'rb') as f:
        # Load partial embeddings
        emb_info = tensors['token_embd.weight']
        print(f"\ntoken_embd.weight: {emb_info['dims']}, type={emb_info['type']}")

        weights = load_q4k_tensor_partial(f, data_start, emb_info, max_blocks=100)
        print(f"Loaded {len(weights)} values")
        print(f"  Range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")
        print(f"  Nonzero: {np.count_nonzero(weights)}/{len(weights)}")

        # Compare to Q1.58 output weights
        out_info = tensors['output.weight']
        print(f"\noutput.weight: {out_info['dims']}, type={out_info['type']}")

        # Just check the value range
        if out_info['type'] == 14:  # Q6_K
            print("  Type is Q6_K")
        elif out_info['type'] == 12:  # Q4_K
            out_weights = load_q4k_tensor_partial(f, data_start, out_info, max_blocks=100)
            print(f"  Loaded {len(out_weights)} values")
            print(f"  Range: [{out_weights.min():.4f}, {out_weights.max():.4f}]")
            print(f"  Mean: {out_weights.mean():.4f}, Std: {out_weights.std():.4f}")

if __name__ == "__main__":
    main()
