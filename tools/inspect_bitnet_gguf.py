#!/usr/bin/env python3
"""Inspect native BitNet GGUF structure."""

import struct
import sys

def read_gguf_header(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        print(f"Magic: {magic}")

        if magic != b'GGUF':
            print("Not a GGUF file!")
            return

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        print(f"Version: {version}")
        print(f"Tensor count: {tensor_count}")
        print(f"Metadata count: {metadata_count}")

        # Read metadata
        print("\n=== METADATA ===")
        metadata = {}
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            metadata[key] = value

            # Print interesting metadata
            if any(x in key for x in ['arch', 'vocab', 'embed', 'hidden', 'layer', 'head', 'context', 'quant']):
                print(f"  {key}: {value}")

        # Read tensor info
        print("\n=== TENSOR TYPES ===")
        type_counts = {}
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

            type_counts[ttype] = type_counts.get(ttype, 0) + 1

        # Print type distribution
        type_names = {
            0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1',
            6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
            10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
            14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS',
            18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
            22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16',
            26: 'I32', 27: 'I64', 28: 'F64', 29: 'BF16',
            30: 'Q4_0_4_4', 31: 'Q4_0_4_8', 32: 'Q4_0_8_8',
            33: 'TQ1_0', 34: 'TQ2_0'
        }

        for ttype, count in sorted(type_counts.items()):
            type_name = type_names.get(ttype, f'TYPE_{ttype}')
            print(f"  {type_name} (type {ttype}): {count} tensors")

        # Show first few tensors of each type
        print("\n=== SAMPLE TENSORS ===")
        shown_types = set()
        for t in tensors[:50]:
            if t['type'] not in shown_types:
                print(f"  {t['name']}: dims={t['dims']}, type={t['type']} ({type_names.get(t['type'], '?')})")
                shown_types.add(t['type'])

        # Get data start
        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        f.read(padding)
        data_start = f.tell()

        print(f"\nData starts at: {data_start}")

        # Read first tensor bytes
        print("\n=== FIRST TENSOR RAW BYTES ===")
        first_tensor = tensors[0]
        f.seek(data_start + first_tensor['offset'])
        raw = f.read(64)
        print(f"  Tensor: {first_tensor['name']}")
        print(f"  Type: {first_tensor['type']} ({type_names.get(first_tensor['type'], '?')})")
        print(f"  First 64 bytes: {raw.hex()}")

def read_value(f, vtype):
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

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/Xilinx/models/bitnet/ggml-model-i2_s.gguf"
    read_gguf_header(path)
