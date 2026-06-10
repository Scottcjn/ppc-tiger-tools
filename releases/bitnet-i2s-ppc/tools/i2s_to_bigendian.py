#!/usr/bin/env python3
"""
Convert Microsoft BitNet I2_S (type 36) GGUF from little-endian to big-endian.
For running native BitNet models on PowerPC G4/G5/POWER8.

I2_S Format (discovered through reverse engineering):
  - 64 bytes packed ternary data per 256 weights (no per-block scale!)
  - +32 bytes per tensor header (likely contains tensor-level scale/metadata)

Native ternary weights {-1, 0, +1} are directly usable without scaling!

Usage:
  python3 i2s_to_bigendian.py input.gguf output-BE.gguf
"""

import struct
import sys
import os
from typing import BinaryIO, Dict, List, Any, Tuple
import numpy as np

# I2_S format constants
I2S_TYPE = 36
I2S_BLOCK_WEIGHTS = 256   # Weights per block
I2S_BLOCK_BYTES = 64      # Just packed data, no per-block scale!
I2S_TENSOR_HEADER = 32    # Per-tensor header/metadata

# Type sizes
TYPE_SIZES = {
    0: 4,   # F32
    1: 2,   # F16
    24: 1,  # I8
    25: 2,  # I16
    26: 4,  # I32
    27: 8,  # I64
    28: 8,  # F64
    29: 2,  # BF16
}


def swap16(data: bytes) -> bytes:
    """Swap bytes in 16-bit value."""
    return bytes([data[1], data[0]])


def swap32(data: bytes) -> bytes:
    """Swap bytes in 32-bit value."""
    return bytes([data[3], data[2], data[1], data[0]])


def swap64(data: bytes) -> bytes:
    """Swap bytes in 64-bit value."""
    return bytes([data[7], data[6], data[5], data[4],
                  data[3], data[2], data[1], data[0]])


def write_swapped(out: BinaryIO, data: bytes, size: int):
    """Write data with byte swapping for big-endian."""
    if size == 2:
        out.write(swap16(data))
    elif size == 4:
        out.write(swap32(data))
    elif size == 8:
        out.write(swap64(data))
    else:
        out.write(data)


def read_and_swap_value(f: BinaryIO, out: BinaryIO, vtype: int) -> Any:
    """Read a GGUF value and write byte-swapped version."""
    if vtype == 0:  # uint8
        data = f.read(1)
        out.write(data)
        return struct.unpack('B', data)[0]
    elif vtype == 1:  # int8
        data = f.read(1)
        out.write(data)
        return struct.unpack('b', data)[0]
    elif vtype == 2:  # uint16
        data = f.read(2)
        write_swapped(out, data, 2)
        return struct.unpack('<H', data)[0]
    elif vtype == 3:  # int16
        data = f.read(2)
        write_swapped(out, data, 2)
        return struct.unpack('<h', data)[0]
    elif vtype == 4:  # uint32
        data = f.read(4)
        write_swapped(out, data, 4)
        return struct.unpack('<I', data)[0]
    elif vtype == 5:  # int32
        data = f.read(4)
        write_swapped(out, data, 4)
        return struct.unpack('<i', data)[0]
    elif vtype == 6:  # float32
        data = f.read(4)
        write_swapped(out, data, 4)
        return struct.unpack('<f', data)[0]
    elif vtype == 7:  # bool
        data = f.read(1)
        out.write(data)
        return struct.unpack('B', data)[0] != 0
    elif vtype == 8:  # string
        len_data = f.read(8)
        write_swapped(out, len_data, 8)
        length = struct.unpack('<Q', len_data)[0]
        str_data = f.read(length)
        out.write(str_data)
        return str_data.decode('utf-8')
    elif vtype == 9:  # array
        type_data = f.read(4)
        write_swapped(out, type_data, 4)
        arr_type = struct.unpack('<I', type_data)[0]

        len_data = f.read(8)
        write_swapped(out, len_data, 8)
        arr_len = struct.unpack('<Q', len_data)[0]

        result = []
        for _ in range(arr_len):
            result.append(read_and_swap_value(f, out, arr_type))
        return result
    elif vtype == 10:  # uint64
        data = f.read(8)
        write_swapped(out, data, 8)
        return struct.unpack('<Q', data)[0]
    elif vtype == 11:  # int64
        data = f.read(8)
        write_swapped(out, data, 8)
        return struct.unpack('<q', data)[0]
    elif vtype == 12:  # float64
        data = f.read(8)
        write_swapped(out, data, 8)
        return struct.unpack('<d', data)[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def compute_tensor_size(n_elements: int, ttype: int) -> int:
    """Compute the byte size of a tensor based on type."""
    if ttype == I2S_TYPE:
        n_blocks = n_elements // I2S_BLOCK_WEIGHTS
        return n_blocks * I2S_BLOCK_BYTES + I2S_TENSOR_HEADER
    elif ttype in TYPE_SIZES:
        return n_elements * TYPE_SIZES[ttype]
    else:
        return 0  # Unknown


def convert_i2s_tensor(f: BinaryIO, out: BinaryIO, n_elements: int, tensor_size: int):
    """
    Convert I2_S tensor data.

    I2_S format has:
    - 32 bytes tensor header (may contain scale, needs swapping)
    - 64 bytes per 256 weights (packed ternary, no swap needed for bit data)
    """
    # Read and analyze the tensor header (32 bytes)
    header = bytearray(f.read(I2S_TENSOR_HEADER))

    # Try to identify scale factors in header - swap any 16/32 bit values
    # The first few bytes might be FP16/FP32 scale factors
    # For safety, swap 16-bit aligned pairs (FP16 scale factors)
    for i in range(0, min(16, len(header)), 2):  # First 16 bytes as FP16s
        if i + 1 < len(header):
            header[i], header[i+1] = header[i+1], header[i]

    out.write(bytes(header))

    # Calculate number of data blocks
    n_blocks = n_elements // I2S_BLOCK_WEIGHTS
    data_bytes = n_blocks * I2S_BLOCK_BYTES

    # The packed ternary data doesn't need byte swapping
    # (individual bits are stored LSB-first in each byte, same on both endians)
    packed_data = f.read(data_bytes)
    out.write(packed_data)


def convert_other_tensor(f: BinaryIO, out: BinaryIO, n_elements: int, ttype: int):
    """Convert other tensor types (F32, F16, etc.) using numpy for bulk byte swapping."""
    if ttype not in TYPE_SIZES:
        print(f"    Warning: Unknown tensor type {ttype}")
        return

    elem_size = TYPE_SIZES[ttype]

    if elem_size == 1:
        # Single byte - no swapping needed
        out.write(f.read(n_elements))
        return

    # Use numpy for efficient bulk byte swapping
    total_bytes = n_elements * elem_size
    raw = f.read(total_bytes)

    if len(raw) < total_bytes:
        print(f"    Warning: Short read ({len(raw)} < {total_bytes})")
        # Pad to even size if needed
        if len(raw) % elem_size != 0:
            raw = raw + b'\x00' * (elem_size - len(raw) % elem_size)

    # Map to numpy dtype for byte swapping
    if elem_size == 2:
        arr = np.frombuffer(raw, dtype='<u2')  # Little-endian uint16
        arr = arr.byteswap()  # Swap to big-endian
    elif elem_size == 4:
        arr = np.frombuffer(raw, dtype='<u4')  # Little-endian uint32
        arr = arr.byteswap()
    elif elem_size == 8:
        arr = np.frombuffer(raw, dtype='<u8')  # Little-endian uint64
        arr = arr.byteswap()
    else:
        raise ValueError(f"Unsupported element size: {elem_size}")

    out.write(arr.tobytes())


def convert_gguf_bigendian(input_path: str, output_path: str):
    """Convert GGUF file from little-endian to big-endian."""
    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")

    with open(input_path, 'rb') as f, open(output_path, 'wb') as out:
        # Read and swap GGUF header
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid magic: {magic}")
        out.write(magic)

        # Version (uint32)
        version_data = f.read(4)
        write_swapped(out, version_data, 4)
        version = struct.unpack('<I', version_data)[0]
        print(f"  GGUF version: {version}")

        # Tensor count (uint64)
        tc_data = f.read(8)
        write_swapped(out, tc_data, 8)
        tensor_count = struct.unpack('<Q', tc_data)[0]
        print(f"  Tensor count: {tensor_count}")

        # Metadata count (uint64)
        mc_data = f.read(8)
        write_swapped(out, mc_data, 8)
        metadata_count = struct.unpack('<Q', mc_data)[0]
        print(f"  Metadata count: {metadata_count}")

        # Process metadata
        print(f"\n  Processing {metadata_count} metadata entries...")
        for i in range(metadata_count):
            kl_data = f.read(8)
            write_swapped(out, kl_data, 8)
            key_len = struct.unpack('<Q', kl_data)[0]

            key = f.read(key_len)
            out.write(key)

            vt_data = f.read(4)
            write_swapped(out, vt_data, 4)
            vtype = struct.unpack('<I', vt_data)[0]

            value = read_and_swap_value(f, out, vtype)

            if 'name' in key.decode('utf-8') or 'arch' in key.decode('utf-8'):
                print(f"    {key.decode('utf-8')}: {value}")

        # Read tensor info
        print(f"\n  Reading {tensor_count} tensor infos...")
        tensors = []

        for i in range(tensor_count):
            nl_data = f.read(8)
            write_swapped(out, nl_data, 8)
            name_len = struct.unpack('<Q', nl_data)[0]

            name = f.read(name_len)
            out.write(name)

            nd_data = f.read(4)
            write_swapped(out, nd_data, 4)
            n_dims = struct.unpack('<I', nd_data)[0]

            n_elements = 1
            for _ in range(n_dims):
                dim_data = f.read(8)
                write_swapped(out, dim_data, 8)
                dim = struct.unpack('<Q', dim_data)[0]
                n_elements *= dim

            type_data = f.read(4)
            write_swapped(out, type_data, 4)
            ttype = struct.unpack('<I', type_data)[0]

            off_data = f.read(8)
            write_swapped(out, off_data, 8)
            offset = struct.unpack('<Q', off_data)[0]

            tensors.append({
                'name': name.decode('utf-8'),
                'n_elements': n_elements,
                'type': ttype,
                'offset': offset
            })

        # Alignment padding
        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        pad_data = f.read(padding)
        out.write(pad_data)

        data_start = f.tell()
        print(f"\n  Data starts at offset: {data_start:,}")

        # Sort tensors by offset
        tensors_sorted = sorted(tensors, key=lambda t: t['offset'])

        # Compute actual sizes from offset differences
        for i, t in enumerate(tensors_sorted):
            if i + 1 < len(tensors_sorted):
                t['actual_size'] = tensors_sorted[i+1]['offset'] - t['offset']
            else:
                t['actual_size'] = None  # Last tensor - read to EOF

        # Count tensor types
        type_counts = {}
        for t in tensors:
            type_counts[t['type']] = type_counts.get(t['type'], 0) + 1

        print(f"\n  Tensor type distribution:")
        type_names = {0: 'F32', 1: 'F16', 36: 'I2_S (BitNet)'}
        for ttype, count in sorted(type_counts.items()):
            print(f"    Type {ttype} ({type_names.get(ttype, 'unknown')}): {count} tensors")

        # Process tensor data
        print(f"\n  Converting tensor data...")
        i2s_count = 0
        other_count = 0

        for i, t in enumerate(tensors_sorted):
            expected_pos = data_start + t['offset']

            # Handle any alignment/padding
            current_out_pos = out.tell()
            if current_out_pos < expected_pos:
                out.write(b'\x00' * (expected_pos - current_out_pos))

            f.seek(expected_pos)

            if t['type'] == I2S_TYPE:
                if i2s_count < 3:
                    actual = t.get('actual_size', 'unknown')
                    print(f"    [{i+1}/{len(tensors)}] I2_S: {t['name'][:40]}... ({t['n_elements']:,} el, {actual:,} bytes)")
                elif i2s_count == 3:
                    print(f"    ... processing remaining {type_counts[I2S_TYPE] - 3} I2_S tensors ...")

                if t['actual_size'] is not None:
                    convert_i2s_tensor(f, out, t['n_elements'], t['actual_size'])
                else:
                    # Last tensor - read remaining
                    remaining = f.read()
                    # Swap header portion
                    header = bytearray(remaining[:I2S_TENSOR_HEADER])
                    for j in range(0, min(16, len(header)), 2):
                        if j + 1 < len(header):
                            header[j], header[j+1] = header[j+1], header[j]
                    out.write(bytes(header))
                    out.write(remaining[I2S_TENSOR_HEADER:])

                i2s_count += 1
            else:
                if other_count < 3:
                    print(f"    [{i+1}/{len(tensors)}] Type {t['type']}: {t['name'][:40]}... ({t['n_elements']:,} elements)")
                elif other_count == 3:
                    print(f"    ... processing remaining {len(tensors) - type_counts[I2S_TYPE] - 3} non-I2_S tensors ...")

                convert_other_tensor(f, out, t['n_elements'], t['type'])
                other_count += 1

        print(f"\n  Conversion complete!")
        print(f"    I2_S tensors: {i2s_count}")
        print(f"    Other tensors: {other_count}")

        out.flush()
        out_size = out.tell()

    in_size = os.path.getsize(input_path)
    print(f"\n  Input size:  {in_size:,} bytes ({in_size/1024/1024:.1f} MB)")
    print(f"  Output size: {out_size:,} bytes ({out_size/1024/1024:.1f} MB)")

    if in_size == out_size:
        print(f"  Size match: OK")
    else:
        diff = out_size - in_size
        print(f"  Size difference: {diff:,} bytes ({diff/1024:.1f} KB)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 i2s_to_bigendian.py input.gguf [output.gguf]")
        print("\nConverts Microsoft BitNet I2_S GGUF from little-endian to big-endian")
        print("for PowerPC G4/G5/POWER8 systems.")
        print("\nI2_S Format (Type 36):")
        print("  - Native ternary weights {-1, 0, +1}")
        print("  - 2 bits per weight, 4 weights per byte")
        print("  - 64 bytes per 256 weights (pure data)")
        print("  - 32 bytes per-tensor header")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        base = input_path.rsplit('.', 1)[0]
        output_path = f"{base}-BE.gguf"

    convert_gguf_bigendian(input_path, output_path)


if __name__ == "__main__":
    main()
