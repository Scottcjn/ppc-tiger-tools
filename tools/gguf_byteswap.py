#!/usr/bin/env python3
"""
GGUF Byte Swapper - Convert little-endian GGUF to big-endian for PowerPC

GGUF files are stored in little-endian format. PowerPC G4/G5 Macs are big-endian.
This tool converts GGUF model files to big-endian byte order.

Usage:
    python3 gguf_byteswap.py input.gguf output_be.gguf

Author: ppc-tiger-tools project
License: MIT
December 2025
"""

import struct
import sys
import os
from pathlib import Path

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_MAGIC_BE = 0x47475546  # "GGUF" in big-endian

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types that need byte swapping
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
GGML_TYPE_I8 = 16
GGML_TYPE_I16 = 17
GGML_TYPE_I32 = 18


class GGUFByteSwapper:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.tensors = []
        self.metadata = {}

    def swap16(self, data):
        """Swap bytes in 16-bit values"""
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            result[i] = data[i+1]
            result[i+1] = data[i]
        return bytes(result)

    def swap32(self, data):
        """Swap bytes in 32-bit values"""
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            result[i] = data[i+3]
            result[i+1] = data[i+2]
            result[i+2] = data[i+1]
            result[i+3] = data[i]
        return bytes(result)

    def swap64(self, data):
        """Swap bytes in 64-bit values"""
        result = bytearray(len(data))
        for i in range(0, len(data), 8):
            for j in range(8):
                result[i+j] = data[i+7-j]
        return bytes(result)

    def read_le_u32(self, f):
        """Read little-endian uint32"""
        return struct.unpack('<I', f.read(4))[0]

    def read_le_u64(self, f):
        """Read little-endian uint64"""
        return struct.unpack('<Q', f.read(8))[0]

    def read_le_i32(self, f):
        """Read little-endian int32"""
        return struct.unpack('<i', f.read(4))[0]

    def read_le_f32(self, f):
        """Read little-endian float32"""
        return struct.unpack('<f', f.read(4))[0]

    def read_string(self, f):
        """Read GGUF string (length-prefixed)"""
        length = self.read_le_u64(f)
        return f.read(length).decode('utf-8')

    def write_be_u32(self, f, value):
        """Write big-endian uint32"""
        f.write(struct.pack('>I', value))

    def write_be_u64(self, f, value):
        """Write big-endian uint64"""
        f.write(struct.pack('>Q', value))

    def write_be_i32(self, f, value):
        """Write big-endian int32"""
        f.write(struct.pack('>i', value))

    def write_be_f32(self, f, value):
        """Write big-endian float32"""
        f.write(struct.pack('>f', value))

    def write_string_be(self, f, s):
        """Write GGUF string in big-endian format"""
        data = s.encode('utf-8')
        self.write_be_u64(f, len(data))
        f.write(data)

    def read_value(self, f, vtype):
        """Read a GGUF value based on type"""
        if vtype == GGUF_TYPE_UINT8:
            return struct.unpack('B', f.read(1))[0]
        elif vtype == GGUF_TYPE_INT8:
            return struct.unpack('b', f.read(1))[0]
        elif vtype == GGUF_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif vtype == GGUF_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif vtype == GGUF_TYPE_UINT32:
            return self.read_le_u32(f)
        elif vtype == GGUF_TYPE_INT32:
            return self.read_le_i32(f)
        elif vtype == GGUF_TYPE_FLOAT32:
            return self.read_le_f32(f)
        elif vtype == GGUF_TYPE_BOOL:
            return struct.unpack('?', f.read(1))[0]
        elif vtype == GGUF_TYPE_STRING:
            return self.read_string(f)
        elif vtype == GGUF_TYPE_UINT64:
            return self.read_le_u64(f)
        elif vtype == GGUF_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif vtype == GGUF_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif vtype == GGUF_TYPE_ARRAY:
            arr_type = self.read_le_u32(f)
            arr_len = self.read_le_u64(f)
            return (arr_type, [self.read_value(f, arr_type) for _ in range(arr_len)])
        else:
            raise ValueError(f"Unknown GGUF type: {vtype}")

    def write_value_be(self, f, vtype, value):
        """Write a GGUF value in big-endian format"""
        if vtype == GGUF_TYPE_UINT8:
            f.write(struct.pack('B', value))
        elif vtype == GGUF_TYPE_INT8:
            f.write(struct.pack('b', value))
        elif vtype == GGUF_TYPE_UINT16:
            f.write(struct.pack('>H', value))
        elif vtype == GGUF_TYPE_INT16:
            f.write(struct.pack('>h', value))
        elif vtype == GGUF_TYPE_UINT32:
            self.write_be_u32(f, value)
        elif vtype == GGUF_TYPE_INT32:
            self.write_be_i32(f, value)
        elif vtype == GGUF_TYPE_FLOAT32:
            self.write_be_f32(f, value)
        elif vtype == GGUF_TYPE_BOOL:
            f.write(struct.pack('?', value))
        elif vtype == GGUF_TYPE_STRING:
            self.write_string_be(f, value)
        elif vtype == GGUF_TYPE_UINT64:
            self.write_be_u64(f, value)
        elif vtype == GGUF_TYPE_INT64:
            f.write(struct.pack('>q', value))
        elif vtype == GGUF_TYPE_FLOAT64:
            f.write(struct.pack('>d', value))
        elif vtype == GGUF_TYPE_ARRAY:
            arr_type, arr_values = value
            self.write_be_u32(f, arr_type)
            self.write_be_u64(f, len(arr_values))
            for v in arr_values:
                self.write_value_be(f, arr_type, v)

    def swap_tensor_data(self, data, tensor_type):
        """Swap bytes in tensor data based on tensor type"""
        if tensor_type == GGML_TYPE_F32:
            return self.swap32(data)
        elif tensor_type == GGML_TYPE_F16:
            return self.swap16(data)
        elif tensor_type in (GGML_TYPE_I16,):
            return self.swap16(data)
        elif tensor_type in (GGML_TYPE_I32,):
            return self.swap32(data)
        elif tensor_type in (GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0,
                            GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1):
            # Quantized formats have block structures with mixed types
            # Q4_0: 2-byte scale (f16) + 16 bytes of 4-bit weights per block
            # Need to swap the scale bytes
            return self.swap_quantized_blocks(data, tensor_type)
        elif tensor_type in (GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
                            GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K):
            # K-quant formats have more complex structures
            return self.swap_kquant_blocks(data, tensor_type)
        else:
            # Unknown or byte-aligned types - return as-is
            return data

    def swap_quantized_blocks(self, data, tensor_type):
        """Swap bytes in standard quantized blocks"""
        result = bytearray(data)

        if tensor_type == GGML_TYPE_Q4_0:
            # Block size: 2 (f16 scale) + 16 (weights) = 18 bytes
            block_size = 18
            for i in range(0, len(data), block_size):
                # Swap f16 scale (2 bytes)
                result[i], result[i+1] = result[i+1], result[i]

        elif tensor_type == GGML_TYPE_Q4_1:
            # Block size: 2 (f16 scale) + 2 (f16 min) + 16 (weights) = 20 bytes
            block_size = 20
            for i in range(0, len(data), block_size):
                result[i], result[i+1] = result[i+1], result[i]
                result[i+2], result[i+3] = result[i+3], result[i+2]

        elif tensor_type == GGML_TYPE_Q8_0:
            # Block size: 2 (f16 scale) + 32 (weights) = 34 bytes
            block_size = 34
            for i in range(0, len(data), block_size):
                result[i], result[i+1] = result[i+1], result[i]

        return bytes(result)

    def swap_kquant_blocks(self, data, tensor_type):
        """Swap bytes in K-quant blocks.

        K-quant blocks have f16 scale values that need byte swapping.
        The quantized values themselves are byte arrays and don't need swapping.
        """
        result = bytearray(data)

        if tensor_type == GGML_TYPE_Q2_K:
            # Q2_K: 16(scales) + 16(qs) + 4(d+dmin as f16) = 84 bytes per 256 elements
            # Actually: scales[16] + qs[64] + d(f16) + dmin(f16) = 84 bytes
            block_size = 84
            for i in range(0, len(data), block_size):
                if i + block_size <= len(data):
                    # d at offset 80, dmin at offset 82
                    result[i+80], result[i+81] = result[i+81], result[i+80]  # d
                    result[i+82], result[i+83] = result[i+83], result[i+82]  # dmin

        elif tensor_type == GGML_TYPE_Q3_K:
            # Q3_K: hmask[32] + qs[64] + scales[12] + d(f16) = 110 bytes
            block_size = 110
            for i in range(0, len(data), block_size):
                if i + block_size <= len(data):
                    # d at offset 108
                    result[i+108], result[i+109] = result[i+109], result[i+108]

        elif tensor_type == GGML_TYPE_Q4_K:
            # Q4_K: d(f16) + dmin(f16) + scales[12] + qs[128] = 144 bytes
            block_size = 144
            for i in range(0, len(data), block_size):
                if i + 4 <= len(data):
                    # d at offset 0, dmin at offset 2
                    result[i], result[i+1] = result[i+1], result[i]      # d
                    result[i+2], result[i+3] = result[i+3], result[i+2]  # dmin

        elif tensor_type == GGML_TYPE_Q5_K:
            # Q5_K: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] = 176 bytes
            block_size = 176
            for i in range(0, len(data), block_size):
                if i + 4 <= len(data):
                    # d at offset 0, dmin at offset 2
                    result[i], result[i+1] = result[i+1], result[i]      # d
                    result[i+2], result[i+3] = result[i+3], result[i+2]  # dmin

        elif tensor_type == GGML_TYPE_Q6_K:
            # Q6_K: ql[128] + qh[64] + scales[16] + d(f16) = 210 bytes
            block_size = 210
            for i in range(0, len(data), block_size):
                if i + block_size <= len(data):
                    # d at offset 208
                    result[i+208], result[i+209] = result[i+209], result[i+208]

        elif tensor_type == GGML_TYPE_Q8_K:
            # Q8_K: d(f32) + qs[256] + bsums[16*2] = 292 bytes
            # d is f32, not f16!
            block_size = 292
            for i in range(0, len(data), block_size):
                if i + 4 <= len(data):
                    # d at offset 0 (f32 - 4 byte swap)
                    result[i], result[i+1], result[i+2], result[i+3] = \
                        result[i+3], result[i+2], result[i+1], result[i]
                    # bsums are int16, need to swap each pair at offset 260-291
                    for j in range(16):
                        bsum_off = i + 260 + j*2
                        if bsum_off + 2 <= len(data):
                            result[bsum_off], result[bsum_off+1] = \
                                result[bsum_off+1], result[bsum_off]

        return bytes(result)

    def convert(self):
        """Convert GGUF file from little-endian to big-endian"""
        file_size = self.input_path.stat().st_size
        print(f"Input file: {self.input_path}")
        print(f"File size: {file_size / (1024*1024*1024):.2f} GB")
        print(f"Output file: {self.output_path}")
        print()

        with open(self.input_path, 'rb') as fin, open(self.output_path, 'wb') as fout:
            # Read and verify magic (4 character bytes, NOT an integer)
            magic = fin.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Invalid GGUF magic: {magic}")

            # Write magic as-is (it's a string, not an integer to swap)
            fout.write(b'GGUF')
            print("✓ Magic number (GGUF)")

            # Read and write version
            version = self.read_le_u32(fin)
            self.write_be_u32(fout, version)
            print(f"✓ Version: {version}")

            # Read tensor count and metadata count
            tensor_count = self.read_le_u64(fin)
            metadata_count = self.read_le_u64(fin)
            self.write_be_u64(fout, tensor_count)
            self.write_be_u64(fout, metadata_count)
            print(f"✓ Tensors: {tensor_count}")
            print(f"✓ Metadata entries: {metadata_count}")

            # Read and write metadata
            print("\nConverting metadata...")
            for i in range(metadata_count):
                key = self.read_string(fin)
                vtype = self.read_le_u32(fin)
                value = self.read_value(fin, vtype)

                self.write_string_be(fout, key)
                self.write_be_u32(fout, vtype)
                self.write_value_be(fout, vtype, value)

                if i % 10 == 0:
                    print(f"  {i}/{metadata_count} metadata entries...", end='\r')

            print(f"  {metadata_count}/{metadata_count} metadata entries done")

            # Read tensor info
            print("\nReading tensor info...")
            tensor_infos = []
            for i in range(tensor_count):
                name = self.read_string(fin)
                n_dims = self.read_le_u32(fin)
                dims = [self.read_le_u64(fin) for _ in range(n_dims)]
                ttype = self.read_le_u32(fin)
                offset = self.read_le_u64(fin)

                tensor_infos.append({
                    'name': name,
                    'n_dims': n_dims,
                    'dims': dims,
                    'type': ttype,
                    'offset': offset
                })

                if i % 50 == 0:
                    print(f"  {i}/{tensor_count} tensor infos...", end='\r')

            print(f"  {tensor_count}/{tensor_count} tensor infos read")

            # Write tensor info in big-endian, track offset positions for patching
            print("\nWriting tensor info...")
            offset_positions = []  # File positions where we wrote each tensor's offset
            for i, info in enumerate(tensor_infos):
                self.write_string_be(fout, info['name'])
                self.write_be_u32(fout, info['n_dims'])
                for d in info['dims']:
                    self.write_be_u64(fout, d)
                self.write_be_u32(fout, info['type'])
                # Record position where offset is written, then write placeholder
                offset_positions.append(fout.tell())
                self.write_be_u64(fout, 0)  # Placeholder - will patch later

            # Align to 32 bytes
            current_pos = fout.tell()
            padding = (32 - (current_pos % 32)) % 32
            fout.write(b'\x00' * padding)

            # Record where tensor data starts in OUTPUT file
            tensor_data_start_out = fout.tell()

            # Read alignment padding in input and record where tensor data starts
            fin_pos = fin.tell()
            fin_padding = (32 - (fin_pos % 32)) % 32
            fin.read(fin_padding)
            tensor_data_start_in = fin.tell()  # THIS is where tensors begin in input

            # Calculate tensor sizes and convert tensor data
            print("\nConverting tensor data...")
            print(f"  Input tensor data starts at: {tensor_data_start_in}")
            print(f"  Output tensor data starts at: {tensor_data_start_out}")

            # Get type sizes for calculating tensor data sizes
            type_sizes = {
                GGML_TYPE_F32: 4,
                GGML_TYPE_F16: 2,
                GGML_TYPE_Q4_0: 18,  # per 32 elements
                GGML_TYPE_Q4_1: 20,
                GGML_TYPE_Q5_0: 22,
                GGML_TYPE_Q5_1: 24,
                GGML_TYPE_Q8_0: 34,
                GGML_TYPE_Q8_1: 36,
                GGML_TYPE_Q2_K: 84,   # per 256 elements
                GGML_TYPE_Q3_K: 110,
                GGML_TYPE_Q4_K: 144,
                GGML_TYPE_Q5_K: 176,
                GGML_TYPE_Q6_K: 210,
                GGML_TYPE_Q8_K: 292,
                GGML_TYPE_I8: 1,
                GGML_TYPE_I16: 2,
                GGML_TYPE_I32: 4,
            }

            block_sizes = {
                GGML_TYPE_Q4_0: 32,
                GGML_TYPE_Q4_1: 32,
                GGML_TYPE_Q5_0: 32,
                GGML_TYPE_Q5_1: 32,
                GGML_TYPE_Q8_0: 32,
                GGML_TYPE_Q8_1: 32,
                GGML_TYPE_Q2_K: 256,
                GGML_TYPE_Q3_K: 256,
                GGML_TYPE_Q4_K: 256,
                GGML_TYPE_Q5_K: 256,
                GGML_TYPE_Q6_K: 256,
                GGML_TYPE_Q8_K: 256,
            }

            # First pass: calculate tensor sizes
            tensor_sizes = []
            for i, info in enumerate(tensor_infos):
                n_elements = 1
                for d in info['dims']:
                    n_elements *= d

                ttype = info['type']
                if ttype in block_sizes:
                    n_blocks = (n_elements + block_sizes[ttype] - 1) // block_sizes[ttype]
                    tensor_size = n_blocks * type_sizes.get(ttype, 1)
                elif ttype in type_sizes:
                    tensor_size = n_elements * type_sizes[ttype]
                else:
                    tensor_size = n_elements  # Assume 1 byte
                tensor_sizes.append(tensor_size)

            # Sort tensors by their input offset to read in file order
            sorted_indices = sorted(range(len(tensor_infos)),
                                   key=lambda i: tensor_infos[i]['offset'])

            # Track new offsets for output file
            new_offsets = [0] * len(tensor_infos)
            current_out_offset = 0

            for sort_idx, i in enumerate(sorted_indices):
                info = tensor_infos[i]
                tensor_size = tensor_sizes[i]

                # Seek to tensor position in input file using its offset
                input_pos = tensor_data_start_in + info['offset']
                fin.seek(input_pos)

                # Read tensor data
                tensor_data = fin.read(tensor_size)
                if len(tensor_data) != tensor_size:
                    print(f"\n  WARNING: Tensor {info['name']} expected {tensor_size} bytes, got {len(tensor_data)}")

                # Swap bytes
                swapped_data = self.swap_tensor_data(tensor_data, info['type'])

                # Record new offset (relative to output data section start)
                new_offsets[i] = current_out_offset

                # Write swapped data
                fout.write(swapped_data)
                current_out_offset += len(swapped_data)

                if sort_idx % 10 == 0:
                    progress = (sort_idx / tensor_count) * 100
                    print(f"  {sort_idx}/{tensor_count} tensors ({progress:.1f}%)...", end='\r')

            print(f"  {tensor_count}/{tensor_count} tensors converted!")

            # Now go back and patch the tensor offsets in the output file
            print("\nPatching tensor offsets...")
            end_pos = fout.tell()
            for i, new_offset in enumerate(new_offsets):
                fout.seek(offset_positions[i])
                self.write_be_u64(fout, new_offset)
            fout.seek(end_pos)
            print(f"  Patched {len(new_offsets)} tensor offsets")

        output_size = self.output_path.stat().st_size
        print(f"\n✓ Conversion complete!")
        print(f"  Output size: {output_size / (1024*1024*1024):.2f} GB")

        return True


def main():
    if len(sys.argv) != 3:
        print("GGUF Byte Swapper - Convert little-endian GGUF to big-endian")
        print()
        print("Usage: python3 gguf_byteswap.py <input.gguf> <output_be.gguf>")
        print()
        print("This tool converts GGUF model files from little-endian (x86/ARM)")
        print("to big-endian format for PowerPC Macs.")
        print()
        print("Example:")
        print("  python3 gguf_byteswap.py tinyllama-1.1b-q4.gguf tinyllama-1.1b-q4-be.gguf")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if os.path.exists(output_path):
        response = input(f"Output file {output_path} exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)

    try:
        swapper = GGUFByteSwapper(input_path, output_path)
        swapper.convert()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
