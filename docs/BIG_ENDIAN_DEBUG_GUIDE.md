# Big-Endian PowerPC Debugging Guide

## The Rosetta Stone: Understanding llama.cpp on Big-Endian PowerPC

This document captures the debugging journey of getting llama.cpp working on big-endian PowerPC (G4/G5 Macs), using the working little-endian POWER8 as a reference.

## System Comparison

| System | Endianness | Status | Why |
|--------|------------|--------|-----|
| POWER8 (Ubuntu 20.04) | Little-endian | **WORKS** | GGUF files are LE, loads directly |
| G4/G5 (Mac OS X Tiger) | Big-endian | **NEEDS CONVERSION** | Must convert GGUF to BE format |
| x86/ARM | Little-endian | Works | Native GGUF format |

## The Problem

GGUF model files store ALL data in little-endian byte order:
- Magic number, version, counts
- Metadata values (floats, ints, strings)
- Tensor info (dimensions, types, offsets)
- Tensor data (weights - F32, F16, quantized)

When big-endian PowerPC reads these files directly:
- Magic "GGUF" becomes "FUGG" (bytes reversed)
- Version 3 becomes 50331648
- All float weights are garbage (bytes reversed)
- Model crashes or produces nonsense

## Two-Part Solution

### Part 1: File Conversion (gguf_byteswap.py)

Convert GGUF files to big-endian ONCE before use:

```bash
python3 gguf_byteswap.py input.gguf output-BE.gguf
```

What gets converted:
- **Header**: Magic (unchanged), version, tensor_count, metadata_count
- **Metadata**: All scalar values (int16/32/64, float32/64), array values
- **Tensor Info**: Dimensions, types, offsets (patched for new positions)
- **Tensor Data**:
  - F32: Every 4 bytes swapped
  - F16: Every 2 bytes swapped
  - Q4_K: Only d (2 bytes) and dmin (2 bytes) swapped per 144-byte block
  - Q6_K: Only d (2 bytes at offset 208) swapped per 210-byte block
  - Q2_K/Q3_K/Q5_K/Q8_K: Similar selective swapping

### Part 2: Runtime Configuration

In `ggml.c`, set `GGUF_IS_BIG_ENDIAN = 0` when using pre-converted models:

```c
// Using pre-converted BE model - NO runtime swapping needed
#define GGUF_IS_BIG_ENDIAN 0
```

## Verification Steps

### 1. Verify Tensor Data Conversion

```python
# Check Q4_K block swapping
# d bytes at offset 0-1, dmin at 2-3 should be swapped

# LE file: 01 00 04 00 fa fc ff f7...
# BE file: 00 01 00 04 fa fc ff f7...
#          ^^^^^ ^^^^^  <-- swapped
#                       ^^^^^^^^^^^^  <-- not swapped (byte arrays)
```

### 2. Test On-Device Reading

Create a test program that reads the BE file on PowerPC:

```c
// test_q4k_read.c - compile on G4
#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

int main() {
    // Open BE model, read block, verify d/dmin values
    // PowerPC reads uint16_t as big-endian natively
}
```

### 3. Verify FP16 Conversion

G4 uses lookup table for FP16→FP32 (no F16C hardware):

```c
// ggml-impl.h selects this path for G4:
#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)

// Table built at init time with native endianness
// Index is raw uint16 bit pattern
// Works correctly with BE-converted model data
```

## What We Verified Works Correctly

| Component | Status | Evidence |
|-----------|--------|----------|
| File metadata parsing | ✓ | Model loads, shows correct tensor count (201) |
| F32 tensor byte swap | ✓ | Comparison shows matching float values |
| Q4_K d/dmin swap | ✓ | Manual comparison: bytes correctly swapped |
| Q6_K d swap | ✓ | Bytes at offset 208-209 correctly swapped |
| PowerPC reads BE data | ✓ | test_q4k_read.c shows correct uint16 values |
| FP16 lookup table | ✓ | Built at runtime with native endianness |
| Quantized scales (6-bit) | ✓ | Single-byte operations, endian-neutral |
| Quantized values (4-bit) | ✓ | Nibble extraction, endian-neutral |

## Known Issues / Open Questions

### 1. Negative Timing Values

```
llama_print_timings: prompt eval time = -390694.61 ms
```

Possible causes:
- `clock_gettime(CLOCK_MONOTONIC)` may not exist on Tiger
- Timer overflow due to 32-bit time_t

### 2. Garbage Output Despite Correct Data Loading

Model generates tokens but output is nonsense:
- "orirlava or then withuk once-login soululare"

Possible remaining issues to investigate:
- Matrix multiplication code assumptions
- Attention mechanism (RoPE)
- Softmax computation
- Type punning in math operations

### 3. nearest_int() Type Punning

```c
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
```

This IEEE 754 trick should work on BE, but worth verifying.

## Code Paths Specific to Architecture

### POWER8 (Little-Endian Ubuntu)

- Uses `__POWER9_VECTOR__` code paths when available
- Has VSX SIMD instructions
- FP16 conversion via inline assembly: `xscvhpdp`
- Reads standard LE GGUF files directly

### G4/G5 (Big-Endian Mac OS X)

- No `__POWER9_VECTOR__` (too old)
- Classic AltiVec available (G4) but needs explicit patches
- FP16 conversion via lookup table
- **Must use BE-converted GGUF files**
- **Must set GGUF_IS_BIG_ENDIAN=0** for pre-converted models

## Files Reference

| File | Purpose |
|------|---------|
| `tools/gguf_byteswap.py` | Convert GGUF files to big-endian |
| `tools/gguf_compare.py` | Compare LE and BE file contents |
| `patches/llama.cpp-b2000-bigendian/` | Runtime byte-swap patches |
| `patches/llama.cpp-altivec/` | AltiVec SIMD support |
| `docs/LLAMA_CPP_PPC_SDK.md` | Complete SDK reference |

## Debugging Commands

```bash
# SSH to G4
sshpass -p 'Elyanlabs12@' ssh -o HostKeyAlgorithms=+ssh-rsa sophia@192.168.0.125

# Check compiler defines
gcc-7 -dM -E - < /dev/null | grep -i "endian\|power\|ppc"

# Run inference test
./main -m ~/models/tinyllama-1.1b-q4-be.gguf -p "Hello" -n 10

# Check what defines are active
grep -n "POWER9\|BIG_ENDIAN\|__ppc" ggml.c ggml-impl.h
```

## Next Steps to Debug Garbage Output

1. **Add debug prints** in dequantization to verify runtime values
2. **Test F32-only model** to isolate quantization from core math
3. **Check matrix multiplication** for any endian assumptions
4. **Verify attention scores** are computed correctly
5. **Test RoPE (rotary position embedding)** computation

## The "Rosetta Stone" Insight

The original Apple Rosetta translated PPC→Intel by handling endianness at the instruction level transparently. For llama.cpp on BE PowerPC:

1. **File level**: Pre-convert GGUF to BE format
2. **Runtime level**: Ensure all multi-byte reads/writes use native endianness
3. **Math level**: Verify IEEE 754 operations work correctly

The conversion is correct. The remaining issue is somewhere in the compute path - the "Rosetta Stone" for this codebase needs to translate more than just the file format.

## December 2025
