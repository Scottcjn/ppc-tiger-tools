# AltiVec Support for llama.cpp

Patches to enable classic AltiVec SIMD on PowerPC G4/G5 Macs.

## The Problem

llama.cpp's GGML backend only supports POWER9 vector instructions (`__POWER9_VECTOR__`), not classic AltiVec that exists on G4/G5 PowerPC Macs.

Key differences:
- POWER9 uses `vec_xl`/`vec_xst` for unaligned loads (not in AltiVec)
- POWER9 uses `vec_extract` to read elements (not in AltiVec)
- AltiVec uses `vec_ld`/`vec_st` which require 16-byte alignment
- AltiVec lacks native F16 support

## Current Status

**ggml-altivec.patch** - Basic F32 vector operations for ggml.c:
- ✅ Vector loads/stores with alignment handling
- ✅ FMA (fused multiply-add)
- ✅ Vector reduction without vec_extract
- ⚠️ F16 falls back to scalar conversion

## Usage

```bash
cd llama.cpp
patch -p1 < /path/to/ggml-altivec.patch
```

Then compile with AltiVec:

```bash
make clean
CFLAGS="-maltivec -faltivec" make
```

## Expected Speedup

With AltiVec enabled, expect:
- 2-4x speedup on vector operations
- F32 matrix multiply benefits most
- Quantized operations still mostly scalar

## Limitations

- Requires 16-byte alignment for optimal performance
- No native F16 vectorization
- Many POWER9-specific quantization paths not ported
- ggml-quants.c needs more extensive porting

## Testing

The patch adds a runtime check:

```c
#if defined(__ALTIVEC__) && !defined(__POWER9_VECTOR__)
    printf("AltiVec enabled (G4/G5 mode)\n");
#endif
```

## December 2025
