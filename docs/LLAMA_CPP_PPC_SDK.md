# llama.cpp PowerPC SDK Documentation

Complete reference for running llama.cpp on PowerPC G4/G5 Macs.

## Quick Reference

| Item | Value |
|------|-------|
| Target | PowerPC G4/G5 (Mac OS X 10.4-10.5) |
| Compiler | GCC 7.5.0 (Tigerbrew) |
| llama.cpp Version | b2000 |
| Working Speed | 0.06-0.12 tok/s (G4), 0.16 tok/s (G5) |

## The Problem

GGUF files are **little-endian**. PowerPC is **big-endian**. Without byte-swapping:
- Magic "GGUF" reads as "FUGG"
- All integers/floats are byte-reversed
- Model crashes at `ggml_compute_forward_mul`

## Solution Overview

Two approaches:

1. **Pre-convert models** (recommended): Use `gguf_byteswap.py` to create big-endian GGUF files
2. **Runtime swapping**: Patch llama.cpp to swap bytes when loading

## File Locations

### On Development Machine (HP Victus)
```
/home/scott/ppc-tiger-tools/
├── tools/
│   ├── gguf_byteswap.py      # LE→BE model converter
│   ├── gguf_to_numpy.py      # Extract weights to NumPy
│   └── numpy_llm.py          # Pure Python inference
├── patches/
│   ├── llama.cpp-altivec/
│   │   ├── ggml-altivec-b2000.c   # AltiVec SIMD code
│   │   └── README.md
│   ├── llama.cpp-b2000/
│   │   └── leopard_compat.h       # 10.5 compatibility shims
│   └── llama.cpp-b2000-bigendian/
│       ├── ggml-bigendian-block.c # Byte-swap code
│       └── README.md
└── docs/
    └── LLAMA_CPP_PPC_SDK.md       # This file
```

### On G4 Mac (192.168.0.125)
```
~/llama.cpp-b2000/
├── ggml.c                    # Main file (patched)
├── ggml-quants.c             # Quantization (needs AltiVec patch)
├── llama.cpp
├── main                      # Built executable
└── *.o                       # Object files

~/models/
├── tinyllama-1.1b-q4.gguf    # Original LE model
└── tinyllama-1.1b-q4-BE.gguf # Converted BE model
```

## Build Commands

### SSH to G4
```bash
sshpass -p 'Elyanlabs12@' ssh -o StrictHostKeyChecking=no g4-mirror
# Or use host alias from ~/.ssh/config
```

### Full Build (Scalar - No SIMD)
```bash
cd ~/llama.cpp-b2000

CC=/usr/local/bin/gcc-7
CXX=/usr/local/bin/g++-7
CFLAGS="-DNDEBUG -O3 -mcpu=7450 -std=gnu11 -fno-strict-aliasing"
CXXFLAGS="-DNDEBUG -O3 -mcpu=7450 -std=gnu++11 -fno-strict-aliasing"
INCLUDES="-I. -I./common"

# C files
$CC $CFLAGS $INCLUDES -c ggml.c -o ggml.o
$CC $CFLAGS $INCLUDES -c ggml-alloc.c -o ggml-alloc.o
$CC $CFLAGS $INCLUDES -c ggml-backend.c -o ggml-backend.o
$CC $CFLAGS $INCLUDES -c ggml-quants.c -o ggml-quants.o

# C++ files
$CXX $CXXFLAGS $INCLUDES -c llama.cpp -o llama.o
$CXX $CXXFLAGS $INCLUDES -c common/common.cpp -o common.o
$CXX $CXXFLAGS $INCLUDES -c common/sampling.cpp -o sampling.o
$CXX $CXXFLAGS $INCLUDES -c common/console.cpp -o console.o
$CXX $CXXFLAGS $INCLUDES -c common/grammar-parser.cpp -o grammar-parser.o
$CXX $CXXFLAGS $INCLUDES -c build-info.cpp -o build-info.o
$CXX $CXXFLAGS $INCLUDES -c examples/main/main.cpp -o main.o

# Link
$CXX -O3 -mcpu=7450 ggml.o ggml-alloc.o ggml-backend.o ggml-quants.o \
    llama.o common.o sampling.o console.o grammar-parser.o build-info.o \
    main.o -o main -lm -lpthread
```

### Build with AltiVec SIMD
```bash
# Same as above but add -maltivec to CFLAGS/CXXFLAGS:
CFLAGS="-DNDEBUG -O3 -mcpu=7450 -maltivec -std=gnu11 -fno-strict-aliasing"
CXXFLAGS="-DNDEBUG -O3 -mcpu=7450 -maltivec -std=gnu++11 -fno-strict-aliasing"

# Also need to insert AltiVec code block into ggml.c (see below)
```

## Big-Endian Byte Swapping

### Option 1: Pre-Convert Model (Recommended)

```bash
# On any machine with Python 3
python3 /home/scott/ppc-tiger-tools/tools/gguf_byteswap.py \
    tinyllama-1.1b-q4.gguf \
    tinyllama-1.1b-q4-BE.gguf

# Copy to G4
scp tinyllama-1.1b-q4-BE.gguf g4-mirror:~/models/
```

The converter handles:
- Header: magic, version, tensor count, kv count
- Metadata: all key-value pairs
- Tensor info: names, dimensions, types, offsets
- Tensor data: F32 (4-byte swap), F16 (2-byte swap), Q4_K/Q6_K (scale swaps)

### Option 2: Runtime Byte-Swapping

Insert this code at line ~19309 in ggml.c (before `gguf_fread_el`):

```c
// Big-endian byte swap helpers for GGUF (PowerPC fix)
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define GGUF_IS_BIG_ENDIAN 1
#elif defined(__BIG_ENDIAN__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc__)
#define GGUF_IS_BIG_ENDIAN 1
#else
#define GGUF_IS_BIG_ENDIAN 0
#endif

static inline uint16_t gguf_bswap16(uint16_t x) { return (x >> 8) | (x << 8); }
static inline uint32_t gguf_bswap32(uint32_t x) {
    return ((x >> 24) & 0x000000FF) | ((x >> 8) & 0x0000FF00) |
           ((x << 8) & 0x00FF0000) | ((x << 24) & 0xFF000000);
}
static inline uint64_t gguf_bswap64(uint64_t x) {
    return ((x >> 56) & 0x00000000000000FFULL) | ((x >> 40) & 0x000000000000FF00ULL) |
           ((x >> 24) & 0x0000000000FF0000ULL) | ((x >> 8) & 0x00000000FF000000ULL) |
           ((x << 8) & 0x000000FF00000000ULL) | ((x << 24) & 0x0000FF0000000000ULL) |
           ((x << 40) & 0x00FF000000000000ULL) | ((x << 56) & 0xFF00000000000000ULL);
}

static bool gguf_fread_scalar(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
#if GGUF_IS_BIG_ENDIAN
    if (n == size) {
        switch (size) {
            case 2: *(uint16_t*)dst = gguf_bswap16(*(uint16_t*)dst); break;
            case 4: *(uint32_t*)dst = gguf_bswap32(*(uint32_t*)dst); break;
            case 8: *(uint64_t*)dst = gguf_bswap64(*(uint64_t*)dst); break;
        }
    }
#endif
    return n == size;
}
```

Then change these reads from `gguf_fread_el` to `gguf_fread_scalar`:
- `ctx->header.version`, `n_tensors`, `n_kv`
- `kv->type`, `kv->value.*` (all 2+ byte scalars)
- `info->n_dims`, `info->ne[]`, `info->type`, `info->offset`

Keep `gguf_fread_el` for: magic (4 chars), string data, uint8/int8/bool, bulk tensor data.

**Important**: If using pre-converted BE model, set `GGUF_IS_BIG_ENDIAN 0` to disable runtime swapping.

## AltiVec SIMD Support

### What AltiVec Provides
- 128-bit vector registers
- vec_madd (fused multiply-add)
- vec_add, vec_ld, vec_st
- Potential 2-4x speedup on F32 operations

### Key Differences from POWER9
| Operation | POWER9 | AltiVec (G4/G5) |
|-----------|--------|-----------------|
| Unaligned load | `vec_xl` | `vec_ld` + alignment check |
| Extract element | `vec_extract` | Union trick |
| F16 support | Native | None (use F32) |

### AltiVec Code Block

Insert at line ~864 in ggml.c (before `#elif defined(__POWER9_VECTOR__)`):

```c
#elif defined(__ALTIVEC__) && !defined(__POWER9_VECTOR__)

// Classic AltiVec for PowerPC G4/G5
#define GGML_SIMD
#include <altivec.h>

#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4              vector float
#define GGML_F32x4_ZERO         ((vector float){0.0f, 0.0f, 0.0f, 0.0f})
#define GGML_F32x4_SET1(x)      ((vector float){(x), (x), (x), (x)})

// Aligned load with scalar fallback for unaligned
static inline vector float ggml_vec_load_f32(const float *p) {
    if (((unsigned long)p & 0xF) == 0) {
        return vec_ld(0, p);
    } else {
        union { vector float v; float f[4]; } u;
        u.f[0] = p[0]; u.f[1] = p[1]; u.f[2] = p[2]; u.f[3] = p[3];
        return u.v;
    }
}

static inline void ggml_vec_store_f32(float *p, vector float v) {
    if (((unsigned long)p & 0xF) == 0) {
        vec_st(v, 0, p);
    } else {
        union { vector float v; float f[4]; } u;
        u.v = v;
        p[0] = u.f[0]; p[1] = u.f[1]; p[2] = u.f[2]; p[3] = u.f[3];
    }
}

#define GGML_F32x4_LOAD(p)      ggml_vec_load_f32(p)
#define GGML_F32x4_STORE(p, r)  ggml_vec_store_f32(p, r)
#define GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define GGML_F32x4_ADD          vec_add
#define GGML_F32x4_MUL(a, b)    vec_madd(a, b, GGML_F32x4_ZERO)

// Reduction using union (no vec_extract on G4/G5)
#define GGML_F32x4_REDUCE(res, x)                     \
{                                                     \
    int offset = GGML_F32_ARR >> 1;                   \
    for (int i = 0; i < offset; ++i) {                \
        x[i] = vec_add(x[i], x[offset+i]);            \
    }                                                 \
    offset >>= 1;                                     \
    for (int i = 0; i < offset; ++i) {                \
        x[i] = vec_add(x[i], x[offset+i]);            \
    }                                                 \
    offset >>= 1;                                     \
    for (int i = 0; i < offset; ++i) {                \
        x[i] = vec_add(x[i], x[offset+i]);            \
    }                                                 \
    union { vector float v; float f[4]; } _u;         \
    _u.v = x[0];                                      \
    res = _u.f[0] + _u.f[1] + _u.f[2] + _u.f[3];      \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 - no native support, use F32 path
#define GGML_F16_STEP       GGML_F32_STEP
#define GGML_F16_EPR        GGML_F32_EPR
#define GGML_F16_VEC        GGML_F32x4
#define GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F16_VEC_SET1   GGML_F32x4_SET1
#define GGML_F16_VEC_FMA    GGML_F32x4_FMA
#define GGML_F16_VEC_ADD    GGML_F32x4_ADD
#define GGML_F16_VEC_MUL    GGML_F32x4_MUL
#define GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
#define GGML_F16_VEC_LOAD(p, i) GGML_F32x4_SET1(GGML_FP16_TO_FP32((p)[i]))
#define GGML_F16_VEC_STORE(p, r, i) /* not vectorized */
```

### Verifying AltiVec is Active

After building, run:
```bash
./main -m model.gguf -p "test" -n 1 2>&1 | grep -i altivec
```

Or check system_info output for `VSX = 1` or custom AltiVec banner.

## Testing

### Basic Test
```bash
./main -m ~/models/tinyllama-1.1b-q4-BE.gguf -p "2+2=" -n 5 --no-display-prompt
```

### Expected Output (Working)
```
llama_model_loader: loaded meta data with 23 key-value pairs and 201 tensors
llama_model_loader: - type  f32:   45 tensors
llama_model_loader: - type q4_K:  135 tensors
llama_model_loader: - type q6_K:   21 tensors
llm_load_tensors:        CPU buffer size =   636.18 MiB

llama_print_timings: prompt eval time =   50752.52 ms /     6 tokens
llama_print_timings:        eval time =  142114.64 ms /     9 runs
```

### Signs of Problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| Magic "FUGG" | LE model without swapping | Use BE model or enable runtime swap |
| Crash at ggml_compute_forward_mul | Byte order issues | Apply big-endian fix |
| Garbage output | Tensor data not swapped correctly | Check gguf_byteswap.py Q4_K handling |
| VSX = 0 in system_info | AltiVec not compiled in | Add -maltivec flag |

## Performance Reference

| System | Model | pp (tok/s) | tg (tok/s) |
|--------|-------|------------|------------|
| G4 1.1GHz (scalar) | TinyLlama Q4 | 0.12 | 0.06 |
| G4 1.1GHz (AltiVec F32) | TinyLlama Q4 | 0.12 | 0.06 |
| G5 2.3GHz (scalar) | TinyLlama Q4 | 0.41 | 0.16 |
| POWER8 (VSX+PSE) | TinyLlama Q4 | 147.54 | 18.88 |

**Note**: AltiVec F32-only provides no speedup because Q4_K models spend 90% of time in quantized operations. Need AltiVec paths in `ggml-quants.c` for real gains.

## Known Issues

1. **Output Quality**: Generated text may be garbled. Q4_K/Q6_K block byte-swapping in converter needs refinement.

2. **AltiVec Q4_K Not Implemented**: The hot path functions in `ggml-quants.c` need AltiVec versions:
   - `ggml_vec_dot_q4_K_q8_K` (line 6226) - 90% of compute time
   - `dequantize_row_q4_K` (line 2318)
   - Currently only NEON (ARM) and AVX2 (x86) have optimized paths

## Future Work: AltiVec Q4_K

To get real speedup on quantized models, need to add AltiVec to ggml-quants.c:

```c
#elif defined(__ALTIVEC__) && !defined(__POWER9_VECTOR__)
// TODO: AltiVec Q4_K implementation
// Key operations needed:
// 1. vec_ld - load 16 bytes
// 2. vec_and - mask nibbles (& 0x0F0F0F0F)
// 3. vec_sr - shift right for high nibbles
// 4. vec_msum - multiply-accumulate (vec_mladd on G4)
// 5. Horizontal sum via vec_add cascade
```

The challenge is Q4_K uses:
- Nibble unpacking (4-bit → 8-bit)
- Scale multiplication per 32-element group
- Accumulation across 256 elements

2. **No F16 SIMD**: AltiVec lacks native F16, falls back to scalar conversion.

3. **Alignment**: AltiVec vec_ld requires 16-byte alignment. Unaligned data uses scalar fallback.

## Troubleshooting

### SSH Connection Issues
```bash
# Add to ~/.ssh/config for old Mac SSH:
Host g4-mirror
    HostName 192.168.0.125
    User sophia
    PubkeyAuthentication no
    HostKeyAlgorithms +ssh-rsa
```

### GCC Not Found
```bash
# Install Tigerbrew GCC
/usr/local/bin/brew install gcc@7
export CC=/usr/local/bin/gcc-7
export CXX=/usr/local/bin/g++-7
```

### Model Too Large
G4 has limited RAM. Use smaller models:
- TinyLlama 1.1B Q4: 636 MiB (works)
- Llama 7B Q4: ~4 GB (may work on G5)
- Larger models: Use numpy_llm.py with mmap

## December 2025
