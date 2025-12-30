# llama.cpp Big-Endian Support for PowerPC

Patches to enable llama.cpp to load GGUF model files on big-endian PowerPC systems.

## Status: Working (December 2025)

**Successfully tested on PowerPC G4 1.1GHz with Mac OS X 10.4 Tiger:**

```
llama.cpp b2000 running on G4 PowerPC!
- Model loads correctly (201 tensors, 636 MiB)
- Inference runs at ~0.06-0.12 tok/s
- No crashes or KERN_INVALID_ADDRESS errors
```

## The Problem

GGUF files store all data in little-endian byte order. PowerPC is big-endian. Without byte-swapping:
- Magic "GGUF" reads as "FUGG"
- Version numbers are wrong (3 → 50331648)
- Tensor dimensions are corrupted
- Model crashes immediately on load

## Two Solutions

### Solution 1: Pre-converted Big-Endian Model (Recommended)

Use `gguf_byteswap.py` to convert models to big-endian format once:

```bash
# On any system with Python 3
python3 gguf_byteswap.py tinyllama-1.1b-q4.gguf tinyllama-1.1b-q4-BE.gguf

# Transfer to PowerPC Mac
scp tinyllama-1.1b-q4-BE.gguf g4-mac:~/models/
```

Then use UNMODIFIED llama.cpp - no runtime byte-swapping needed since the model is already in native format.

### Solution 2: Runtime Byte-Swapping (Alternative)

Patch llama.cpp to swap bytes when reading little-endian GGUF files. This works but is more complex and slower.

## Files

| File | Description |
|------|-------------|
| `ggml-bigendian-block.c` | Code block with byte-swap helpers |
| `../../tools/gguf_byteswap.py` | GGUF converter (recommended approach) |

## How to Apply Runtime Byte-Swapping

Insert this block after line 19308 (before `gguf_fread_el` function) in `ggml.c`:

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

// Create gguf_fread_scalar for metadata swapping (keep gguf_fread_el for raw bytes)
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

Then change all scalar value reads in the GGUF loader from `gguf_fread_el` to `gguf_fread_scalar`:
- `ctx->header.version`
- `ctx->header.n_tensors`
- `ctx->header.n_kv`
- `kv->type`
- `kv->value.*` (all scalar types)
- `info->n_dims`, `info->ne[]`, `info->type`, `info->offset`

**Keep `gguf_fread_el` for:**
- Magic bytes (4 chars)
- String data (raw bytes)
- Single-byte values (uint8, int8, bool)
- Bulk tensor data (needs separate per-element handling)

## Build Instructions (G4/Tiger)

```bash
cd ~/llama.cpp-b2000

# Using GCC 7 from Tigerbrew
CC=/usr/local/bin/gcc-7
CXX=/usr/local/bin/g++-7
CFLAGS="-DNDEBUG -O3 -mcpu=7450 -std=gnu11 -fno-strict-aliasing"
CXXFLAGS="-DNDEBUG -O3 -mcpu=7450 -std=gnu++11 -fno-strict-aliasing"

# Compile all files
$CC $CFLAGS -I. -c ggml.c -o ggml.o
$CC $CFLAGS -I. -c ggml-alloc.c -o ggml-alloc.o
$CC $CFLAGS -I. -c ggml-backend.c -o ggml-backend.o
$CC $CFLAGS -I. -c ggml-quants.c -o ggml-quants.o
$CXX $CXXFLAGS -I. -c llama.cpp -o llama.o
$CXX $CXXFLAGS -I. -I./common -c common/common.cpp -o common.o
$CXX $CXXFLAGS -I. -I./common -c common/sampling.cpp -o sampling.o
$CXX $CXXFLAGS -I. -I./common -c common/console.cpp -o console.o
$CXX $CXXFLAGS -I. -I./common -c common/grammar-parser.cpp -o grammar-parser.o
$CXX $CXXFLAGS -I. -I./common -c build-info.cpp -o build-info.o
$CXX $CXXFLAGS -I. -I./common -c examples/main/main.cpp -o main.o

# Link
$CXX -O3 -mcpu=7450 ggml.o ggml-alloc.o ggml-backend.o ggml-quants.o \
    llama.o common.o sampling.o console.o grammar-parser.o build-info.o \
    main.o -o main -lm -lpthread
```

## Test Results

**PowerPC G4 (Dual 1.1GHz, Mac OS X 10.4 Tiger):**

```
$ ./main -m ~/models/tinyllama-1.1b-q4-BE.gguf -p "2+2=" -n 10

llama_model_loader: loaded meta data with 23 key-value pairs and 201 tensors
llama_model_loader: - type  f32:   45 tensors
llama_model_loader: - type q4_K:  135 tensors
llama_model_loader: - type q6_K:   21 tensors
llm_load_print_meta: model size       = 636.18 MiB (4.85 BPW)
llm_load_tensors:        CPU buffer size =   636.18 MiB

llama_print_timings: prompt eval time =   50752.52 ms /     6 tokens ( 8458.75 ms per token,     0.12 tokens per second)
llama_print_timings:        eval time =  142114.64 ms /     9 runs   (15790.52 ms per token,     0.06 tokens per second)
llama_print_timings:       total time =  192928.09 ms /    15 tokens
```

**Performance Summary:**
- Load time: ~25 seconds for 636 MiB model
- Prompt eval: 0.12 tok/s (8.5 seconds per token)
- Generation: 0.06 tok/s (16 seconds per token)

## Known Issues

### Output Quality
The model generates tokens but output may be garbled. This suggests tensor data byte-swapping needs additional work for quantized formats (Q4_K, Q6_K).

The `gguf_byteswap.py` converter handles:
- F32: 4-byte swap (full support)
- F16: 2-byte swap (full support)
- Q4_K: d/dmin scale swaps (may need refinement)
- Q6_K: d scale swap (may need refinement)

### Performance
PowerPC G4 is very slow for LLM inference:
- ~0.06 tok/s generation (vs ~100 tok/s on modern CPUs)
- No SIMD acceleration (AltiVec not yet enabled for llama.cpp)

For practical use on G4, consider:
- Smaller models (GPT-2 124M)
- Pure Python inference with NumPy (`numpy_llm.py`)
- Using the model for research/educational purposes

## What Was Fixed

1. **Metadata byte order** - Header values now read correctly
2. **Key-value parsing** - All 23 metadata entries parse correctly
3. **Tensor info** - Dimensions, types, offsets are correct
4. **No crashes** - Previously crashed at `ggml_compute_forward_mul`

## December 2025
