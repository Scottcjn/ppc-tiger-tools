# Native BitNet I2_S for PowerPC AltiVec/VSX

**First native BitNet implementation for PowerPC!** December 2025.

## What is This?

This package enables running Microsoft's native BitNet b1.58 models on PowerPC systems (G4, G5, POWER8+) using AltiVec/VSX SIMD acceleration.

### Key Features

- **Native ternary weights**: Weights are exactly {-1, 0, +1} - no approximation
- **Pure integer math**: No floating point until final output scaling
- **40% sparsity**: 40% of weights are zero = 40% fewer operations!
- **2-bit storage**: 4 weights per byte = extreme compression
- **Big-endian support**: Full byte-swap converter for PowerPC

## Model Information

| Property | Value |
|----------|-------|
| Model | BitNet b1.58 2B4T |
| Parameters | 2 billion |
| Format | I2_S (Type 36) |
| Size | 1.13 GB |
| Weights | 210 I2_S tensors |
| Embeddings | 1 F16 tensor (328M elements) |
| Norms | 121 F32 tensors |

## I2_S Format (Reverse Engineered)

Microsoft's I2_S format is elegantly simple:

```
Per-Tensor Header (32 bytes):
  bytes 0-1:  FP16 scale factor
  bytes 2-31: Reserved/padding

Per Block (256 weights → 64 bytes):
  64 bytes packed ternary data
  - 2 bits per weight
  - 4 weights per byte
  - Encoding: 00=-1, 01=0, 10=+1, 11=unused
```

**No per-block scales!** Native BitNet models are trained to output exact ternary values.

## Weight Distribution (Actual Data)

```
From blk.0.ffn_down.weight (17.7M weights):
  -1:  29.4%  (negative)
   0:  40.5%  (zero - FREE computation savings!)
  +1:  30.1%  (positive)

Operations needed: 59.5% (40.5% savings from zeros!)
```

## Files Included

```
tools/
├── i2s_to_bigendian.py      # Convert I2_S GGUF to big-endian
├── reverse_i2s_format.py    # Format analysis tool
├── test_i2s_inference.py    # Verify ternary math
└── analyze_i2s_format.py    # Deep format inspection

patches/llama.cpp-bitnet/
└── ggml-i2s-altivec.h       # AltiVec kernels for I2_S
```

## Quick Start

### 1. Download Native BitNet Model

```bash
# From Hugging Face
wget https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main/ggml-model-i2_s.gguf
```

### 2. Convert to Big-Endian (for PowerPC)

```bash
python3 i2s_to_bigendian.py ggml-model-i2_s.gguf ggml-model-i2_s-BE.gguf
```

Output:
```
Converting: ggml-model-i2_s.gguf
Output: ggml-model-i2_s-BE.gguf
  GGUF version: 3
  Tensor count: 332
  Tensor type distribution:
    Type 0 (F32): 121 tensors
    Type 1 (F16): 1 tensors
    Type 36 (I2_S (BitNet)): 210 tensors
  Converting tensor data...
  Input size:  1,187,801,280 bytes (1132.8 MB)
  Output size: 1,187,801,280 bytes (1132.8 MB)
  Size match: OK
```

### 3. Verify Conversion

```bash
python3 test_i2s_inference.py ggml-model-i2_s-BE.gguf --big-endian
```

Output:
```
Testing I2_S inference (big-endian)
Tensor scale (FP16): 2634.0
Weight distribution:
  -1:   3015 (29.4%)
   0:   4146 (40.5%)
  +1:   3079 (30.1%)
PASS: All weights are ternary {-1, 0, +1}
PASS: Ternary math verified!
```

## AltiVec Implementation

### Core Insight

With ternary weights {-1, 0, +1}, matrix multiplication becomes:

```
y = Σ(w_i × x_i)
  = Σ(x_i where w=+1) - Σ(x_i where w=-1)
  = pure integer addition/subtraction!
```

### Key Functions

```c
// Unpack 16 ternary weights from 4 bytes
vector signed char unpack_i2s_16(const uint8_t* packed);

// Integer dot product (no FP until final scale!)
vector signed int vec_dot_i2s_q8(
    vector signed char weights,      // 16 x {-1,0,+1}
    vector signed char activations); // 16 x int8

// Full block dot product
int32_t ggml_vec_dot_i2s_q8_block(
    const block_i2s* weights,   // 256 packed weights
    const int8_t* activations); // 256 Q8 values
```

### G4 AltiVec Optimization

```c
// vec_perm-based lookup for ternary unpacking
static const vector signed char lut = {
    -1, 0, 1, 0,  // Maps 2-bit code to ternary
    -1, 0, 1, 0,
    -1, 0, 1, 0,
    -1, 0, 1, 0
};

// One vec_perm unpacks 16 weights!
return vec_perm(lut, lut, indices);
```

## Performance Expectations

### PowerPC G4 1.5 GHz

| Metric | Value | Notes |
|--------|-------|-------|
| Weights loaded | 256 per 64 bytes | 2.1x smaller than Q4_K |
| FP operations | Only at final scale | Integer compute |
| Sparsity skip | 40% of ops | Zero weights |
| Cache efficiency | Excellent | Tiny working set |

**Estimated**: 2-3x faster than Q4_K on same model size.

### POWER8

With VSX (128-bit SIMD) and 576GB RAM:
- Full model in memory
- 128 threads for parallel inference
- vec_msum for optimized dot products

## Why Native BitNet Matters

### Post-Quantization vs Native Training

| Approach | Quality | Compression |
|----------|---------|-------------|
| Post-quantize to ternary | BROKEN | 2.28x |
| **Native BitNet training** | **GOOD** | **~4x** |

Our earlier Q1.58 post-quantization experiments showed that converting a trained model to ternary destroys the learned patterns - output becomes uniform/random.

Microsoft's BitNet b1.58 is **trained from scratch** with ternary weights. The model learns to work with {-1, 0, +1} constraints, producing coherent outputs.

## Integration with llama.cpp

To add I2_S support to llama.cpp:

1. Add type constant:
```c
#define GGML_TYPE_I2_S 36
```

2. Add block structure:
```c
typedef struct {
    uint8_t qs[64];  // 256 weights @ 2 bits
} block_i2s;
```

3. Include AltiVec header:
```c
#include "ggml-i2s-altivec.h"
```

4. Register dequantization and dot product functions.

## References

- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764) - Microsoft Research
- [BitNet.cpp](https://github.com/microsoft/BitNet) - Official framework
- [Hugging Face Model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [AltiVec Programming](https://developer.apple.com/documentation/kernel/altivec)

## Authors

PPC-Tiger-Tools Project
December 2025

## License

MIT License - Use freely, attribution appreciated.
