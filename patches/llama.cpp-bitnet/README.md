# BitNet I2_S Support for llama.cpp

Native BitNet b1.58 inference with ternary weights {-1, 0, +1} for llama.cpp.

## Features

- **I2_S quantization type** (GGML_TYPE_I2_S = 36) - 2-bit ternary weights
- **BitNet architecture** (`bitnet-b1.58`) support
- **Big-endian GGUF** support for PowerPC
- **AltiVec SIMD** acceleration (9x speedup on G4/G5)
- **Cross-platform** - works on x86, ARM, PowerPC

## Supported Models

- Microsoft BitNet b1.58 3B
- Any GGUF model with I2_S quantized weights

## Quick Start

```bash
# Clone llama.cpp b2000 or later
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Apply BitNet patches
python3 /path/to/patches/apply_all_patches.py

# Build
make -j4

# Run
./main -m bitnet-3b-i2s.gguf -p "Hello" -n 100
```

## Patch Files

| File | Description |
|------|-------------|
| `apply_all_patches.py` | Master patch script - applies all patches |
| `patch_ggml_i2s.py` | Adds I2_S type to ggml.c |
| `patch_gguf_bool.py` | Fixes bool size for big-endian platforms |
| `patch_bitnet_arch.py` | Adds BitNet architecture to llama.cpp |
| `ggml-i2s-patch.h` | I2_S type definitions |
| `ggml-i2s-altivec.h` | AltiVec SIMD kernels for PowerPC |

## Technical Details

### I2_S Format
- 256 weights per block
- 64 bytes packed data (2 bits per weight)
- Encoding: 00=-1, 01=0, 10=+1, 11=unused
- Per-tensor scale factor (FP16)

### BitNet Architecture
Similar to LLaMA but with:
- Ternary I2_S weights for Q, K, V, O, FFN layers
- Sub-layer RMS norms after attention and in FFN
- F32 normalization weights, F16 embeddings

### Performance (PowerPC G4)
- AltiVec dot product: 9x speedup vs scalar
- ~2560 M ops/s ternary multiply-accumulate

## Platform Notes

### PowerPC (Tiger/Leopard)
- Use GCC 7+ for C++11 support
- Requires `LLAMA_NO_METAL=1 LLAMA_NO_ACCELERATE=1`
- Big-endian GGUF models required

### x86/ARM  
- Standard llama.cpp build process
- Little-endian GGUF models (default)

## License

MIT License - Same as llama.cpp
