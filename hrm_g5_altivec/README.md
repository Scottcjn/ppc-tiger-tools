# HRM Inference for PowerPC G5

C++ implementation of Samsung's Hierarchical Reasoning Model (HRM) for Sudoku, optimized for PowerPC G5 with explicit AltiVec SIMD and Apple Accelerate framework.

## Performance

| Implementation | Time | Speedup |
|---------------|------|---------|
| Python 2.5 + NumPy 1.0 | 77.63s | 1.0x |
| Python 2.5 + NumPy 1.6 (Accelerate) | 76.02s | 1.0x |
| C++ (Accelerate BLAS only) | 16.56s | 4.6x |
| C++ (AltiVec + Accelerate) | 15.56s | 4.9x |
| Python 3.7 + NumPy 1.16 | 13.03s | 5.8x |
| **Python 3.7 + NumPy 1.21 (patched)** | **12.84s** | **5.9x** |

**Python 3.7 with modern NumPy beats hand-optimized C++ with AltiVec!**

NumPy 1.21.6 required a [PowerPC support patch](../patches/numpy-1.21.6/) to build on 32-bit PowerPC.

## Requirements

- Mac OS X 10.5 Leopard (PowerPC G5)
- GCC 7+ or GCC 10 (installed via Homebrew/MacPorts or built from source)
- Apple Accelerate framework (included with OS X)

## Building

```bash
# Ensure g++-10 is in PATH
export PATH=/usr/local/bin:$PATH

# Build
make

# Or manually:
g++-10 -O1 -mcpu=970 -mtune=970 -maltivec -framework Accelerate -o hrm_inference hrm_inference.cpp
```

**Note**: `-O2` and `-O3` cause bus errors due to aggressive auto-vectorization. `-O1` is the highest safe optimization level.

## Usage

```bash
# Run inference
./hrm_inference weights_be

# Or specify weights directory
./hrm_inference /path/to/weights_be
```

## Weight Format

Weights are stored as big-endian binary files with format:
- 4 bytes: ndim (number of dimensions)
- ndim * 4 bytes: shape dimensions
- data: float32 values in row-major order

## Model Architecture

- Hidden size: 512
- Attention heads: 8
- Head dimension: 64
- H layers: 4
- L layers: 4
- Vocab size: 11 (0-9 digits + empty)
- Sequence length: 81 (9x9 grid)
- Recurrent steps: 16

## Key Optimizations

1. **Accelerate BLAS**: Matrix multiplies via `cblas_sgemm` use optimized AltiVec SIMD internally
2. **Explicit AltiVec**: Hand-written intrinsics for dot product, RMS norm, softmax, vector add
3. **970-tuned**: Compiled with `-mcpu=970 -mtune=970 -maltivec` for G5-specific scheduling
4. **Aligned memory**: All buffers use `valloc()` for 16-byte alignment (AltiVec requirement)
5. **Minimal allocations**: Reuses buffers across transformer layers

## AltiVec Functions

The following operations use explicit AltiVec SIMD intrinsics:
- `dot_product_altivec()` - 4 floats per cycle using `vec_madd`
- `sum_squares_altivec()` - For RMS normalization
- `scale_vector_altivec()` - Broadcast multiply with `vec_splats`
- `add_vectors_altivec()` - Simple vector addition
- `find_max_altivec()` - For softmax normalization using `vec_max`

## Future Improvements

- [x] Add explicit AltiVec intrinsics for RMS norm, softmax, dot product
- [ ] Dual-CPU parallelization using pthreads
- [ ] Memory pooling to reduce allocation overhead
- [ ] Vectorized exp() using polynomial approximation

## Credits

- Samsung SAIL - Original HRM model architecture
- December 2025 - C++ port for PowerPC G5
