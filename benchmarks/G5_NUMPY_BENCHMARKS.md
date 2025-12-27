# PowerPC G5 NumPy 1.24 Benchmarks

Benchmarks run on PowerPC G5 Dual 2.3GHz with NumPy 1.24.4 using Apple Accelerate framework.

## System Configuration

| Component | Value |
|-----------|-------|
| **CPU** | PowerPC G5 Dual 2.3GHz (970fx) |
| **RAM** | 8GB DDR2 |
| **OS** | Mac OS X 10.5.8 (Leopard) |
| **Compiler** | GCC 10.5.0 |
| **Python** | 3.9.19 |
| **NumPy** | 1.24.4 |
| **BLAS** | Apple Accelerate (vecLib) |
| **Endianness** | Big-endian |

## Matrix Multiplication (GEMM)

Using `np.dot()` with float32 matrices:

| Size | Time | GFLOPS |
|------|------|--------|
| 500x500 | 0.065s | **3.87** |
| 1000x1000 | 0.154s | **13.01** |
| 2000x2000 | 2.133s | **7.50** |

Peak performance: **13 GFLOPS** at 1000x1000

## Transformer Operations Benchmark

Simulating LLM operations with:
- Batch size: 1
- Sequence length: 128
- Hidden dimension: 2048
- Attention heads: 16
- Vocabulary: 32,000

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Embedding lookup | 3.41 | Index-based, memory bound |
| QKV projection | 372.67 | (128, 2048) @ (2048, 6144) |
| Attention scores | 13.08 | Batched matmul |
| Attention softmax | 35.53 | Pure NumPy |
| Attention @ V | 15.60 | Batched matmul |
| FFN up projection | 623.67 | (128, 2048) @ (2048, 8192) |
| FFN down projection | 16,610.25 | (128, 8192) @ (8192, 2048) |
| Vocab projection | 4,618.91 | (128, 2048) @ (2048, 32000) |

## Analysis

### Strong Points
- Accelerate framework provides hardware-optimized BLAS
- Peak 13 GFLOPS competitive for 2005 hardware
- AltiVec/VMX SIMD utilized by vecLib

### Limitations
- Big-endian architecture limits GGUF model compatibility
- Large vocab projections are memory-bandwidth limited
- No half-precision (FP16) acceleration

### Comparison

| System | Peak GFLOPS | Year |
|--------|-------------|------|
| G5 2.3GHz Dual | 13 | 2005 |
| G4 1.25GHz Dual | ~2 | 2003 |
| Modern x86 (AVX2) | 100+ | 2020+ |

## Running the Benchmarks

```bash
# SSH to G5
ssh sophia@192.168.0.130

# Run matrix benchmark
~/python39_install/bin/python3.9 -c "
import numpy as np
import time
for size in [500, 1000, 2000]:
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    t0 = time.time()
    c = np.dot(a, b)
    t = time.time() - t0
    gflops = (2 * size**3) / t / 1e9
    print(f'{size}x{size}: {t:.3f}s ({gflops:.2f} GFLOPS)')
"
```

## December 2025
