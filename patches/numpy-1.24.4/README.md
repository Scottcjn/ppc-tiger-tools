# NumPy 1.24.4 PowerPC G5 (Leopard 10.5) Build Guide

This guide explains how to build NumPy 1.24.4 on PowerPC G5 Mac with GCC 10.

## Requirements

- Mac OS X 10.5.8 (Leopard) on PowerPC G5
- Python 3.9+ (built with GCC 10)
- GCC 10.5.0 (C++11 support required)
- Cython 0.29.x (NOT 3.x)

## Python 3.9 Requirement

NumPy 1.24 requires Python 3.8+, but we recommend Python 3.9 for better compatibility.
See `../python-3.9.19/` for the Leopard compatibility patch.

## Build Instructions

```bash
# Download NumPy 1.24.4
curl -LO https://github.com/numpy/numpy/archive/refs/tags/v1.24.4.tar.gz
tar xzf v1.24.4.tar.gz
cd numpy-1.24.4

# Set compiler and critical flags
export CC=/usr/local/bin/gcc-10
export CXX=/usr/local/bin/g++-10

# IMPORTANT: Use these flags to avoid bus errors on PPC
export CFLAGS="-mcpu=970 -O2 -fno-strict-aliasing"
export NPY_DISABLE_SVML=1

# Build
~/python39_install/bin/python3.9 setup.py build

# Install
~/python39_install/bin/python3.9 setup.py install
```

## Critical Build Flags

| Flag | Purpose |
|------|---------|
| `-mcpu=970` | Optimize for G5 (970 processor) |
| `-O2` | Safe optimization level |
| `-fno-strict-aliasing` | Prevents bus errors from pointer aliasing |
| `NPY_DISABLE_SVML=1` | Disable Intel SVML (not relevant on PPC) |

**DO NOT use `-maltivec`** - NumPy 1.24's SIMD detection doesn't properly support AltiVec and may cause runtime crashes.

## Performance

On PowerPC G5 Dual 2.3GHz with Accelerate framework:

| Benchmark | Time |
|-----------|------|
| 1000x1000 matrix multiply | 0.28s |

Accelerate framework automatically detected and used for BLAS/LAPACK.

## Cython Compatibility

NumPy 1.24 requires Cython 0.29.x, NOT Cython 3.x:

```bash
# Download Cython 0.29.37
pip download --no-binary=:all: 'cython<3'

# Install locally (no SSL needed)
tar xzf Cython-0.29.37.tar.gz
cd Cython-0.29.37
python3.9 setup.py install
```

## Tested On

- Mac OS X 10.5.8 (Leopard) - PowerPC G5 Dual 2.3GHz
- Darwin 9.8.0
- GCC 10.5.0
- Python 3.9.19

## Known Issues

- NumPy's CPU feature detection shows "unsupported" architecture
- No VSX/AVX optimizations (G5 only has AltiVec)
- Accelerate framework provides hardware BLAS acceleration

## December 2025
