# NumPy 1.21.6 PowerPC (32-bit) Support Patch

This patch adds support for 32-bit PowerPC (G4/G5) on Mac OS X to NumPy 1.21.6.

## Background

NumPy 1.17+ dropped explicit 32-bit PowerPC support for Mac OS X. The `#error "unknown architecture"` in `numpyconfig.h` was only checking for `__x86_64` and `__arm64__`, ignoring PowerPC.

## Changes

This patch modifies three files:

### 1. `numpy/core/include/numpy/numpyconfig.h`
- Adds PowerPC architecture detection (`__ppc__`, `__powerpc__`)
- Sets correct long double sizes for IBM double-double format (16 bytes)

### 2. `numpy/core/src/common/npy_cpu_features.c.src`
- Adds `NPY_CPU_PPC` to the POWER section
- Correctly handles G4/G5 which have AltiVec but not VSX

### 3. `numpy/distutils/ccompiler_opt.py`
- Adds `ppc` architecture detection pattern
- Adds `cc_on_ppc` compiler flag
- Adds 32-bit PPC to architecture lists

## How to Apply

```bash
# Download NumPy 1.21.6
wget https://github.com/numpy/numpy/archive/refs/tags/v1.21.6.tar.gz
tar xzf v1.21.6.tar.gz
cd numpy-1.21.6

# Apply patch
patch -p1 < ../numpy-1.21.6-ppc32-support.patch

# Build (requires Cython 0.29.x)
python3.7 setup.py build
python3.7 setup.py install --user
```

## Performance Results

Tested on PowerPC G5 (Dual 2.0GHz) running Samsung HRM transformer inference:

| NumPy Version | Time | Notes |
|---------------|------|-------|
| NumPy 1.16.6 | 13.03s | Last official PPC support |
| **NumPy 1.21.6 (patched)** | **12.84s** | Faster! |

## Requirements

- Mac OS X 10.4 Tiger or 10.5 Leopard
- Python 3.7+
- Cython 0.29.x (for building from source)
- GCC 4.0+ (Xcode) or GCC 7+ (MacPorts/Homebrew)

## December 2025
