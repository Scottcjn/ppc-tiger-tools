# NumPy 2.2.6 PowerPC (32-bit) Support Patch

This patch adds support for 32-bit PowerPC (G4/G5) on Mac OS X to NumPy 2.2.6.

## IMPORTANT: Python Version Requirement

**NumPy 2.x requires Python 3.10+**

The PowerPC G5 running Mac OS X 10.5 currently has Python 3.7. To use NumPy 2.x, you would need to build Python 3.10+ from source first.

For Python 3.7, use **NumPy 1.21.6** instead (see `../numpy-1.21.6/`).

## Changes

This patch modifies two files:

### 1. `meson_cpu/meson.build`
- Adds `'ppc': []` to `min_features` dict
- Adds `'ppc': PPC64_FEATURES` to `max_features_dict`

### 2. `numpy/_core/src/common/npy_cpu_features.c`
- Adds `NPY_CPU_PPC` to the POWER section
- Correctly handles G4/G5 which have AltiVec but not VSX

## How to Apply (requires Python 3.10+)

```bash
# Download NumPy 2.2.6
wget https://github.com/numpy/numpy/archive/refs/tags/v2.2.6.tar.gz
tar xzf v2.2.6.tar.gz
cd numpy-2.2.6

# Apply patch
patch -p1 < ../numpy-2.2.6-ppc32-support.patch

# Build with meson (requires meson and meson-python)
pip install meson meson-python cython
pip install . --no-build-isolation
```

## Building Python 3.10+ on PowerPC

To use this patch, you first need Python 3.10+. Building from source:

```bash
wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tar.xz
tar xf Python-3.10.14.tar.xz
cd Python-3.10.14
./configure --prefix=$HOME/python310
make -j2
make install
```

## December 2025
