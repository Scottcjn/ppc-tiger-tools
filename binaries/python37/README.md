# Python 3.7 Binaries for PowerPC Mac OS X

Pre-built Python packages for PowerPC G4/G5 Macs running Mac OS X 10.4 Tiger or 10.5 Leopard.

## Packages

| Package | Version | Size | Description |
|---------|---------|------|-------------|
| NumPy | 1.16.6 | 5.0 MB | Scientific computing with BLAS/LAPACK |
| Cython | 0.29.36 | 2.7 MB | C extensions for Python |

## Performance Results

Tested on PowerPC G5 (Dual 2.0GHz) running Samsung HRM transformer inference:

| Implementation | Time | Speedup |
|----------------|------|---------|
| Python 2.5 + NumPy 1.6 | 76.02s | 1.0x |
| **Python 3.7 + NumPy 1.16** | **13.03s** | **5.8x** |
| C++ AltiVec | 15.56s | 4.9x |

**Python 3.7 with modern NumPy beats hand-optimized C++ with AltiVec!**

## Installation

Requires Python 3.7 installed on PowerPC Mac OS X.

```bash
# Extract NumPy
cd ~
tar xzf numpy-1.16.6-py37-macosx-ppc.tar.gz

# Extract Cython (optional, for building other packages)
tar xzf cython-0.29.36-py37-macosx-ppc.tar.gz

# Verify installation
python3.7 -c "import numpy; print(numpy.__version__)"
# Output: 1.16.6
```

## Building From Source

If you need to rebuild these packages:

```bash
# 1. Install Cython first
tar xzf Cython-0.29.36.tar.gz
cd Cython-0.29.36
python3.7 setup.py install --user

# 2. Build NumPy
unzip numpy-1.16.6.zip
cd numpy-1.16.6
python3.7 setup.py build
python3.7 setup.py install --user
```

## Why NumPy 1.16?

NumPy 1.21+ dropped PowerPC architecture detection. NumPy 1.16.6 is the last version with full PowerPC support and still provides excellent performance through:

- Optimized BLAS/LAPACK integration with Apple Accelerate
- Efficient memory operations
- Modern Python 3.7 features (f-strings, type hints, etc.)

## Credits

- Built on PowerPC G5 running Mac OS X 10.5 Leopard
- December 2025
