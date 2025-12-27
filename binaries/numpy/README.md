# NumPy 1.6.2 for Mac OS X 10.5 Leopard (PowerPC)

Pre-built NumPy with **AltiVec SIMD** and **Apple Accelerate Framework** support.

## Platform
- **OS**: Mac OS X 10.5 Leopard
- **Architecture**: PowerPC (G4/G5)
- **Python**: 2.5
- **Compiler**: GCC with `-faltivec` flag

## Features
- AltiVec SIMD vectorization via `-faltivec`
- Apple Accelerate framework for optimized BLAS/LAPACK
- vecLib integration for matrix operations

## Build Configuration
```
blas_opt_info:
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
    extra_compile_args = ['-faltivec', '-I/System/Library/Frameworks/vecLib.framework/Headers']

lapack_opt_info:
    extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
    extra_compile_args = ['-faltivec']
```

## Installation

```bash
# Extract to site-packages (requires admin)
cd /Library/Python/2.5/site-packages
sudo tar xzf /path/to/numpy-1.6.2-macosx-10.5-ppc-altivec.tar.gz

# Verify installation
PYTHONPATH=/Library/Python/2.5/site-packages python -c "import numpy; print numpy.__version__; numpy.show_config()"
```

**Note**: The system NumPy 1.0.1 in `/System/Library/Frameworks/Python.framework` takes precedence. Use `PYTHONPATH=/Library/Python/2.5/site-packages` to load 1.6.2.

## Usage with HRM Inference

```bash
cd ~/trm-sophiacord
PYTHONPATH=/Library/Python/2.5/site-packages python hrm_g5_raw.py
```

## Build from Source

If you need to rebuild:

```bash
# Download NumPy 1.6.2
curl -O https://github.com/numpy/numpy/archive/refs/tags/v1.6.2.tar.gz
tar xzf v1.6.2.tar.gz
cd numpy-1.6.2

# Create site.cfg for Accelerate
cat > site.cfg << 'EOF'
[DEFAULT]
library_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
include_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers

[blas]
blas_libs = BLAS
library_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
include_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers

[lapack]
lapack_libs = LAPACK
library_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A

[atlas]
atlas_libs = vecLib
library_dirs = /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
EOF

# Build and install
export CFLAGS="-faltivec"
python setup.py build
sudo python setup.py install
```

## Credits
- Built December 2025 for Samsung HRM Sudoku inference on PowerPC G5
- Part of the TRM-Sophiacord project
