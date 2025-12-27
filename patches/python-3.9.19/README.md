# Python 3.9.19 Mac OS X Leopard (10.5) Compatibility Patch

This patch enables Python 3.9.19 to build on Mac OS X Leopard (10.5, Darwin 9).

## Issue

Python 3.9+ uses `pthread_threadid_np()` for thread native IDs, but this API
was introduced in Snow Leopard (10.6, Darwin 10). On Leopard, calling this
function causes linker errors.

## Changes

This patch modifies one file:

### `Python/thread_pthread.h`
- Adds version check for `MAC_OS_X_VERSION_10_6`
- Falls back to `pthread_self()` on Leopard and Tiger
- Preserves original behavior on 10.6+

## Requirements

- **GCC 10+**: Python 3.9 benefits from C11 features. GCC 10 available via MacPorts/Tigerbrew.
- **libffi**: For ctypes module support.

```bash
# Install GCC 10 via MacPorts (if available) or build from source
export CC=/usr/local/bin/gcc-10
export CXX=/usr/local/bin/g++-10
```

## How to Apply

```bash
# Download Python 3.9.19
wget https://www.python.org/ftp/python/3.9.19/Python-3.9.19.tar.xz
tar xf Python-3.9.19.tar.xz
cd Python-3.9.19

# Apply patch
patch -p1 < ../python-3.9.19-leopard-pthread.patch

# Configure with GCC 10
export CC=/usr/local/bin/gcc-10
export CXX=/usr/local/bin/g++-10
./configure --prefix=$HOME/python39_install

# Build
make -j2 && make install
```

## Known Limitations

- **No SSL**: Leopard's OpenSSL is too old for Python 3.9's ssl module
- **No _hashlib**: Related to SSL
- **No _tkinter**: Tcl/Tk framework linking issue on Leopard

## Tested On

- Mac OS X 10.5.8 (Leopard) - PowerPC G5 Dual 2.3GHz
- Darwin 9.8.0
- GCC 10.5.0

## Why GCC 10?

Python 3.9 and NumPy 1.24+ require C++11 support. The stock GCC 4.0 on Leopard
doesn't support C++11. GCC 10 provides:
- C++11/14/17 support
- Better optimization for PowerPC
- Required for building NumPy 1.24+

## December 2025
