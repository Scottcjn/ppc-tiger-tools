# Python 3.8.18 Mac OS X Tiger (10.4) Compatibility Patch

This patch enables Python 3.8.18 to build on Mac OS X Tiger (10.4, Darwin 8).

## Issue

Python 3.8+ uses `copyfile.h` for efficient file copying on macOS, but this API
was introduced in Leopard (10.5, Darwin 9). On Tiger, including this header
causes build failures.

## Changes

This patch modifies two files:

### 1. `Modules/posixmodule.c`
- Guards `copyfile.h` include with version check `__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 1050`
- Defines `HAVE_FCOPYFILE` macro to conditionally compile fcopyfile code
- Guards `COPYFILE_DATA` constant export

### 2. `Modules/clinic/posixmodule.c.h`
- Guards the fcopyfile method definition with `HAVE_FCOPYFILE`
- Defines empty `OS__FCOPYFILE_METHODDEF` when not available

## Requirements

- **libffi**: Python 3.8 ctypes requires libffi. Build libffi 3.4.x first:
  ```bash
  wget https://github.com/libffi/libffi/releases/download/v3.4.4/libffi-3.4.4.tar.gz
  tar xzf libffi-3.4.4.tar.gz
  cd libffi-3.4.4
  ./configure --prefix=$HOME/libffi_install
  make && make install
  ```

## How to Apply

```bash
# Download Python 3.8.18
wget https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tar.xz
tar xf Python-3.8.18.tar.xz
cd Python-3.8.18

# Apply patch
patch -p1 < ../python-3.8.18-tiger-copyfile.patch

# Configure with libffi
export LDFLAGS="-L$HOME/libffi_install/lib"
export CPPFLAGS="-I$HOME/libffi_install/include"
./configure --prefix=$HOME/python38_install

# Build
make -j2 && make install
```

## Known Limitations

- **No SSL**: Tiger's OpenSSL is too old for Python 3.8's ssl module
- **No _hashlib**: Related to SSL
- **No readline**: Need to install readline library

## Tested On

- Mac OS X 10.4.12 (Tiger) - PowerPC G4 Dual 1.25GHz
- Darwin 8.11.0
- GCC 4.0.1

## December 2025
