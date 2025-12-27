# llama.cpp Leopard (10.5) Compatibility

This patch enables llama.cpp (b2000 release) to compile on Mac OS X Leopard (10.5).

## Issue

llama.cpp uses modern POSIX APIs not available on Leopard:
- `clock_gettime()` - Added in macOS 10.12
- `posix_memalign()` - May not be properly declared on 10.5

## Solution

The `leopard_compat.h` header provides shims:
- `clock_gettime()` → Uses `gettimeofday()`
- `posix_memalign()` → Uses `valloc()` (page-aligned allocation)

## How to Apply

```bash
# Copy header to llama.cpp directory
cp leopard_compat.h ~/llama.cpp-b2000/

# Patch ggml.c to include it after <inttypes.h>
sed -i.bak 's/#include <inttypes.h>/#include <inttypes.h>\n\n#include "leopard_compat.h"/' ~/llama.cpp-b2000/ggml.c

# Build with GCC 10
cd ~/llama.cpp-b2000
export CC=/usr/local/bin/gcc-10
export CXX=/usr/local/bin/g++-10
make -j2 LLAMA_NO_METAL=1 LLAMA_NO_ACCELERATE=1 main
```

## Known Limitations

### Big-Endian GGUF Issue

**GGUF files are little-endian only.** PowerPC G5 is big-endian, so standard GGUF models will not load correctly. The file format stores integers and floats in little-endian byte order.

Symptoms:
- "failed to read key-value pairs"
- "Non-aligned pointer being freed" errors
- Segmentation faults during model loading

### Workarounds

1. **Pure Python inference** - Use NumPy-based implementations that handle endianness
2. **Byte-swap GGUF** - Convert model files to big-endian (requires custom tool)
3. **Patch llama.cpp** - Add endianness conversion in GGUF loader

## Tested On

- Mac OS X 10.5.8 (Leopard)
- PowerPC G5 Dual 2.3GHz
- GCC 10.5.0
- llama.cpp b2000 (January 2024)

## December 2025
