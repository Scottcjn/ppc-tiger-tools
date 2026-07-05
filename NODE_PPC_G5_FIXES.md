# Node.js v22 on PowerPC G5 (Mac OS X Leopard) - Build Fixes

## Target System
- **Hardware**: Power Mac G5 Dual 2.0GHz (PowerPC 970)
- **OS**: Mac OS X Leopard 10.5.8 (Darwin 9.8.0)
- **Compiler**: GCC 10.5.0 (cross-compiled for PPC)
- **Architecture**: PowerPC 64-bit Big Endian

## Prerequisites
- GCC 10.5.0 installed in `/usr/local/gcc-10/`
- Python 3.7+ for build scripts
- Node.js v22.12.0 source

## Key Fixes

### 1. 64-bit Mode Compilation
The G5 defaults to 32-bit mode but V8 requires 64-bit for `V8_TARGET_ARCH_PPC64`.

**Fix**: Add `-m64` to all CFLAGS in makefiles and replace `-arch i386` with `-m64`:

```bash
# In all *.target.mk files:
sed -i 's/-arch i386/-m64/g' out/*.target.mk out/*/*.target.mk
```

### 2. C++20 Standard
Node.js v22 requires C++20 features.

**Fix**: Add `-std=gnu++20` to all CFLAGS_CC_Release in makefiles.

### 3. Missing C++ Runtime Symbols
GCC 10's libstdc++ is 32-bit only. Use system libstdc++ with compatibility shims.

**Create `stdc++_compat.cpp`**:
```cpp
// Compatibility shims for missing libstdc++ symbols
#include <cstdlib>
#include <new>

namespace std {
  void __throw_bad_function_call() { abort(); }
}

// C++14 sized deallocation - forward to regular delete
void operator delete(void* ptr, unsigned long) noexcept {
  ::operator delete(ptr);
}
```

**Compile and add to link**:
```bash
/usr/local/gcc-10/bin/g++ -m64 -c stdc++_compat.cpp -o stdc++_compat.o
```

### 4. PPC64 GPR Save/Restore Functions
The 64-bit build requires register save/restore functions not in system libgcc.

**Create `ppc64_gpr.s`**:
```asm
.text
.align 2

.globl _saveGPR
.globl saveGPR
_saveGPR:
saveGPR:
    std r14, -144(r1)
    std r15, -136(r1)
    std r16, -128(r1)
    std r17, -120(r1)
    std r18, -112(r1)
    std r19, -104(r1)
    std r20, -96(r1)
    std r21, -88(r1)
    std r22, -80(r1)
    std r23, -72(r1)
    std r24, -64(r1)
    std r25, -56(r1)
    std r26, -48(r1)
    std r27, -40(r1)
    std r28, -32(r1)
    std r29, -24(r1)
    std r30, -16(r1)
    std r31, -8(r1)
    blr

.globl _restGPR
.globl restGPR
_restGPR:
restGPR:
    ld r14, -144(r1)
    ld r15, -136(r1)
    ld r16, -128(r1)
    ld r17, -120(r1)
    ld r18, -112(r1)
    ld r19, -104(r1)
    ld r20, -96(r1)
    ld r21, -88(r1)
    ld r22, -80(r1)
    ld r23, -72(r1)
    ld r24, -64(r1)
    ld r25, -56(r1)
    ld r26, -48(r1)
    ld r27, -40(r1)
    ld r28, -32(r1)
    ld r29, -24(r1)
    ld r30, -16(r1)
    ld r31, -8(r1)
    blr

.globl _restGPRx
.globl restGPRx
_restGPRx:
restGPRx:
    ld r14, -144(r1)
    ld r15, -136(r1)
    ld r16, -128(r1)
    ld r17, -120(r1)
    ld r18, -112(r1)
    ld r19, -104(r1)
    ld r20, -96(r1)
    ld r21, -88(r1)
    ld r22, -80(r1)
    ld r23, -72(r1)
    ld r24, -64(r1)
    ld r25, -56(r1)
    ld r26, -48(r1)
    ld r27, -40(r1)
    ld r28, -32(r1)
    ld r29, -24(r1)
    ld r30, -16(r1)
    ld r31, -8(r1)
    ld r0, 16(r1)
    mtlr r0
    blr
```

**Assemble**:
```bash
/usr/local/gcc-10/bin/gcc -m64 -c ppc64_gpr.s -o ppc64_gpr.o
```

### 5. OpenSSL Configuration
Use `linux-ppc64` with `no-asm` configuration for Big Endian PPC.

**Create `deps/openssl/config/archs/linux-ppc64/no-asm/crypto/buildinf.h`**:
```c
#define PLATFORM "platform: linux-ppc64"
#define DATE "built on: reproducible build, date unspecified"
static const char compiler_flags[] = {
    'c','o','m','p','i','l','e','r',':',' ',
    'g','c','c','-','1','0','.','5','.','0',
    ' ','-','m','6','4','\0'
};
```

### 6. Complete OpenSSL Headers
Copy all OpenSSL headers to the config-specific include directory:

```bash
for f in deps/openssl/openssl/include/openssl/*.h; do
    name=$(basename $f)
    if [ ! -f deps/openssl/config/archs/linux-ppc64/no-asm/include/openssl/$name ]; then
        cp $f deps/openssl/config/archs/linux-ppc64/no-asm/include/openssl/
    fi
done
```

### 7. LIBS Modification
Update makefile LIBS to use system libstdc++:

```makefile
LIBS := \
    -L/usr/lib -lstdc++.6 -lm
```

Add compat objects to LD_INPUTS:
```makefile
LD_INPUTS := \
    ... \
    $(builddir)/stdc++_compat.o \
    $(builddir)/ppc64_gpr.o
```

## Configure Command
```bash
./configure \
    --openssl-no-asm \
    --without-inspector \
    --without-intl \
    --dest-cpu=ppc64 \
    --dest-os=mac
```

## Build Command
After applying all fixes, build from the `out` directory to avoid reconfiguration:
```bash
cd out && make BUILDTYPE=Release V=1
```

## Known Issues
- V8 may have additional PPC64 Big Endian compatibility issues
- Some tests may fail due to platform-specific assumptions
- Inspector/debugging features may not work

## Status
Build in progress - OpenSSL completed, continuing with V8 and Node core.

## Author
Built with Claude Code assistance for the RustChain/Sophiacord project.
GitHub: https://github.com/Scottcjn/node-ppc
