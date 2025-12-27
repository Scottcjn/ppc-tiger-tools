# LLVM 3.9.1 GCC 10 Compatibility Patches

These patches fix compilation issues when building LLVM 3.9.1 with GCC 10.x on Mac OS X Leopard (10.5).

## Successfully Built

With these patches, we successfully built:
- **Clang 3.9.1** - Full C/C++ compiler
- **56 LLVM tools** - llc, lli, opt, llvm-ar, llvm-nm, etc.
- Target: `powerpc-apple-darwin9.8.0`

## Quick Apply

```bash
cd /path/to/llvm-3.9.1.src

# 1. Fix CGOpenMPRuntime.cpp lambda conflicts (3 instances)
sed -i 's/\[&D, &CGF, &BasePointersArray/[&D, &BasePointersArray/g' tools/clang/lib/CodeGen/CGOpenMPRuntime.cpp
sed -i 's/\[&CGF, &BasePointersArray/[&BasePointersArray/g' tools/clang/lib/CodeGen/CGOpenMPRuntime.cpp
sed -i 's/\[&D, &CGF, Device\]/[&D, Device]/g' tools/clang/lib/CodeGen/CGOpenMPRuntime.cpp

# 2. Fix Lexer.cpp AltiVec keyword
sed -i 's/const vector unsigned char/const __vector unsigned char/g' tools/clang/lib/Lex/Lexer.cpp

# 3. Add strnlen polyfill to multiple files (Mac OS X 10.5 doesn't have strnlen)
for file in tools/clang/lib/Lex/HeaderMap.cpp \
            tools/llvm-pdbdump/LLVMOutputStyle.cpp \
            tools/obj2yaml/macho2yaml.cpp; do
    cat > /tmp/strnlen_fix.h << 'EOF'
#include <cstring>
#if defined(__APPLE__) && !defined(__MAC_10_7)
static inline size_t _strnlen_compat(const char *s, size_t maxlen) {
    const char *p = (const char *)memchr(s, '\0', maxlen);
    return p ? (size_t)(p - s) : maxlen;
}
#define strnlen _strnlen_compat
#endif
EOF
    cat /tmp/strnlen_fix.h "$file" > /tmp/fixed.cpp && mv /tmp/fixed.cpp "$file"
done
```

## Patch Descriptions

| Patch | Issue | Solution |
|-------|-------|----------|
| `llvm-3.9.1-gcc10-cgopenmp.patch` | Lambda parameter 'CGF' conflicts with capture | Remove `&CGF` from 3 capture lists |
| `llvm-3.9.1-gcc10-lexer-altivec.patch` | `vector` keyword conflicts in C++ | Use `__vector` for AltiVec types |
| `llvm-3.9.1-gcc10-strnlen-polyfill.patch` | `strnlen` not in 10.5 libc | Add inline strnlen polyfill |
| `llvm-3.9.1-gcc10-llvmoutputstyle-strnlen.patch` | Same strnlen issue | Same fix |
| `llvm-3.9.1-gcc10-headermap-strnlen.patch` | Same strnlen issue | Same fix |

## Known Issues (Optional Tools)

Some auxiliary tools may fail to build:
- **obj2yaml** - strnlen issue (apply polyfill)
- **llvm-objdump** - xar library functions missing on 10.5

These are optional tools. The main clang/llvm binaries build successfully.

## Notes

- These issues only appear with GCC 10+ due to stricter C++ standards enforcement
- GCC 7.x and earlier compile LLVM 3.9.1 without these patches
- Mac OS X 10.5 Leopard lacks `strnlen()` in libc (added in 10.7)
- Built with distributed compilation using ppc-distcc on 2x G5 Macs
