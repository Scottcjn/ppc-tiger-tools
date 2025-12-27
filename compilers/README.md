# Modern Compilers for Mac OS X Tiger/Leopard (PowerPC)

These are modern compiler toolchains built for vintage PowerPC Macs.

## Available Toolchains

| Directory | Version | Language Support | Size |
|-----------|---------|------------------|------|
| [gcc7/](gcc7/) | GCC 7.5.0 | C11/C17, C++14/17 | 59MB |
| [gcc10/](gcc10/) | GCC 10.5.0 | C17/C2x, C++17/20 | 86MB |
| [llvm-3.9/](llvm-3.9/) | LLVM/Clang 3.9.1 | C11/C14, C++11/14 | Patches only |
| [perl5/](perl5/) | Perl 5.34.3 | Modern Perl | 13MB |

## Quick Start

```bash
# Download and extract GCC 7
cd /usr/local
sudo tar xzf gcc7-7.5.0-tiger-ppc.tar.gz

# Compile with modern C++
gcc-7 -mcpu=7450 -maltivec -O2 -std=c++17 -o app app.cpp
```

## System Requirements

- Mac OS X 10.4 Tiger or 10.5 Leopard
- PowerPC G4 or G5 processor
- ~200MB disk space for all compilers

## RustChain Proof of Antiquity

These compilers enable running modern software on vintage hardware for RustChain's "Proof of Antiquity" consensus. Real 2025 code on 2005 hardware!

## Future Additions

- [ ] Python 3.10+ (available via MacPorts)
- [ ] Rust (in progress)
- [ ] Node.js (researching ppc32/ppc64 build)
- [x] LLVM/Clang 3.9.1 - **DONE!** GCC 10 compatibility patches in [llvm-3.9/](llvm-3.9/)

## Credits

Built by the RustChain/Sophiacord team for PowerPC preservation.
