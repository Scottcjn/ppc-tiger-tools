# Modern Compilers for Mac OS X Tiger (PowerPC)

These are modern GCC builds compiled for Mac OS X 10.4 Tiger on PowerPC.

## Available Compilers

### GCC 7.5.0 (gcc7-7.5.0-tiger-ppc.tar.gz)
- **Version**: GCC 7.5.0
- **Target**: PowerPC G4/G5
- **OS**: Mac OS X 10.4 Tiger / 10.5 Leopard
- **Source**: Built via Tigerbrew
- **Size**: ~59MB compressed

**Installation:**
```bash
cd /usr/local
sudo tar xzf gcc7-7.5.0-tiger-ppc.tar.gz
```

**After extraction:**
- `/usr/local/bin/gcc-7` → Main compiler
- `/usr/local/bin/g++-7` → C++ compiler
- `/usr/local/Cellar/gcc7/7.5.0/` → Full installation

**Usage:**
```bash
# C compilation
gcc-7 -mcpu=7450 -maltivec -O2 -o program program.c

# C++ compilation
g++-7 -mcpu=7450 -maltivec -O2 -o program program.cpp
```

## Why This Matters

Apple stopped supporting Tiger in 2007. The default GCC 4.0.1 cannot compile modern software.

These modern GCC builds enable:
- C11/C17 support
- C++14/C++17 support
- Modern optimizations
- Building Node.js, QuickJS, and other modern tools
- CVE patches for abandoned software

## RustChain Proof of Antiquity

These compilers were built to support RustChain's "Proof of Antiquity" consensus where vintage PowerPC hardware earns 2.5x mining rewards. Real vintage hardware requires real vintage software development.

## License

GCC is licensed under the GNU General Public License v3.

## Credits

Built by the RustChain/Sophiacord team for vintage Mac preservation.
