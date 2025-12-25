# PPC Tiger Tools

**A collection of tools and utilities for Mac OS X Tiger (10.4) and Leopard (10.5) on PowerPC G4/G5**

Run modern software on 20-year-old Macs!

## What's Included

### AltiVec/Velocity Engine Tools
| File | Description |
|------|-------------|
| `altivec_test.c` | Basic AltiVec SIMD test |
| `altivec_cognitive_transform.c` | Cognitive transforms using AltiVec |
| `transformer_altivec.c` | Transformer architecture with AltiVec |
| `transformer_g4.c` | G4-optimized transformer |
| `benchmark_transformer.c` | Performance benchmarks |

### LLM Inference on G4
| File | Description |
|------|-------------|
| `g4_llm.c` | Basic LLM inference for G4 |
| `g4_llm_demo.c` | LLM demonstration |
| `g4_chat.c` | Interactive chat on G4 |
| `sophia_chat_g4.c` | Sophia AI chat client |
| `tiny_sophia_g4.c` | Minimal Sophia implementation |
| `g4_beast_mode.c` | Maximum performance mode |

### AltiVec LLM Framework (`altivec_llm/`)
- `llm_transformer_altivec.c` - Full transformer with AltiVec
- `sophia_subbrain.c` - Distributed AI node
- `Makefile.ppc` - PowerPC build configuration

### Crypto & Mining (Python 2.3 Compatible!)
| File | Description |
|------|-------------|
| `rustchain_miner_ppc23.py` | RustChain miner for Python 2.3 |
| `rustchain_wallet_ppc23.py` | Wallet for Python 2.3 |
| `g4_miner.py` | G4 mining client |
| `g4_rustchain_miner.py` | Full RustChain miner |
| `ppc_g4_entropy.c` | Hardware entropy collector |
| `g4_mac_entropy_collector.cpp` | Mac-specific entropy |

### TLS 1.2 Shims (Modern HTTPS on Tiger!)
| File | Description |
|------|-------------|
| `openssl_tls12_shim.c` | TLS 1.2 support for old OpenSSL |
| `tls12_system_shim.c` | System-level TLS shim |

### Server Components
| File | Description |
|------|-------------|
| `altivec_http_service.c` | HTTP service with AltiVec |

## Building

### Prerequisites
- Mac OS X Tiger (10.4) or Leopard (10.5)
- Xcode 2.5 (Tiger) or Xcode 3.1 (Leopard)
- GCC 4.0 with AltiVec support

### Compile for G4
```bash
gcc -O3 -mcpu=7450 -maltivec -o program program.c
```

### Compile for G5
```bash
gcc -O3 -mcpu=970 -maltivec -o program program.c
```

### Using the Makefile
```bash
cd altivec_llm
make -f Makefile.ppc
```

## Python 2.3 Compatibility

Tiger ships with Python 2.3. The `*_ppc23.py` scripts are compatible:

```bash
python rustchain_miner_ppc23.py
python rustchain_wallet_ppc23.py
```

## TLS 1.2 on Tiger

Tiger's OpenSSL only supports TLS 1.0. The shims enable TLS 1.2:

```bash
gcc -o tls_shim openssl_tls12_shim.c -lssl -lcrypto
```

## Hardware Requirements

| Machine | RAM | Status |
|---------|-----|--------|
| Power Mac G5 | 2GB+ | Best performance |
| PowerBook G4 | 1GB+ | Works well |
| Power Mac G4 | 1GB+ | Works |
| iMac G4/G5 | 1GB+ | Works |
| Mac mini G4 | 512MB+ | Limited |

## Part of Sophiacord

These tools are part of the Sophiacord distributed AI system. The G4/G5 Macs serve as vintage compute nodes alongside modern POWER8 servers.

## Performance Tips

1. **Use AltiVec** - 4x speedup on vector operations
2. **Optimize for your CPU** - Use `-mcpu=7450` (G4) or `-mcpu=970` (G5)
3. **Disable Spotlight** - Frees up CPU cycles
4. **Use RAM disk** - For temporary files
5. **Close Dashboard** - Saves memory on Tiger

## Related Projects

- [ppc-compilers](https://github.com/Scottcjn/ppc-compilers) - **GCC 7, GCC 10, Perl 5.34 for PowerPC**
- [rust-ppc-tiger](https://github.com/Scottcjn/rust-ppc-tiger) - Rust compiler for PowerPC
- [llama-cpp-tigerleopard](https://github.com/Scottcjn/llama-cpp-tigerleopard) - llama.cpp for Tiger/Leopard
- [claude-code-power8](https://github.com/Scottcjn/claude-code-power8) - Claude Code for POWER8
- [llama-cpp-power8](https://github.com/Scottcjn/llama-cpp-power8) - llama.cpp for POWER8

## Attribution

**A year of work went into this project.** If you use it, please give credit:

```
PowerPC Tiger Tools by Scott (Scottcjn)
https://github.com/Scottcjn/ppc-tiger-tools
```

If this helped you, please:
- ⭐ **Star this repo** - It helps others find it
- 📝 **Credit in your project** - Keep the attribution
- 🔗 **Link back** - Share the love

## License

MIT License - Free to use, but please keep the copyright notice and attribution.

---

*"Your 2005 Mac still has value. Let it compute."*

**A year of development, real hardware, electricity bills, and a dedicated lab went into this. Free for you to use - just give credit where it's due.**
