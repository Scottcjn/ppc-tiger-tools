# Contributing to PPC Tiger Tools

Thanks for helping keep PowerPC Tiger and Leopard useful. This repository spans
AltiVec C, Python 2.3-compatible scripts, TLS shims, RustChain tooling, and
patches for old Apple hardware, so focused changes with clear hardware context
are easiest to review.

## Development Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/ppc-tiger-tools.git
   cd ppc-tiger-tools
   ```

2. Use the toolchain that matches the code you are changing:

   - Mac OS X Tiger 10.4 with Xcode 2.5 for Tiger compatibility work.
   - Mac OS X Leopard 10.5 with Xcode 3.1 for Leopard builds.
   - GCC 4.0 with AltiVec enabled for G4/G5 C code.
   - Python 2.3 when editing `*_ppc23.py` scripts.
   - Modern Python only for host-side conversion or helper tools that already
     require it.

3. Create a focused branch:

   ```bash
   git checkout -b fix/short-description
   ```

## Good Contributions

Helpful changes include:

- Build notes for specific G4/G5 machines, Mac OS X versions, or Xcode releases.
- Small AltiVec fixes that preserve big-endian and 32-bit PowerPC behavior.
- Python 2.3 compatibility fixes for RustChain miner and wallet scripts.
- TLS shim documentation and narrowly scoped compatibility improvements.
- Clear README updates for `patches/`, `tools/`, `tls/`, and release packages.

Avoid sweeping rewrites. Many files exist to preserve compatibility with old
compilers and operating systems where modern conveniences are unavailable.

## Code Style

- Prefer simple C that compiles with old Apple GCC where possible.
- Keep PowerPC endianness and alignment in mind for SIMD and model-format code.
- Use explicit comments around AltiVec intrinsics, byte swaps, and TLS shims.
- Keep Python 2.3 scripts free of f-strings, type annotations, pathlib, and
  modern standard-library assumptions.
- Do not commit generated binaries, private keys, wallet files, or local lab
  configuration.

## Verification

Include the checks you actually ran in the PR description:

- For C changes, list the compiler, flags, and target CPU, such as
  `gcc -O3 -mcpu=7450 -maltivec`.
- For Python 2.3 scripts, run or syntax-check them with Python 2.3 when
  available.
- For TLS changes, state the Mac OS X version, OpenSSL version, and endpoint
  used for testing.
- For docs-only changes, proofread the edited section and verify links or file
  paths.

If you cannot test on real PowerPC hardware, say so and describe the emulator,
cross-compiler, or static review you used.

## Pull Requests

Before opening a PR:

- Keep the diff limited to one tool, patch family, or documentation topic.
- Explain the affected hardware or OS version.
- Include exact build or verification steps.
- Preserve attribution and AGPL v3 licensing expectations from the README.
- Update related docs when behavior, build flags, or compatibility changes.

Use concise commit messages such as `fix: preserve big-endian q1_58 loads` or
`docs: clarify Tiger TLS shim setup`.

## Reporting Issues

When filing a bug, include:

- Machine model, CPU, RAM, and Mac OS X version.
- Compiler, Python version, or patch set used.
- The exact command that failed.
- Any crash log, compiler output, or runtime error text.
- Whether the issue occurs on real hardware, QEMU, or another emulator.

Hardware details matter here. A PowerBook G4, Power Mac G5, and Mac mini G4 can
behave differently even when running similar software.
