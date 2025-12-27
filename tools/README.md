# PowerPC Tiger Tools - Utilities

Conversion and compatibility tools for PowerPC Macs.

## gguf_byteswap.py

Converts GGUF model files from little-endian (x86/ARM) to big-endian format for PowerPC G4/G5 Macs.

### The Problem

GGUF files (used by llama.cpp) are stored in little-endian byte order. PowerPC Macs are big-endian. Without byte-swapping, GGUF models fail to load with errors like:
- "failed to read key-value pairs"
- "Non-aligned pointer being freed"
- Segmentation faults during model loading

### Usage

```bash
python3 gguf_byteswap.py <input.gguf> <output_be.gguf>
```

### Example

```bash
# Convert TinyLlama model to big-endian
python3 gguf_byteswap.py tinyllama-1.1b-q4.gguf tinyllama-1.1b-q4-be.gguf

# Transfer to G5
scp tinyllama-1.1b-q4-be.gguf user@g5-mac:~/models/
```

### Supported Formats

| Type | Description |
|------|-------------|
| F32 | Full precision float |
| F16 | Half precision float |
| Q4_0 | 4-bit quantization |
| Q4_1 | 4-bit quantization with min |
| Q4_K | K-quant 4-bit |
| Q5_K | K-quant 5-bit |
| Q6_K | K-quant 6-bit |
| Q8_0 | 8-bit quantization |

### Requirements

- Python 3.x (tested with 3.9)
- No external dependencies (uses only stdlib)

### How It Works

1. Reads GGUF header and swaps magic/version bytes
2. Converts all metadata key-value pairs to big-endian
3. Reads tensor info (names, dimensions, offsets) and converts
4. For each tensor, swaps bytes based on data type:
   - F32: 4-byte swap
   - F16: 2-byte swap
   - Quantized: Swap scale values in block headers

### Performance

| Model Size | Conversion Time |
|------------|-----------------|
| 638 MB (TinyLlama Q4) | ~30 seconds |
| 4 GB (7B Q4) | ~3 minutes |

### Known Limitations

- Large models (70B+) require significant memory
- Some exotic quantization formats may not be fully supported
- Always verify converted models by loading in llama.cpp

### After Conversion

Use the converted model with llama.cpp on your G5:

```bash
# Build llama.cpp with Leopard patches (see ../patches/llama.cpp-b2000/)
cd ~/llama.cpp-b2000
./main -m ~/models/tinyllama-1.1b-q4-be.gguf -p "Hello" -n 32
```

### Note on llama.cpp Compatibility

Even with byte-swapped GGUF files, llama.cpp needs additional patches for big-endian support. The GGUF loader in llama.cpp assumes little-endian throughout. See the patches directory for compatibility work in progress.

## December 2025
