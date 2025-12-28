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
| 638 MB (TinyLlama Q4) | ~15 seconds |
| 4 GB (7B Q4) | ~3 minutes |

### Verified Working (December 2025)

Successfully tested on PowerPC G5 Dual 2.3GHz with llama.cpp b2000:

```
$ ./main -m tinyllama-1.1b-q4-be.gguf -p "The meaning of life is" -n 32

The meaning of life is to find your purpose, your passion, and live with purpose.
- Mark Twain I am not interested in money or material possessions...

llama_print_timings:        load time =    4989.12 ms
llama_print_timings: prompt eval time =   14757.75 ms /     6 tokens (0.41 tokens/s)
llama_print_timings:        eval time =  196746.44 ms /    31 runs   (0.16 tokens/s)
```

**First successful LLM inference on PowerPC G5 using GGUF format!**

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

---

## gguf_to_numpy.py

Extracts and dequantizes GGUF model weights to NumPy format (.npy files). NumPy handles endianness automatically, making the weights portable across architectures.

### Usage

```bash
python3 gguf_to_numpy.py <input.gguf> <output.npz>
```

### Why This Matters

- GGUF format is little-endian only
- NumPy's .npy format is endian-safe
- Extracted weights work on any architecture (x86, ARM, PowerPC)

### Supported Quantization Formats

| Format | Status |
|--------|--------|
| F32 | ✅ Full support |
| F16 | ✅ Full support |
| Q4_0 | ✅ Dequantized to F32 |
| Q4_K | ✅ Dequantized to F32 |
| Q6_K | ✅ Dequantized to F32 |

---

## numpy_llm.py

Pure NumPy LLM inference engine. Runs transformer inference using only NumPy - no llama.cpp, no CUDA, no special dependencies.

### Usage

```bash
# From .npz file
python3 numpy_llm.py weights.npz "Hello world" 16

# From directory of .npy files
python3 numpy_llm.py tinyllama_weights/ "Hello world" 16
```

### Arguments

- `weights_path`: Path to .npz file or directory of .npy files
- `prompt`: Input text (default: "Hello")
- `max_tokens`: Number of tokens to generate (default: 16)

### Features

- GQA (Grouped Query Attention) support
- RMS Layer Normalization
- SwiGLU Feed-Forward
- Memory-mapped weight loading
- Auto-detects model architecture from weights

### Verified Working (December 2025)

| System | Python | NumPy | Speed |
|--------|--------|-------|-------|
| PowerPC G5 (Mac OS X 10.5) | 3.9.18 | 1.24.4 | 0.01 tok/s |
| PowerPC G4 (Mac OS X 10.4) | 3.8.18 | 1.21.6 | 0.005 tok/s |

**First LLM to run on Mac OS X 10.4 Tiger!**

### Performance Notes

Pure NumPy is ~16x slower than optimized llama.cpp, but:
- Works on any system with Python + NumPy
- No compilation required
- Endianness handled automatically
- Great for testing and compatibility verification

---

## Workflow: GGUF to PowerPC

1. **Convert GGUF to NumPy weights** (on any machine):
   ```bash
   python3 gguf_to_numpy.py tinyllama-1.1b-q4.gguf tinyllama_weights.npz
   ```

2. **Split for transfer** (if needed for old Macs):
   ```bash
   # Extract .npz to directory
   python3 -c "import numpy as np; d=np.load('weights.npz'); [np.save(f'weights/{k}.npy', d[k]) for k in d.files]"
   tar czf weights.tar.gz weights/
   ```

3. **Transfer to PowerPC Mac**:
   ```bash
   scp weights.tar.gz user@g4-mac:~/models/
   ssh user@g4-mac 'cd ~/models && tar xzf weights.tar.gz'
   ```

4. **Run inference**:
   ```bash
   python3 numpy_llm.py ~/models/weights/ "The meaning of life is" 8
   ```

## December 2025
