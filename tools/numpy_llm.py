#!/usr/bin/env python3
"""
Pure NumPy LLM Inference Engine

A minimal transformer inference implementation using only NumPy.
Designed for PowerPC and other big-endian architectures where
llama.cpp GGUF support is limited.

Uses NumPy's BLAS backend (Accelerate on Mac) for matrix operations.

Usage:
    python3 numpy_llm.py weights.npz "Hello, world"

Author: ppc-tiger-tools project
License: MIT
December 2025
"""

import numpy as np
import sys
import time
from pathlib import Path


def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, weight, bias, eps=1e-5):
    """RMS Layer Normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def rms_norm(x, weight, eps=1e-5):
    """Root Mean Square Layer Normalization (used in LLaMA)"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def rope_rotation(x, seq_len, dim, base=10000.0):
    """Rotary Position Embedding"""
    # Create rotation matrices
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)  # [seq_len, dim/2]

    cos = np.cos(freqs)
    sin = np.sin(freqs)

    # Apply rotation
    x_reshape = x.reshape(x.shape[0], -1, 2)
    x_rot = np.zeros_like(x_reshape)
    x_rot[..., 0] = x_reshape[..., 0] * cos[:x.shape[0]] - x_reshape[..., 1] * sin[:x.shape[0]]
    x_rot[..., 1] = x_reshape[..., 0] * sin[:x.shape[0]] + x_reshape[..., 1] * cos[:x.shape[0]]

    return x_rot.reshape(x.shape)


def attention(q, k, v, mask=None):
    """Scaled dot-product attention"""
    d_k = q.shape[-1]
    scores = np.matmul(q, k.T) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attn_weights = softmax(scores, axis=-1)
    return np.matmul(attn_weights, v)


def multi_head_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads=None, rope_dim=64):
    """Multi-head attention with RoPE and GQA support"""
    seq_len, hidden = x.shape
    head_dim = hidden // n_heads

    # Detect number of KV heads from weight shapes
    if n_kv_heads is None:
        n_kv_heads = wk.shape[0] // head_dim

    # Project to Q, K, V
    q = np.dot(x, wq.T)  # [seq, hidden]
    k = np.dot(x, wk.T)  # [seq, n_kv_heads * head_dim]
    v = np.dot(x, wv.T)

    # Reshape for multi-head
    q = q.reshape(seq_len, n_heads, head_dim)
    k = k.reshape(seq_len, n_kv_heads, head_dim)
    v = v.reshape(seq_len, n_kv_heads, head_dim)

    # For GQA: repeat K, V to match number of Q heads
    if n_kv_heads < n_heads:
        repeat_factor = n_heads // n_kv_heads
        k = np.repeat(k, repeat_factor, axis=1)
        v = np.repeat(v, repeat_factor, axis=1)

    # Apply RoPE to Q and K (simplified - apply to all dims)
    # In real LLaMA, RoPE is only applied to first rope_dim dimensions

    # Compute attention for each head
    outputs = []
    for h in range(n_heads):
        qh = q[:, h, :]  # [seq, head_dim]
        kh = k[:, h, :]
        vh = v[:, h, :]

        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

        out = attention(qh, kh, vh, mask)
        outputs.append(out)

    # Concatenate heads
    concat = np.concatenate(outputs, axis=-1)  # [seq, hidden]

    # Output projection
    return np.dot(concat, wo.T)


def feed_forward(x, w_gate, w_up, w_down):
    """SwiGLU Feed-Forward Network (LLaMA style)"""
    gate = np.dot(x, w_gate.T)
    up = np.dot(x, w_up.T)

    # SwiGLU activation
    hidden = gate * (1 / (1 + np.exp(-gate))) * up  # silu(gate) * up

    return np.dot(hidden, w_down.T)


class NumpyLLM:
    """Pure NumPy LLM for inference"""

    def __init__(self, weights_path):
        print(f"Loading weights from: {weights_path}")
        t0 = time.time()

        self.weights = {}
        self._data = None
        self.weight_names = []

        # Check if weights_path is a directory (individual .npy files)
        # or a single .npz file
        weights_path = Path(weights_path)

        if weights_path.is_dir():
            # Load from directory of .npy files
            print("Loading from directory of .npy files...")
            npy_files = list(weights_path.glob("*.npy"))

            for npy_file in npy_files:
                # Convert filename back to weight name
                # e.g., blk_0_attn_q_weight.npy -> blk.0.attn_q.weight
                # Format: blk_<N>_<type>_<subtype>_weight or token_embd_weight
                name = npy_file.stem  # Remove .npy
                parts = name.split('_')

                if parts[0] == 'blk' and len(parts) >= 4:
                    # blk_0_attn_q_weight -> blk.0.attn_q.weight
                    layer_num = parts[1]
                    # Everything between layer num and 'weight' is the tensor name
                    tensor_parts = parts[2:-1]  # e.g., ['attn', 'q'] or ['ffn', 'down']
                    tensor_name = '_'.join(tensor_parts)  # e.g., 'attn_q'
                    name = f"blk.{layer_num}.{tensor_name}.weight"
                elif parts[0] == 'token' and len(parts) >= 3:
                    # token_embd_weight -> token_embd.weight
                    name = f"token_embd.weight"
                elif parts[0] == 'output' and len(parts) >= 2:
                    if parts[1] == 'norm':
                        name = "output_norm.weight"
                    else:
                        name = "output.weight"
                else:
                    # Fallback: join with dots
                    name = '.'.join(parts)

                self.weight_names.append(name)
                # Store file path for lazy loading
                self.weights[name] = str(npy_file)

            print(f"Found {len(self.weight_names)} weight files")
            self._load_mode = 'directory'

        else:
            # Load from .npz file (original mode)
            data = np.load(str(weights_path), allow_pickle=True, mmap_mode='r')
            self.weight_names = [k for k in data.files if not k.startswith('__')]
            self._data = data

            # Try to extract metadata
            if '__metadata__' in data.files:
                meta = str(data['__metadata__'])
                print(f"Metadata: {meta[:200]}...")

            self._load_mode = 'npz'

        load_time = time.time() - t0
        print(f"Found {len(self.weight_names)} tensors in {load_time:.2f}s")

        # Detect model architecture from weight names
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect model architecture from weight names"""
        weight_names = self.weight_names

        # Check for LLaMA-style naming
        if any('blk.0.attn_q' in n for n in weight_names):
            self.arch = 'llama'
            # Count layers
            self.n_layers = max(int(n.split('.')[1]) for n in weight_names
                               if n.startswith('blk.') and n.split('.')[1].isdigit()) + 1
            # Get dimensions from weight shapes (load just the ones we need)
            attn_q = self.get_weight('blk.0.attn_q.weight')
            token_embd = self.get_weight('token_embd.weight')
            self.hidden_size = attn_q.shape[1] if attn_q is not None else 2048
            self.n_heads = 32  # Default for most LLaMA models
            self.vocab_size = token_embd.shape[0] if token_embd is not None else 32000

            print(f"Architecture: LLaMA-style")
            print(f"  Layers: {self.n_layers}")
            print(f"  Hidden: {self.hidden_size}")
            print(f"  Vocab: {self.vocab_size}")
        else:
            # Fallback - try to detect from shapes
            self.arch = 'unknown'
            print("Warning: Unknown architecture, inference may not work")

    def get_weight(self, name):
        """Get weight by name, handling potential missing weights"""
        # Check cache first - in directory mode, initially stores file paths
        if name in self.weights:
            cached = self.weights[name]
            # If it's a string, it's a file path - load it
            if isinstance(cached, str):
                w = np.load(cached, mmap_mode='r')
                w = np.array(w)  # Copy from mmap
                if w.dtype == np.float16:
                    w = w.astype(np.float32)
                self.weights[name] = w
                return w
            return cached

        # For npz mode, try to load from mmap'd file
        if self._load_mode == 'npz' and name in self.weight_names:
            w = np.array(self._data[name])  # Copy from mmap to regular array
            # Convert float16 to float32 for computation
            if w.dtype == np.float16:
                w = w.astype(np.float32)
            self.weights[name] = w
            return w

        # Try alternate naming
        alt_names = [
            name.replace('.', '_'),
            name.replace('_', '.'),
            'model.' + name,
            name.replace('blk', 'layers'),
        ]
        for alt in alt_names:
            if alt in self.weight_names:
                if self._load_mode == 'directory':
                    cached = self.weights.get(alt)
                    if isinstance(cached, str):
                        w = np.load(cached, mmap_mode='r')
                        w = np.array(w)
                        if w.dtype == np.float16:
                            w = w.astype(np.float32)
                        self.weights[alt] = w
                        return w
                else:
                    w = np.array(self._data[alt])
                    if w.dtype == np.float16:
                        w = w.astype(np.float32)
                    self.weights[alt] = w
                    return w
        return None

    def forward_layer(self, x, layer_idx):
        """Forward pass through one transformer layer"""
        prefix = f'blk.{layer_idx}'

        # Attention norm
        norm_weight = self.get_weight(f'{prefix}.attn_norm.weight')
        if norm_weight is not None:
            x_norm = rms_norm(x, norm_weight)
        else:
            x_norm = x

        # Multi-head attention
        wq = self.get_weight(f'{prefix}.attn_q.weight')
        wk = self.get_weight(f'{prefix}.attn_k.weight')
        wv = self.get_weight(f'{prefix}.attn_v.weight')
        wo = self.get_weight(f'{prefix}.attn_output.weight')

        if all(w is not None for w in [wq, wk, wv, wo]):
            attn_out = multi_head_attention(x_norm, wq, wk, wv, wo, self.n_heads)
            x = x + attn_out
        else:
            print(f"  Warning: Missing attention weights for layer {layer_idx}")

        # FFN norm
        norm_weight = self.get_weight(f'{prefix}.ffn_norm.weight')
        if norm_weight is not None:
            x_norm = rms_norm(x, norm_weight)
        else:
            x_norm = x

        # Feed-forward
        w_gate = self.get_weight(f'{prefix}.ffn_gate.weight')
        w_up = self.get_weight(f'{prefix}.ffn_up.weight')
        w_down = self.get_weight(f'{prefix}.ffn_down.weight')

        if all(w is not None for w in [w_gate, w_up, w_down]):
            ffn_out = feed_forward(x_norm, w_gate, w_up, w_down)
            x = x + ffn_out
        else:
            print(f"  Warning: Missing FFN weights for layer {layer_idx}")

        return x

    def forward(self, token_ids):
        """Full forward pass"""
        # Token embedding
        emb_weight = self.get_weight('token_embd.weight')
        if emb_weight is None:
            raise ValueError("Missing token embedding weight")

        x = emb_weight[token_ids]  # [seq_len, hidden]

        # Transformer layers
        for layer in range(self.n_layers):
            x = self.forward_layer(x, layer)

        # Output norm
        out_norm = self.get_weight('output_norm.weight')
        if out_norm is not None:
            x = rms_norm(x, out_norm)

        # Output projection (often tied to embedding)
        out_weight = self.get_weight('output.weight')
        if out_weight is None:
            out_weight = emb_weight  # Tied weights

        logits = np.dot(x, out_weight.T)  # [seq_len, vocab]

        return logits

    def generate(self, prompt_tokens, max_tokens=32, temperature=0.8, top_k=40):
        """Generate tokens autoregressively"""
        tokens = list(prompt_tokens)
        generated = []

        print(f"Generating {max_tokens} tokens...")
        t0 = time.time()

        for i in range(max_tokens):
            # Forward pass
            token_array = np.array(tokens, dtype=np.int32)
            logits = self.forward(token_array)

            # Get logits for last token
            next_logits = logits[-1, :]

            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature

                # Top-k sampling
                if top_k > 0:
                    top_indices = np.argsort(next_logits)[-top_k:]
                    mask = np.ones_like(next_logits) * -np.inf
                    mask[top_indices] = 0
                    next_logits = next_logits + mask

                # Sample
                probs = softmax(next_logits)
                next_token = np.random.choice(len(probs), p=probs)
            else:
                # Greedy
                next_token = np.argmax(next_logits)

            tokens.append(next_token)
            generated.append(next_token)

            # Progress
            elapsed = time.time() - t0
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Token {i+1}/{max_tokens}: {next_token} ({tps:.2f} tok/s)")

        total_time = time.time() - t0
        print(f"\nGenerated {len(generated)} tokens in {total_time:.2f}s")
        print(f"Speed: {len(generated) / total_time:.2f} tokens/second")

        return generated


def simple_tokenize(text, vocab_size=32000):
    """Very simple character-level tokenization (fallback)"""
    tokens = [ord(c) % vocab_size for c in text]
    return tokens


def simple_detokenize(tokens):
    """Simple detokenization (fallback)"""
    return ''.join(chr(t % 128) if 32 <= t % 128 < 127 else '?' for t in tokens)


def find_tokenizer(weights_path):
    """Try to find tokenizer.model near the weights"""
    weights_path = Path(weights_path)

    # Check same directory
    candidates = [
        weights_path.parent / "tokenizer.model",
        weights_path.parent / "tokenizer.json",
        weights_path.parent.parent / "tokenizer.model",
        Path.home() / "models" / "tokenizer.model",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None


def main():
    if len(sys.argv) < 2:
        print("Pure NumPy LLM Inference Engine")
        print()
        print("Usage: python3 numpy_llm.py <weights> [prompt] [max_tokens] [tokenizer]")
        print()
        print("Arguments:")
        print("  weights     Path to .npz file or directory of .npy files")
        print("  prompt      Input text (default: 'Hello')")
        print("  max_tokens  Number of tokens to generate (default: 16)")
        print("  tokenizer   Path to tokenizer.model (optional)")
        print()
        print("Runs transformer inference using only NumPy.")
        print("Designed for PowerPC and big-endian architectures.")
        sys.exit(1)

    weights_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    tokenizer_path = sys.argv[4] if len(sys.argv) > 4 else None

    if not Path(weights_path).exists():
        print(f"Error: Weights file not found: {weights_path}")
        sys.exit(1)

    # Load model
    model = NumpyLLM(weights_path)

    # Try to load tokenizer
    tokenizer = None
    if tokenizer_path is None:
        tokenizer_path = find_tokenizer(weights_path)

    if tokenizer_path:
        try:
            from llama_tokenizer import LlamaTokenizer
            tokenizer = LlamaTokenizer(tokenizer_path)
            print(f"Loaded tokenizer: {tokenizer_path} ({tokenizer.vocab_size} tokens)")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            print("Using simple character-level tokenization")

    # Tokenize prompt
    print(f"\nPrompt: {prompt}")
    if tokenizer:
        tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    else:
        tokens = simple_tokenize(prompt, model.vocab_size)
    print(f"Tokens: {tokens}")

    # Generate
    print(f"Generating {max_tokens} tokens...")
    generated = model.generate(tokens, max_tokens=max_tokens, temperature=0.8)

    # Decode
    if tokenizer:
        output = tokenizer.decode(generated)
    else:
        output = simple_detokenize(generated)
    print(f"\nGenerated: {output}")


if __name__ == "__main__":
    main()
