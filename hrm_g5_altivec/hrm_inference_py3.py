#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Inference for PowerPC G5 - Python 3.7+ / NumPy 1.17+
Samsung Hierarchical Reasoning Model for Sudoku

Uses modern NumPy features:
- keepdims parameter
- @ matrix multiply operator
- f-strings
- Type hints
"""
import numpy as np
import struct
import time
import sys
import os
from typing import Dict, Tuple, List

CONFIG = {
    'hidden_size': 512,
    'num_heads': 8,
    'head_dim': 64,
    'H_layers': 4,
    'L_layers': 4,
    'vocab_size': 11,
    'seq_len': 81,
    'max_steps': 16,
}


def load_weight(weights_dir: str, name: str) -> np.ndarray:
    """Load a weight tensor from big-endian binary file."""
    fname = name.replace('.', '_') + '.bin'
    fpath = os.path.join(weights_dir, fname)

    with open(fpath, 'rb') as f:
        # Read ndim (big-endian uint32)
        ndim = struct.unpack('>I', f.read(4))[0]

        # Read shape
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndim))

        # Read data as big-endian float32
        total_size = np.prod(shape)
        data = f.read(total_size * 4)
        arr = np.frombuffer(data, dtype='>f4')

    return arr.astype(np.float32).reshape(shape)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax with keepdims."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Root Mean Square Layer Normalization."""
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps))


def rope_embed(seq_len: int, head_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Rotary Position Embeddings."""
    inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate half the hidden dims."""
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(q: np.ndarray, k: np.ndarray,
               cos: np.ndarray, sin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Position Embeddings to Q and K."""
    B, L, H, D = q.shape
    cos_exp = cos.reshape(1, L, 1, D)
    sin_exp = sin.reshape(1, L, 1, D)

    q_rot = (q * cos_exp) + (rotate_half(q) * sin_exp)
    k_rot = (k * cos_exp) + (rotate_half(k) * sin_exp)
    return q_rot, k_rot


class HRMNumPy:
    """Samsung Hierarchical Reasoning Model - NumPy implementation."""

    def __init__(self, weights_dir: str):
        print(f"Loading HRM weights from {weights_dir}")

        self.cfg = CONFIG
        self.weights: Dict[str, np.ndarray] = {}

        # Define all weight names
        weight_names = [
            'H_init', 'L_init',
            'embed_tokens.embedding_weight',
            'lm_head.weight',
        ]

        # Add layer weights for both H and L levels
        for level in ['H', 'L']:
            for layer in range(4):
                prefix = f'{level}_level.layers.{layer}'
                weight_names.extend([
                    f'{prefix}.self_attn.qkv_proj.weight',
                    f'{prefix}.self_attn.o_proj.weight',
                    f'{prefix}.mlp.gate_up_proj.weight',
                    f'{prefix}.mlp.down_proj.weight',
                ])

        # Load weights
        for name in weight_names:
            self.weights[name] = load_weight(weights_dir, name)

        # Precompute RoPE embeddings
        self.cos, self.sin = rope_embed(self.cfg['seq_len'], self.cfg['head_dim'])
        self.embed_scale = np.sqrt(float(self.cfg['hidden_size']))
        self.H_init = self.weights['H_init']
        self.L_init = self.weights['L_init']

        print(f"Loaded {len(self.weights)} weight tensors")

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Token embedding lookup with scaling."""
        W = self.weights['embed_tokens.embedding_weight']
        # Use advanced indexing for embedding lookup
        return self.embed_scale * W[x]

    def attention(self, x: np.ndarray, layer_prefix: str) -> np.ndarray:
        """Multi-head self-attention."""
        B, L, D = x.shape
        H = self.cfg['num_heads']
        Dh = self.cfg['head_dim']

        # Project to Q, K, V
        qkv_w = self.weights[f'{layer_prefix}.self_attn.qkv_proj.weight']
        qkv = x @ qkv_w.T

        # Split and reshape
        q = qkv[:, :, :D].reshape(B, L, H, Dh)
        k = qkv[:, :, D:2*D].reshape(B, L, H, Dh)
        v = qkv[:, :, 2*D:].reshape(B, L, H, Dh)

        # Apply RoPE
        q, k = apply_rope(q, k, self.cos, self.sin)

        # Transpose for attention: (B, H, L, Dh)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(float(Dh))
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = softmax(scores, axis=-1)

        # Apply attention to values
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, D)

        # Output projection
        o_w = self.weights[f'{layer_prefix}.self_attn.o_proj.weight']
        return out @ o_w.T

    def mlp(self, x: np.ndarray, layer_prefix: str) -> np.ndarray:
        """Feed-forward MLP with SiLU gating."""
        gate_up_w = self.weights[f'{layer_prefix}.mlp.gate_up_proj.weight']
        down_w = self.weights[f'{layer_prefix}.mlp.down_proj.weight']

        gate_up = x @ gate_up_w.T
        half_dim = gate_up.shape[-1] // 2
        gate = gate_up[:, :, :half_dim]
        up = gate_up[:, :, half_dim:]

        hidden = silu(gate) * up
        return hidden @ down_w.T

    def transformer_block(self, x: np.ndarray, layer_idx: int, level: str) -> np.ndarray:
        """Single transformer layer with pre-norm."""
        prefix = f'{level}_level.layers.{layer_idx}'

        # Attention with residual
        attn_out = self.attention(x, prefix)
        x = rms_norm(x + attn_out)

        # MLP with residual
        mlp_out = self.mlp(x, prefix)
        x = rms_norm(x + mlp_out)

        return x

    def forward_step(self, input_emb: np.ndarray,
                     z_h: np.ndarray, z_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single recurrent step of HRM."""
        # L-level processing
        l = z_l + z_h + input_emb
        for i in range(self.cfg['L_layers']):
            l = self.transformer_block(l, i, 'L')
        z_l = l

        # H-level processing
        h = z_h + z_l
        for i in range(self.cfg['H_layers']):
            h = self.transformer_block(h, i, 'H')
        z_h = h

        return z_h, z_l

    def forward(self, puzzle: np.ndarray, max_steps: int = 16) -> np.ndarray:
        """Full forward pass with recurrent reasoning."""
        B = puzzle.shape[0]
        L = self.cfg['seq_len']
        D = self.cfg['hidden_size']

        # Embed input
        input_emb = self.embed(puzzle)

        # Initialize hidden states
        z_h = np.tile(self.H_init.reshape(1, 1, D), (B, L, 1)).astype(np.float32)
        z_l = np.tile(self.L_init.reshape(1, 1, D), (B, L, 1)).astype(np.float32)

        # Recurrent reasoning steps
        for step in range(max_steps):
            z_h, z_l = self.forward_step(input_emb, z_h, z_l)
            if (step + 1) % 4 == 0:
                print(f"  Step {step + 1}/{max_steps}")

        # Final projection to vocabulary
        lm_head = self.weights['lm_head.weight']
        return z_h @ lm_head.T

    def solve(self, puzzle: List[int]) -> Tuple[List[int], float]:
        """Solve a Sudoku puzzle."""
        x = np.array([puzzle], dtype=np.int64)
        start = time.time()
        logits = self.forward(x)
        elapsed = time.time() - start
        preds = np.argmax(logits[0], axis=-1).tolist()
        return preds, elapsed


def print_grid(grid: List[int], title: str):
    """Pretty print a Sudoku grid."""
    print(f"\n{title}:")
    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("-" * 21)
        row = grid[i*9:(i+1)*9]
        cells = [str(x) if x > 0 else '.' for x in row]
        print(f" {' '.join(cells[:3])} | {' '.join(cells[3:6])} | {' '.join(cells[6:9])}")


def demo():
    print("=" * 60)
    print("Samsung HRM Sudoku - Python 3.7+ / NumPy")
    print("=" * 60)

    # Find weights directory - check command line first
    if len(sys.argv) > 1:
        weights_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(script_dir, 'weights_be')

        # Also check parent directory
        if not os.path.exists(weights_dir):
            weights_dir = os.path.join(os.path.dirname(script_dir), 'trm-sophiacord', 'weights_be')

    if not os.path.exists(weights_dir):
        print(f"ERROR: Weights not found at: {weights_dir}")
        print("Usage: python3 hrm_inference_py3.py [weights_dir]")
        return

    model = HRMNumPy(weights_dir)

    # Classic Sudoku puzzle
    puzzle = [
        5,3,0,0,7,0,0,0,0, 6,0,0,1,9,5,0,0,0, 0,9,8,0,0,0,0,6,0,
        8,0,0,0,6,0,0,0,3, 4,0,0,8,0,3,0,0,1, 7,0,0,0,2,0,0,0,6,
        0,6,0,0,0,0,2,8,0, 0,0,0,4,1,9,0,0,5, 0,0,0,0,8,0,0,7,9
    ]

    correct = [
        5,3,4,6,7,8,9,1,2, 6,7,2,1,9,5,3,4,8, 1,9,8,3,4,2,5,6,7,
        8,5,9,7,6,1,4,2,3, 4,2,6,8,5,3,7,9,1, 7,1,3,9,2,4,8,5,6,
        9,6,1,5,3,7,2,8,4, 2,8,7,4,1,9,6,3,5, 3,4,5,2,8,6,1,7,9
    ]

    print_grid(puzzle, "Input")

    print("\nRunning inference...")
    solution, elapsed = model.solve(puzzle)

    # Print solution with accuracy check
    print(f"\nSolution ({elapsed:.2f}s):")
    matches = 0
    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("-" * 21)
        row = solution[i*9:(i+1)*9]
        corr = correct[i*9:(i+1)*9]
        cells = []
        for j in range(9):
            if row[j] == corr[j]:
                cells.append(str(row[j]))
                matches += 1
            else:
                cells.append("X")
        print(f" {' '.join(cells[:3])} | {' '.join(cells[3:6])} | {' '.join(cells[6:9])}")

    print(f"\nAccuracy: {matches}/81 = {100.0*matches/81:.1f}%")


if __name__ == '__main__':
    demo()
