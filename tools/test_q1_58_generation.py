#!/usr/bin/env python3
"""
Test Q1.58 model generation quality.
Performs simple forward pass to see if outputs are coherent.
"""

import struct
import numpy as np
import sys

# Constants
QK_Q1_58 = 256
BLOCK_SIZE_Q1_58 = 68


def read_gguf_header(path):
    """Read GGUF header and return tensor info + metadata."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Invalid magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        # Read metadata
        metadata = {}
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_value(f, vtype)
            metadata[key] = value

        # Read tensor info
        tensors = {}
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {
                'dims': dims,
                'type': ttype,
                'offset': offset
            }

        # Get data start
        current_pos = f.tell()
        padding = (32 - (current_pos % 32)) % 32
        f.read(padding)
        data_start = f.tell()

        return metadata, tensors, data_start


def read_value(f, vtype):
    """Read a GGUF value."""
    if vtype in (0, 1, 7):
        return struct.unpack('B', f.read(1))[0]
    elif vtype in (2, 3):
        return struct.unpack('<H', f.read(2))[0]
    elif vtype in (4, 5):
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == 6:
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == 8:
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
    elif vtype in (10, 11):
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 12:
        return struct.unpack('<d', f.read(8))[0]
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    return None


def unpack_ternary_block(packed_bytes):
    """Unpack 256 ternary values from 64 bytes."""
    decode = [-1, 0, 1, 0]
    weights = []
    for byte in packed_bytes:
        weights.extend([
            decode[(byte >> 0) & 3],
            decode[(byte >> 2) & 3],
            decode[(byte >> 4) & 3],
            decode[(byte >> 6) & 3]
        ])
    return np.array(weights, dtype=np.float32)


def load_q1_58_tensor(f, data_start, tensor_info):
    """Load a Q1.58 tensor."""
    f.seek(data_start + tensor_info['offset'])

    n_elements = 1
    for d in tensor_info['dims']:
        n_elements *= d

    n_blocks = n_elements // QK_Q1_58

    weights = []
    for _ in range(n_blocks):
        block_data = f.read(BLOCK_SIZE_Q1_58)
        packed = block_data[0:64]
        scale = np.frombuffer(block_data[64:66], dtype=np.float16)[0]

        block_weights = unpack_ternary_block(packed) * float(scale)
        weights.extend(block_weights)

    return np.array(weights[:n_elements], dtype=np.float32).reshape(tensor_info['dims'])


def softmax(x):
    """Compute softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_q1_58_generation.py model.gguf")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Testing Q1.58 model: {model_path}")
    print("=" * 60)

    metadata, tensors, data_start = read_gguf_header(model_path)

    # Get model info
    vocab_size = metadata.get('llama.vocab_size', 32000)
    hidden_size = metadata.get('llama.embedding_length', 2048)

    print(f"Vocab size: {vocab_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Tensors: {len(tensors)}")

    # Load token embeddings
    print("\nLoading token embeddings...")
    with open(model_path, 'rb') as f:
        if 'token_embd.weight' in tensors:
            emb_info = tensors['token_embd.weight']
            print(f"  Shape: {emb_info['dims']}")

            if emb_info['type'] == 20:  # Q1.58
                embeddings = load_q1_58_tensor(f, data_start, emb_info)
                print(f"  Loaded: {embeddings.shape}")
            else:
                print(f"  Type {emb_info['type']} - not Q1.58, skipping full load")
                embeddings = None
        else:
            print("  token_embd.weight not found!")
            embeddings = None

    if embeddings is None:
        print("\nCannot test generation without embeddings")
        sys.exit(1)

    # Test: Get embedding for a few tokens and check they're different
    print("\n--- Embedding Sanity Check ---")
    test_tokens = [1, 100, 1000, 5000, 10000]  # Various token IDs

    # Embeddings may be [hidden, vocab] or [vocab, hidden]
    # Transpose if needed
    if embeddings.shape[0] == hidden_size and embeddings.shape[1] == vocab_size:
        embeddings = embeddings.T  # Now [vocab, hidden]
        print(f"  Transposed to: {embeddings.shape}")

    for tok in test_tokens:
        if tok < embeddings.shape[0]:
            emb = embeddings[tok]
            mean = np.mean(emb)
            std = np.std(emb)
            norm = np.linalg.norm(emb)
            nonzero = np.count_nonzero(emb)
            print(f"  Token {tok:5d}: mean={mean:+.4f} std={std:.4f} norm={norm:.2f} nonzero={nonzero}/{hidden_size}")

    # Check if embeddings are distinct
    print("\n--- Embedding Distinctness ---")
    if len(test_tokens) >= 2:
        e1 = embeddings[test_tokens[0]]
        e2 = embeddings[test_tokens[1]]
        cosine_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        print(f"  Cosine similarity (token {test_tokens[0]} vs {test_tokens[1]}): {cosine_sim:.4f}")

        if abs(cosine_sim) > 0.99:
            print("  WARNING: Embeddings nearly identical! Model may be broken.")
        elif abs(cosine_sim) < 0.5:
            print("  OK: Embeddings are distinct")
        else:
            print("  Embeddings have moderate similarity")

    # Load output projection and test logit generation
    print("\n--- Output Logits Test ---")
    with open(model_path, 'rb') as f:
        if 'output.weight' in tensors:
            out_info = tensors['output.weight']
            print(f"  output.weight shape: {out_info['dims']}")

            if out_info['type'] == 20:  # Q1.58
                output_weight = load_q1_58_tensor(f, data_start, out_info)
                print(f"  Loaded: {output_weight.shape}")

                # Transpose if needed to [vocab_size, hidden_size]
                if output_weight.shape[0] == hidden_size and output_weight.shape[1] == vocab_size:
                    output_weight = output_weight.T
                    print(f"  Transposed to: {output_weight.shape}")

                # Simulate: hidden state → logits
                # Use embedding of token 1 as "hidden state"
                hidden = embeddings[1]  # Shape: [hidden_size]

                # Compute logits: output_weight @ hidden
                # output_weight is [vocab_size, hidden_size], hidden is [hidden_size]
                logits = output_weight @ hidden  # Should give [vocab_size]

                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
                print(f"  Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")

                # Check if logits look reasonable
                probs = softmax(logits)
                top_k = 10
                top_indices = np.argsort(probs)[-top_k:][::-1]

                print(f"\n  Top {top_k} token predictions:")
                for i, idx in enumerate(top_indices):
                    print(f"    {i+1}. Token {idx}: prob={probs[idx]:.4f} logit={logits[idx]:.2f}")

                # Check entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(vocab_size)
                print(f"\n  Entropy: {entropy:.2f} / {max_entropy:.2f} ({100*entropy/max_entropy:.1f}%)")

                if entropy < max_entropy * 0.3:
                    print("  OK: Model has clear preferences (low entropy)")
                elif entropy > max_entropy * 0.9:
                    print("  WARNING: Near-uniform distribution (model may be broken)")
                else:
                    print("  Model has moderate confidence")
            else:
                print(f"  Type {out_info['type']} - not Q1.58")
        else:
            print("  output.weight not found!")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
