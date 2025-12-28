#!/usr/bin/env python3
"""
Pure Python LLaMA Tokenizer

A minimal SentencePiece BPE tokenizer implementation that can load
LLaMA tokenizer.model files without requiring the C++ library.

Designed for portability on PowerPC and other systems where
installing SentencePiece is difficult.

Usage:
    from llama_tokenizer import LlamaTokenizer
    tok = LlamaTokenizer("tokenizer.model")
    tokens = tok.encode("Hello, world!")
    text = tok.decode(tokens)

Author: ppc-tiger-tools project
License: MIT
December 2025
"""

import struct
import os
from pathlib import Path


class LlamaTokenizer:
    """Pure Python LLaMA tokenizer (SentencePiece BPE)"""

    def __init__(self, model_path=None):
        self.vocab = {}           # token string -> id
        self.vocab_inv = {}       # id -> token string
        self.scores = {}          # id -> score (for BPE merging)
        self.special_tokens = {}

        # Default special tokens
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 0
        self.pad_id = -1

        if model_path:
            self.load(model_path)

    def load(self, model_path):
        """Load tokenizer from SentencePiece .model file or vocab file"""
        model_path = Path(model_path)

        if model_path.suffix == '.model':
            self._load_sentencepiece(model_path)
        elif model_path.suffix == '.json':
            self._load_json(model_path)
        elif model_path.suffix in ('.txt', '.vocab'):
            self._load_vocab_txt(model_path)
        else:
            # Try to detect format
            with open(model_path, 'rb') as f:
                magic = f.read(4)
                if magic[:2] == b'\x0a\x02':  # Protobuf
                    self._load_sentencepiece(model_path)
                else:
                    self._load_vocab_txt(model_path)

    def _load_sentencepiece(self, path):
        """Load SentencePiece .model file (simplified protobuf parsing)"""
        # SentencePiece uses protobuf, but we can parse it manually
        # The format is:
        # - Each vocab entry is a message with piece (string) and score (float)

        with open(path, 'rb') as f:
            data = f.read()

        # Parse protobuf manually (simplified)
        pos = 0
        token_id = 0

        while pos < len(data):
            # Read field header (varint)
            field_num, wire_type, pos = self._read_varint_field(data, pos)

            if field_num == 1 and wire_type == 2:  # pieces message
                # Length-delimited message
                msg_len, pos = self._read_varint(data, pos)
                msg_end = pos + msg_len

                piece = ""
                score = 0.0
                piece_type = 1  # NORMAL

                while pos < msg_end:
                    sub_field, sub_wire, pos = self._read_varint_field(data, pos)

                    if sub_field == 1 and sub_wire == 2:  # piece string
                        str_len, pos = self._read_varint(data, pos)
                        piece = data[pos:pos+str_len].decode('utf-8', errors='replace')
                        pos += str_len
                    elif sub_field == 2 and sub_wire == 5:  # score float
                        score = struct.unpack('<f', data[pos:pos+4])[0]
                        pos += 4
                    elif sub_field == 3 and sub_wire == 0:  # type
                        piece_type, pos = self._read_varint(data, pos)
                    else:
                        # Skip unknown field
                        pos = self._skip_field(data, pos, sub_wire)

                # Store token
                self.vocab[piece] = token_id
                self.vocab_inv[token_id] = piece
                self.scores[token_id] = score

                # Mark special tokens
                if piece_type == 2:  # UNKNOWN
                    self.unk_id = token_id
                elif piece_type == 3:  # CONTROL (BOS/EOS)
                    if '<s>' in piece or 'bos' in piece.lower():
                        self.bos_id = token_id
                    elif '</s>' in piece or 'eos' in piece.lower():
                        self.eos_id = token_id
                elif piece_type == 4:  # USER_DEFINED
                    self.special_tokens[piece] = token_id

                token_id += 1
                pos = msg_end

            elif field_num == 2 and wire_type == 2:  # trainer_spec
                msg_len, pos = self._read_varint(data, pos)
                pos += msg_len  # Skip
            elif field_num == 3 and wire_type == 2:  # normalizer_spec
                msg_len, pos = self._read_varint(data, pos)
                pos += msg_len  # Skip
            else:
                pos = self._skip_field(data, pos, wire_type)

        print(f"Loaded {len(self.vocab)} tokens from SentencePiece model")

    def _read_varint(self, data, pos):
        """Read a varint from data at pos, return (value, new_pos)"""
        result = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        return result, pos

    def _read_varint_field(self, data, pos):
        """Read field number and wire type from varint"""
        tag, pos = self._read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        return field_num, wire_type, pos

    def _skip_field(self, data, pos, wire_type):
        """Skip a field based on wire type"""
        if wire_type == 0:  # Varint
            _, pos = self._read_varint(data, pos)
        elif wire_type == 1:  # 64-bit
            pos += 8
        elif wire_type == 2:  # Length-delimited
            length, pos = self._read_varint(data, pos)
            pos += length
        elif wire_type == 5:  # 32-bit
            pos += 4
        return pos

    def _load_vocab_txt(self, path):
        """Load vocabulary from text file (one token per line)"""
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if '\t' in line:
                    token, score = line.rsplit('\t', 1)
                    score = float(score)
                elif ' ' in line:
                    parts = line.rsplit(' ', 1)
                    if len(parts) == 2:
                        token, score = parts
                        try:
                            score = float(score)
                        except ValueError:
                            token = line
                            score = 0.0
                    else:
                        token = line
                        score = 0.0
                else:
                    token = line
                    score = 0.0

                self.vocab[token] = i
                self.vocab_inv[i] = token
                self.scores[i] = score

        print(f"Loaded {len(self.vocab)} tokens from vocab file")

    def _load_json(self, path):
        """Load vocabulary from JSON file"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'model' in data and 'vocab' in data['model']:
            # HuggingFace format
            vocab_list = data['model']['vocab']
            for i, (token, score) in enumerate(vocab_list):
                self.vocab[token] = i
                self.vocab_inv[i] = token
                self.scores[i] = score
        elif 'vocab' in data:
            # Simple vocab format
            for token, idx in data['vocab'].items():
                self.vocab[token] = idx
                self.vocab_inv[idx] = token
                self.scores[idx] = 0.0

        print(f"Loaded {len(self.vocab)} tokens from JSON")

    def encode(self, text, add_bos=True, add_eos=False):
        """Encode text to token IDs using BPE"""
        if not self.vocab:
            raise ValueError("Tokenizer not loaded")

        # Handle special tokens first
        for special, tok_id in self.special_tokens.items():
            if special in text:
                # This is a simplified approach - real impl would be more careful
                pass

        # SentencePiece preprocessing: replace spaces with ▁ (U+2581)
        # and prepend ▁ to start
        text = '▁' + text.replace(' ', '▁')

        # Character-level tokenization first
        tokens = list(text)

        # Apply BPE merges
        while len(tokens) > 1:
            # Find the best pair to merge (lowest score = highest priority)
            best_pair = None
            best_idx = -1
            best_score = float('inf')

            for i in range(len(tokens) - 1):
                pair = tokens[i] + tokens[i + 1]
                if pair in self.vocab:
                    score = self.scores.get(self.vocab[pair], 0)
                    if score < best_score:
                        best_score = score
                        best_pair = pair
                        best_idx = i

            if best_pair is None:
                break

            # Merge the best pair
            tokens = tokens[:best_idx] + [best_pair] + tokens[best_idx + 2:]

        # Convert to IDs
        ids = []
        if add_bos:
            ids.append(self.bos_id)

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Try character fallback or use UNK
                for char in token:
                    if char in self.vocab:
                        ids.append(self.vocab[char])
                    else:
                        # Byte fallback
                        for byte in char.encode('utf-8'):
                            byte_token = f"<0x{byte:02X}>"
                            if byte_token in self.vocab:
                                ids.append(self.vocab[byte_token])
                            else:
                                ids.append(self.unk_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, skip_special=True):
        """Decode token IDs to text"""
        if not self.vocab_inv:
            raise ValueError("Tokenizer not loaded")

        pieces = []
        for tok_id in ids:
            if skip_special and tok_id in (self.bos_id, self.eos_id, self.pad_id):
                continue

            if tok_id in self.vocab_inv:
                piece = self.vocab_inv[tok_id]

                # Handle byte tokens
                if piece.startswith('<0x') and piece.endswith('>'):
                    try:
                        byte_val = int(piece[3:-1], 16)
                        piece = bytes([byte_val]).decode('utf-8', errors='replace')
                    except (ValueError, UnicodeDecodeError):
                        pass

                pieces.append(piece)
            else:
                pieces.append(f'[{tok_id}]')

        # Join and convert ▁ back to spaces
        text = ''.join(pieces)
        text = text.replace('▁', ' ')

        # Remove leading space (artifact of preprocessing)
        if text.startswith(' '):
            text = text[1:]

        return text

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.vocab)


def main():
    """Test the tokenizer"""
    import sys

    if len(sys.argv) < 2:
        print("LLaMA Tokenizer - Pure Python Implementation")
        print()
        print("Usage: python3 llama_tokenizer.py <tokenizer.model> [text]")
        print()
        print("Loads LLaMA/SentencePiece tokenizer and encodes/decodes text.")
        return

    model_path = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else "Hello, world!"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"Loading tokenizer: {model_path}")
    tok = LlamaTokenizer(model_path)
    print(f"Vocabulary size: {tok.vocab_size}")

    print(f"\nText: {text}")
    tokens = tok.encode(text)
    print(f"Tokens: {tokens}")

    decoded = tok.decode(tokens)
    print(f"Decoded: {decoded}")

    # Show some vocab entries
    print("\nFirst 10 tokens:")
    for i in range(min(10, len(tok.vocab_inv))):
        print(f"  {i}: {repr(tok.vocab_inv[i])}")


if __name__ == "__main__":
    main()
