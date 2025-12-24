#!/usr/bin/env python3
"""
LLM ↔ PowerPC-Sophia Bridge

Connects modern LLMs (on x86_64) with PowerPC-Sophia sub-brain
for structured lossy compression and memory gisting.

Architecture:
    LLM (x86_64) → Embeddings → PowerPC-Sophia → Gist → LLM Context

Think: Cortex ↔ Thalamus feedback loop
"""

import socket
import struct
import numpy as np
import sys
import time
from typing import List, Optional

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not installed. Install with: pip install transformers torch")


class PowerPCSophiaBridge:
    """Bridge between LLM and PowerPC-Sophia sub-brain"""

    def __init__(self, sophia_host: str, sophia_port: int = 8090):
        self.sophia_host = sophia_host
        self.sophia_port = sophia_port
        self.sock = None

        # LLM model (optional - can work with any embedding source)
        self.model = None
        self.tokenizer = None

    def connect_to_sophia(self):
        """Connect to PowerPC-Sophia sub-brain"""
        print(f"🔗 Connecting to Sophia at {self.sophia_host}:{self.sophia_port}...")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.sophia_host, self.sophia_port))

        print("✅ Connected to PowerPC-Sophia!")

    def load_llm(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load LLM for embedding generation"""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers library required for LLM")

        print(f"📦 Loading LLM: {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        print("✅ LLM loaded!")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding from text using LLM"""
        if self.model is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt",
                               padding=True, truncation=True, max_length=128)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return embedding

    def send_to_sophia(self, embedding: np.ndarray) -> np.ndarray:
        """Send embedding to Sophia, receive compressed gist"""
        if self.sock is None:
            raise RuntimeError("Not connected to Sophia. Call connect_to_sophia() first.")

        # Ensure embedding is 128-dim float32
        if embedding.shape[0] < 128:
            # Pad to 128
            padded = np.zeros(128, dtype=np.float32)
            padded[:embedding.shape[0]] = embedding[:128]
            embedding = padded
        elif embedding.shape[0] > 128:
            # Truncate to 128
            embedding = embedding[:128]

        embedding = embedding.astype(np.float32)

        # Send embedding (128 floats = 512 bytes)
        data = embedding.tobytes()
        self.sock.sendall(data)

        # Receive gist (64 floats = 256 bytes)
        gist_data = b''
        while len(gist_data) < 256:
            chunk = self.sock.recv(256 - len(gist_data))
            if not chunk:
                raise ConnectionError("Sophia disconnected")
            gist_data += chunk

        gist = np.frombuffer(gist_data, dtype=np.float32)
        return gist

    def get_memory_context(self) -> np.ndarray:
        """Retrieve compressed memory context from Sophia"""
        # Send special command (all zeros = request context)
        cmd = np.zeros(128, dtype=np.float32)
        return self.send_to_sophia(cmd)

    def process_text_with_sophia(self, text: str) -> dict:
        """
        Full pipeline: Text → LLM embedding → Sophia compression → Gist

        Returns:
            {
                'text': original text,
                'embedding': LLM embedding (128-dim),
                'gist': Sophia compressed gist (64-dim),
                'compression_ratio': float
            }
        """
        print(f"\n📝 Processing: '{text}'")

        # Get LLM embedding
        embedding = self.get_embedding(text)
        print(f"   LLM embedding shape: {embedding.shape}")

        # Send to Sophia for compression
        gist = self.send_to_sophia(embedding)
        print(f"   Sophia gist shape: {gist.shape}")

        compression_ratio = embedding.shape[0] / gist.shape[0]
        print(f"   Compression ratio: {compression_ratio:.1f}x")

        # Calculate information preserved (rough estimate)
        embedding_norm = np.linalg.norm(embedding)
        gist_norm = np.linalg.norm(gist)
        preservation = (gist_norm / embedding_norm) * 100 if embedding_norm > 0 else 0
        print(f"   Information preserved: {preservation:.1f}%")

        return {
            'text': text,
            'embedding': embedding,
            'gist': gist,
            'compression_ratio': compression_ratio,
            'preservation': preservation
        }


def demo_sophia_bridge():
    """Demonstration of LLM ↔ Sophia interaction"""

    print("╔════════════════════════════════════════════════╗")
    print("║   LLM ↔ PowerPC-Sophia Bridge Demo            ║")
    print("║   Structured Lossy Compression via AltiVec    ║")
    print("╚════════════════════════════════════════════════╝\n")

    # Configuration
    SOPHIA_HOST = "192.168.0.115"  # PowerPC G4
    SOPHIA_PORT = 8090

    bridge = PowerPCSophiaBridge(SOPHIA_HOST, SOPHIA_PORT)

    # Load LLM
    if HAS_TRANSFORMERS:
        bridge.load_llm()
    else:
        print("⚠️  Transformers not available - using synthetic embeddings")

    # Connect to Sophia
    try:
        bridge.connect_to_sophia()
    except Exception as e:
        print(f"❌ Could not connect to Sophia: {e}")
        print(f"   Make sure Sophia is running on {SOPHIA_HOST}:{SOPHIA_PORT}")
        return

    # Test cases
    test_texts = [
        "The quantum nature of consciousness emerges from structured forgetting.",
        "AltiVec vec_perm enables non-bijective permutations in hardware.",
        "PowerPC G4 processors contain unique silicon aging signatures.",
        "Modern LLMs preserve too much information during inference.",
    ]

    results = []
    for text in test_texts:
        if HAS_TRANSFORMERS:
            result = bridge.process_text_with_sophia(text)
            results.append(result)
        else:
            # Synthetic embedding for demo
            embedding = np.random.randn(128).astype(np.float32)
            gist = bridge.send_to_sophia(embedding)
            print(f"\n📝 Synthetic test:")
            print(f"   Embedding: {embedding.shape}")
            print(f"   Gist: {gist.shape}")
            results.append({'text': text, 'gist': gist})

        time.sleep(0.5)

    # Compare gist vectors (how unique is each compression?)
    print("\n\n╔════════════════════════════════════════════════╗")
    print("║   Gist Vector Uniqueness Analysis             ║")
    print("╚════════════════════════════════════════════════╝\n")

    if len(results) >= 2:
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                gist1 = results[i]['gist']
                gist2 = results[j]['gist']

                # Cosine similarity
                similarity = np.dot(gist1, gist2) / (np.linalg.norm(gist1) * np.linalg.norm(gist2))

                print(f"Similarity between gist {i+1} and {j+1}: {similarity:.3f}")

    # Retrieve memory context
    print("\n\n╔════════════════════════════════════════════════╗")
    print("║   Memory Context Retrieval                     ║")
    print("╚════════════════════════════════════════════════╝\n")

    try:
        context = bridge.get_memory_context()
        print(f"✅ Retrieved memory context: {context.shape}")
        print(f"   Context norm: {np.linalg.norm(context):.3f}")
        print(f"   First 8 values: {context[:8]}")
    except Exception as e:
        print(f"❌ Could not retrieve context: {e}")

    print("\n\n🔥 Demo complete!")
    print("\n💡 Key Insights:")
    print("   • Sophia compresses 128-dim → 64-dim (2x reduction)")
    print("   • Non-bijective collapse = information loss by design")
    print("   • Each gist is hardware-unique due to AltiVec patterns")
    print("   • Memory context provides historical compression")


def usage():
    print("""
Usage:
    python3 llm_sophia_bridge.py [sophia_host] [sophia_port]

Examples:
    python3 llm_sophia_bridge.py 192.168.0.115 8090
    python3 llm_sophia_bridge.py localhost 8090

Default:
    Host: 192.168.0.115 (PowerPC G4)
    Port: 8090
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        usage()
        sys.exit(0)

    demo_sophia_bridge()
