#!/usr/bin/env python3
"""Download GGUF model for local LLM."""
import os
import sys
import urllib.request
import zipfile

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if model already exists
for f in os.listdir(MODEL_DIR):
    if f.endswith(".gguf"):
        print(f"Model already exists: {f}")
        sys.exit(0)

print("No GGUF model found in models/")
print("Download a GGUF model (e.g. Qwen3-3B-Q4_K_M) from Hugging Face")
print(f"Place it in: {os.path.abspath(MODEL_DIR)}")
print("")
print("Example:")
print("  pip install huggingface-hub")
print("  huggingface-cli download Qwen/Qwen3-3B-GGUF qwen3-3b-q4_k_m.gguf --local-dir models/")
