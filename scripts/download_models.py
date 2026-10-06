#!/usr/bin/env python3
"""Descarga el modelo LLM local (Qwen2.5-1.5B-Instruct GGUF) desde HuggingFace.

Idempotente: si ya hay un `.gguf` en `models/`, no descarga nada.

- **Este script** baja el LLM (Qwen2.5-1.5B-Instruct Q4_K_M, ~1 GB).
- Los **embeddings** (MiniLM) y el **reranker** (bge-reranker-base) se descargan
  solos la primera vez que arranca la API (requiere internet una vez; quedan en
  `hf_cache/`).
- Los modelos de **OCR** (RapidOCR PP-OCRv5) van dentro de la imagen Docker
  (se pre-descargan en el build).

Uso:
    pip install huggingface-hub
    python scripts/download_models.py
"""
from __future__ import annotations

import os
import sys

REPO_ID = os.environ.get("LLM_HF_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
FILENAME = os.environ.get("LLM_HF_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)

    target = os.path.join(MODEL_DIR, FILENAME)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"Ya existe el modelo: {target}")
        return 0

    existing = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
    if existing:
        print(f"Ya hay un modelo GGUF en {MODEL_DIR}: {existing[0]} — no se descarga nada.")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Falta 'huggingface-hub'. Instálalo con:\n    pip install huggingface-hub",
            file=sys.stderr,
        )
        return 1

    print(f"Descargando {REPO_ID}/{FILENAME} (~1 GB) en {MODEL_DIR} ...")
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODEL_DIR)
    print(f"Listo: {path}")
    print("Siguiente paso: docker compose up --build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
