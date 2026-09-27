from __future__ import annotations

import gc
import logging
import os
from typing import Optional

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

_model = None
settings = get_settings()
_threads_configured = False


def _configure_threads() -> None:
    """Use all CPUs for torch encode (measured default was 2 of 4).

    Do NOT set OMP_NUM_THREADS/MKL in os.environ: pytesseract passes the live
    environ to every tesseract child, so a global OMP=4 becomes N*4 OpenMP
    threads when OCR runs in parallel (load thrash, multi-minute single pages).
    torch.set_num_threads controls intra-op parallelism for encode only.
    """
    global _threads_configured
    if _threads_configured:
        return
    cpu = os.cpu_count() or 4
    # Prefer blocking over spin-wait (OMP ACTIVE burns CPU at idle)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    try:
        import torch
        torch.set_num_threads(cpu)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already initialized
        logger.info(f"torch threads={torch.get_num_threads()} (cpu={cpu})")
    except Exception as e:
        logger.warning(f"Could not set torch threads: {e}")
    _threads_configured = True


def _get_model():
    global _model
    if _model is None:
        _configure_threads()
        from sentence_transformers import SentenceTransformer
        model_name = settings.EMBED_MODEL_NAME
        logger.info(f"Loading embeddings: {model_name}")
        import time
        t0 = time.time()
        _model = SentenceTransformer(
            model_name,
            device="cpu",
            trust_remote_code=True,
        )
        logger.info(f"Embeddings loaded in {time.time()-t0:.1f}s")
    return _model


def preload():
    _get_model()
    logger.info("Embeddings preloaded and ready")


def encode_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.array([])

    _configure_threads()
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings if isinstance(embeddings, np.ndarray) else np.asarray(embeddings)


def encode_query(query: str) -> list[float]:
    _configure_threads()
    model = _get_model()
    emb = model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return emb.tolist() if hasattr(emb, "tolist") else list(emb)


def unload_embeddings():
    global _model
    if _model is not None:
        del _model
        _model = None
        gc.collect()
        logger.info("Embeddings model unloaded")
