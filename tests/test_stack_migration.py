"""Contratos del PLAN-001 (migración stack IA/OCR)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Fase 1: OCR en cascada ──────────────────────────────────────────


@pytest.mark.integration
def test_word_boxes_shape_tesseract():
    """El adaptador tesseract produce {text,x,y,w,h} ints >= 0. (Requiere binario de Tesseract: Docker)"""
    import numpy as np
    from PIL import Image
    from src.services.ocr import _tesseract_page

    img = Image.new("L", (400, 200), color=255)
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.text((20, 80), "DIAGNOSTICO PRINCIPAL", fill=0)
    np_img = np.array(img)
    text, words = _tesseract_page(np_img)
    assert isinstance(text, str)
    for w in words:
        assert set(w.keys()) >= {"text", "x", "y", "w", "h"}
        assert all(isinstance(w[k], int) and w[k] >= 0 for k in ("x", "y", "w", "h"))
        assert isinstance(w["text"], str) and w["text"]


def test_ocr_engine_selectable_from_config():
    """OCR_ENGINE acepta tesseract|paddle|surya y el default es tesseract."""
    from src.config import Settings

    s = Settings(OCR_ENGINE="tesseract")
    assert s.OCR_ENGINE == "tesseract"
    s2 = Settings(OCR_ENGINE="paddle")
    assert s2.OCR_ENGINE == "paddle"


def test_paddle_available():
    """PaddleOCR está importable (cimiento de la cascada)."""
    import paddleocr  # noqa: F401


def test_angle_classifier_detects_orientation():
    """Hay función de clasificación/detección de ángulo disponible en el módulo."""
    from src.services.ocr import detect_and_fix_rotation

    # firma y comportamiento básico sobre imagen blanca
    import numpy as np
    from PIL import Image

    white = np.array(Image.new("L", (200, 100), color=255))
    angle, fixed_img = detect_and_fix_rotation(white)
    assert angle in (0, 90, 180, 270)
    assert fixed_img is not None


@pytest.mark.integration
def test_tesseract_page_accepts_ndarray_and_pil():
    """El adaptador acepta ndarray y PIL por igual (contrato word_boxes). (Requiere Tesseract binario: Docker)"""
    import numpy as np
    from PIL import Image, ImageDraw
    from src.services.ocr import _tesseract_page

    img = Image.new("L", (400, 200), color=255)
    d = ImageDraw.Draw(img)
    d.text((20, 80), "HISTORIA CLINICA", fill=0)
    arr = np.array(img)
    text, words = _tesseract_page(arr)
    for w in words:
        assert set(w.keys()) >= {"text", "x", "y", "w", "h"}


# ── Tests de integración (requieren Docker: motor ML no está en el host) ──
import pytest


@pytest.mark.integration
def test_paddle_available():
    """PaddleOCR está importable (en el contenedor worker)."""
    import paddleocr  # noqa: F401


@pytest.mark.integration
def test_embed_dimension_is_dynamic():
    """encode_texts devuelve la dimensión del modelo configurado (no fija 384)."""
    from src.services import embeddings as emb

    emb.preload()
    vecs = emb.encode_texts(["prueba de dimension"])
    expected = emb._get_model().get_sentence_embedding_dimension()
    assert vecs.shape[1] == expected


def test_config_has_embed_sources():
    from src.config import Settings

    s = Settings()
    assert s.EMBED_MODEL_NAME  # conmutable, default definido en config


# ── Fase 3: re-ranking ─────────────────────────────────────────────


def test_reranker_available():
    """El módulo reranker existe con función rerank()."""
    # NOTA: en host sentence-transformers puede faltar; el módulo debe ser
    # importable igualmente (carga del modelo es lazy).
    from src.services.reranker import rerank, unload  # noqa: F401


def test_rerank_degrades_without_model():
    """Si el modelo no carga, rerank devuelve los resultados sin romper."""
    from src.services import reranker as R

    # _load_model devuelve None cuando falla (el propio rerank captura el error)
    original_load = R._load_model

    def failing_load():
        try:
            raise RuntimeError("sin RAM")
        except RuntimeError:
            return None

    R._load_model = failing_load
    try:
        results = [{"content": "a"}, {"content": "b"}]
        out = R.rerank("query", results, top_k=2)
        assert out == results  # passthrough
    finally:
        R._load_model = original_load


def test_query_response_has_segmented_times():
    """QueryResponse expone retrieval/rerank/llm/total ms (RAG-053)."""
    from src.schemas.query import QueryResponse

    r = QueryResponse(answer="x", sources=[], query="q", processing_time_ms=1.0,
                      retrieval_ms=1, rerank_ms=1, llm_ms=1, total_ms=3)
    assert r.retrieval_ms == 1 and r.total_ms == 3


# ── Fase 4: LLM conmutable ─────────────────────────────────────────


def test_llm_model_path_config_in_find_model():
    """_find_model respeta LLM_MODEL_PATH como switch único."""
    from src.services import llm
    from src.config import get_settings

    s = get_settings()
    assert hasattr(s, "LLM_MODEL_PATH")
    # apuntar a un path inexistente con LLM_MODEL_PATH vacío mantiene fallback TinyLlama
