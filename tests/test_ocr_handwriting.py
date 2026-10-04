"""PLAN-009 Fase B/C — cascada OCR `auto` con motor ML (RapidOCR) + score por página.

Contratos:
- `OCR_ENGINE=auto`: Tesseract primero (rápido, impreso). Si la confianza de la
  página es baja (o no hay texto) → re-OCR con RapidOCR (PP-OCRv5 latin).
- `OCR_ENGINE=tesseract` sigue siendo rollback puro (no toca RapidOCR).
- Si RapidOCR no está disponible/falla, se cae al texto de Tesseract sin romper.
- Cada página expone `ocr_score` (0-100) y `ocr_engine`; los chunks derivados
  heredan la peor página OCR y se marcan `low_confidence` si cae bajo el umbral.
"""
from __future__ import annotations

from PIL import Image

from src.services import ocr


def _set_engine(monkeypatch, engine: str, min_score: int | None = None) -> None:
    from src.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "OCR_ENGINE", engine, raising=False)
    if min_score is not None:
        monkeypatch.setattr(s, "OCR_AUTO_MIN_SCORE", min_score, raising=False)


def _img():
    return Image.new("RGB", (120, 60), "white")


def test_auto_cascade_usa_rapidocr_si_tesseract_da_baja_confianza(monkeypatch):
    _set_engine(monkeypatch, "auto")
    monkeypatch.setattr(
        ocr, "_tesseract_page_scored", lambda img: ("basura m4nusc", [{"text": "x"}], 12.0)
    )
    called = {"n": 0}

    def fake_rapid(img):
        called["n"] += 1
        return ("HISTORIA CLINICA PERALTA", [{"text": "HISTORIA"}], 88.0)

    monkeypatch.setattr(ocr, "_rapidocr_page", fake_rapid)

    text, words, meta = ocr._ocr_page_dispatch(_img())

    assert called["n"] == 1
    assert meta["ocr_engine"] == "rapidocr"
    assert meta["ocr_score"] == 88.0
    assert "HISTORIA" in text


def test_auto_cascade_no_usa_rapidocr_si_tesseract_es_buena(monkeypatch):
    _set_engine(monkeypatch, "auto")
    monkeypatch.setattr(
        ocr, "_tesseract_page_scored", lambda img: ("texto impreso", [{"text": "texto"}], 92.0)
    )
    called = {"n": 0}
    monkeypatch.setattr(
        ocr,
        "_rapidocr_page",
        lambda img: (called.__setitem__("n", called["n"] + 1) or ("x", [], 99.0)),
    )

    text, words, meta = ocr._ocr_page_dispatch(_img())

    assert called["n"] == 0
    assert meta["ocr_engine"] == "tesseract"
    assert text == "texto impreso"


def test_auto_cascade_cae_a_tesseract_si_rapidocr_falla(monkeypatch):
    _set_engine(monkeypatch, "auto")
    monkeypatch.setattr(
        ocr, "_tesseract_page_scored", lambda img: ("parcial", [{"text": "p"}], 20.0)
    )

    def boom(img):
        raise RuntimeError("sin modelos")

    monkeypatch.setattr(ocr, "_rapidocr_page", boom)

    text, words, meta = ocr._ocr_page_dispatch(_img())

    assert meta["ocr_engine"] == "tesseract"
    assert text == "parcial"


def test_auto_cascade_prefiere_rapidocr_si_tesseract_vacio(monkeypatch):
    _set_engine(monkeypatch, "auto")
    monkeypatch.setattr(ocr, "_tesseract_page_scored", lambda img: ("", [], 0.0))
    monkeypatch.setattr(
        ocr, "_rapidocr_page", lambda img: ("recuperado", [{"text": "recuperado"}], 70.0)
    )

    text, words, meta = ocr._ocr_page_dispatch(_img())

    assert meta["ocr_engine"] == "rapidocr"
    assert text == "recuperado"


def test_engine_tesseract_no_llama_rapidocr(monkeypatch):
    _set_engine(monkeypatch, "tesseract")
    monkeypatch.setattr(
        ocr, "_tesseract_page_scored", lambda img: ("solo tesseract", [], 80.0)
    )
    called = {"n": 0}
    monkeypatch.setattr(
        ocr,
        "_rapidocr_page",
        lambda img: (called.__setitem__("n", called["n"] + 1) or ("", [], 0.0)),
    )

    text, words, meta = ocr._ocr_page_dispatch(_img())

    assert called["n"] == 0
    assert meta["ocr_engine"] == "tesseract"
    assert text == "solo tesseract"


def test_tesseract_page_sigue_devolviendo_dos_valores(monkeypatch):
    """Contrato heredado: `_tesseract_page` (2-tuple) no debe romperse."""
    monkeypatch.setattr(
        ocr, "_tesseract_page_scored", lambda img: ("hola", [{"text": "hola"}], 77.0)
    )
    out = ocr._tesseract_page(_img())
    assert out == ("hola", [{"text": "hola"}])


def test_aggregate_ocr_confidence_toma_la_peor_pagina():
    page_meta = {
        1: {"ocr_score": 90.0, "ocr_engine": "tesseract"},
        2: {"ocr_score": 30.0, "ocr_engine": "rapidocr"},
    }
    score, engine, low = ocr.aggregate_ocr_confidence([1, 2], page_meta, threshold=45)
    assert score == 30.0
    assert engine == "rapidocr"
    assert low is True


def test_aggregate_ocr_confidence_sin_paginas_ocr():
    score, engine, low = ocr.aggregate_ocr_confidence([1, 2], {}, threshold=45)
    assert score is None
    assert engine is None
    assert low is False


def test_remap_boxes_180_devuelve_el_box_a_la_orientacion_original():
    """Un box leído 'abajo' en la imagen girada 180 vuelve arriba (página real)."""
    words = [{"text": "HC-2025", "x": 452, "y": 1907, "w": 362, "h": 39}]
    out = ocr._remap_boxes_to_original(words, 180, 1262, 1947)
    assert out[0]["y"] <= 5
    assert abs(out[0]["x"] - 447) <= 2
    assert out[0]["w"] == 362


def test_remap_boxes_angle_cero_no_cambia():
    words = [{"text": "HC", "x": 10, "y": 20, "w": 30, "h": 5}]
    assert ocr._remap_boxes_to_original(words, 0, 100, 200) == words


def test_fallback_180_exige_score_absoluto(monkeypatch):
    """Páginas con poco texto (score bajo y ruidoso) NO deben invertirse."""
    import sys
    import types

    from PIL import Image

    def _raise(*_a, **_k):
        raise RuntimeError("sin osd")

    fake = types.SimpleNamespace()
    fake.image_to_osd = _raise
    fake.Output = types.SimpleNamespace(DICT="DICT")
    monkeypatch.setitem(sys.modules, "pytesseract", fake)

    scores = iter([30.0, 41.0])  # 0° bajo, 180° algo mejor pero aún bajo
    monkeypatch.setattr(ocr, "_quick_ocr_score", lambda _img: next(scores))

    angle, _fixed = ocr.detect_and_fix_rotation(Image.new("RGB", (200, 100), "white"))
    assert angle == 0
