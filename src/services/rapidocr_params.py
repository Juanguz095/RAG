"""Selección de modelos RapidOCR (PLAN-009 Fase B) — fuente única.

La comparten el runtime (src/services/ocr.py) y el script de pre-descarga
(scripts/ocr_models_download.py). Se usan `params` (no un YAML) porque RapidOCR
los fusiona con su config por defecto; un YAML parcial la reemplazaría entera.

Modelo: PP-OCRv5 mobile `latin` (cubre español: tildes/ñ) sobre ONNX Runtime,
sin paddlepaddle → ligero para CPU y 100% offline una vez pre-descargado.
"""
from __future__ import annotations

import os


def build_params() -> dict:
    """Params para `RapidOCR(params=...)`. Vacío si rapidocr no está disponible."""
    try:
        from rapidocr import EngineType, ModelType, OCRVersion
    except Exception:  # pragma: no cover - rapidocr ausente en host
        return {}

    params: dict = {
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Cls.engine_type": EngineType.ONNXRUNTIME,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_type": ModelType.MOBILE,
    }
    lang = os.environ.get("OCR_RAPID_LANG", "latin")
    try:
        from rapidocr import LangRec

        params["Rec.lang_type"] = getattr(LangRec, lang.upper(), lang)
    except Exception:
        params["Rec.lang_type"] = lang
    return params
