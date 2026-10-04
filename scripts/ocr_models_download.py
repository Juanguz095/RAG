"""Pre-descarga los modelos RapidOCR (PP-OCRv5 latin) durante el build Docker.

Así la ingesta OCR funciona 100% offline (sin descargar en el primer uso). Los
params se toman de src/services/rapidocr_params.py (misma fuente que el runtime).

Uso: python scripts/ocr_models_download.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_params() -> dict:
    """Carga `build_params()` sin depender de que `src` sea importable."""
    root = Path(__file__).resolve().parent.parent
    params_file = root / "src" / "services" / "rapidocr_params.py"
    spec = importlib.util.spec_from_file_location("rapidocr_params", params_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.build_params()


def main() -> int:
    from PIL import Image
    from rapidocr import RapidOCR

    engine = RapidOCR(params=_load_params())
    # Una inferencia trivial fuerza la descarga de det/cls/rec a la caché local
    # (rapidocr/models dentro de site-packages, que se copia a la imagen runtime).
    engine(Image.new("RGB", (320, 120), "white"))
    print("[ocr_models_download] modelos RapidOCR pre-descargados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
