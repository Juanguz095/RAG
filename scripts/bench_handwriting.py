"""Spike: comparar reconocedores de manuscrito en páginas reales (NO producción).

Detección de líneas con RapidOCR (solo det) y, sobre los MISMOS recortes de
línea, comparar varios reconocedores:

  - rapidocr            : rec actual (PP-OCRv5 latin mobile)
  - rapidocr_prep       : rec sobre el recorte preprocesado (CLAHE/contraste)
  - trocr-small         : microsoft/trocr-small-handwritten
  - trocr-base          : microsoft/trocr-base-handwritten

Mide tiempo por página/línea y compara el texto. Opcionalmente cuenta aciertos
sobre un conjunto de tokens esperados (referencia curada).

Uso:
  python scripts/bench_handwriting.py <pdf> <pages> [recognizers_csv]
  pages: "0" o "0,18,19,20,34,35"
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/app")

import numpy as np
import pymupdf
from PIL import Image, ImageOps

from src.services.ocr import OCR_DPI, _get_rapidocr, _render_page

TROCR_MODELS = {
    "trocr-small": "microsoft/trocr-small-handwritten",
    "trocr-base": "microsoft/trocr-base-handwritten",
}
_trocr_cache: dict = {}


def detect_lines(img: Image.Image) -> list[tuple[int, int, int, int]]:
    engine = _get_rapidocr()
    res = engine(np.array(img.convert("RGB")), use_det=True, use_cls=True, use_rec=False)
    boxes = getattr(res, "boxes", None)
    lines = []
    if boxes is not None:
        for box in boxes:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            if x1 > x0 and y1 > y0:
                lines.append((x0, y0, x1, y1))
    lines.sort(key=lambda b: (b[1], b[0]))
    return lines


def _prep(crop: Image.Image) -> Image.Image:
    """Realce barato: gris + autocontraste + CLAHE (tinta manuscrita)."""
    import cv2

    g = np.array(crop.convert("L"))
    g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(g)
    return Image.fromarray(g).convert("RGB")


def rapidocr_rec(crops: list[Image.Image], preprocessed: bool = False) -> tuple[list[str], float]:
    engine = _get_rapidocr()
    outs = []
    t0 = time.time()
    for c in crops:
        im = _prep(c) if preprocessed else c
        res = engine(np.array(im.convert("RGB")), use_det=False, use_cls=True, use_rec=True)
        txts = getattr(res, "txts", None)
        outs.append(" ".join(txts) if txts else "")
    return outs, time.time() - t0


def trocr_rec(crops: list[Image.Image], model_key: str) -> tuple[list[str], float]:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if model_key not in _trocr_cache:
        name = TROCR_MODELS[model_key]
        proc = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name)
        model.eval()
        _trocr_cache[model_key] = (proc, model)
    proc, model = _trocr_cache[model_key]
    t0 = time.time()
    outs = []
    with torch.no_grad():
        pv = proc(images=[c.convert("RGB") for c in crops], return_tensors="pt").pixel_values
        ids = model.generate(pv, max_new_tokens=48)
        outs = [s.strip() for s in proc.batch_decode(ids, skip_special_tokens=True)]
    return outs, time.time() - t0


def main() -> None:
    pdf = sys.argv[1] if len(sys.argv) > 1 else "/app/uploads/primer.pdf"
    pages = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0").split(",")]
    recs = (sys.argv[3] if len(sys.argv) > 3 else "rapidocr,rapidocr_prep,trocr-small").split(",")

    doc = pymupdf.open(pdf)
    print(f"PDF {pdf} pages={doc.page_count} test={pages} recs={recs}", flush=True)

    for p in pages:
        img = _render_page(doc, p, OCR_DPI)
        t0 = time.time()
        lines = detect_lines(img)
        det_s = time.time() - t0
        crops = [img.crop(b) for b in lines]
        print(f"\n=== PAGE {p} ({img.size[0]}x{img.size[1]}) lines={len(lines)} det={det_s:.2f}s ===", flush=True)

        results: dict[str, list[str]] = {}
        for r in recs:
            if r == "rapidocr":
                out, dt = rapidocr_rec(crops, preprocessed=False)
            elif r == "rapidocr_prep":
                out, dt = rapidocr_rec(crops, preprocessed=True)
            elif r in TROCR_MODELS:
                out, dt = trocr_rec(crops, r)
            else:
                print(f"  [skip] {r}", flush=True)
                continue
            results[r] = out
            print(f"  [{r}] {len(crops)} lines in {dt:.2f}s ({dt/max(len(crops),1):.3f}s/line)", flush=True)

        # Volcado por línea
        for i, box in enumerate(lines):
            print(f"  L{i:02d} {box}:", flush=True)
            for r, out in results.items():
                txt = out[i] if i < len(out) else ""
                print(f"      {r:14s} | {txt}", flush=True)

    doc.close()


if __name__ == "__main__":
    main()
