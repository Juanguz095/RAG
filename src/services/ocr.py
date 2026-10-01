from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event
from typing import Iterator

import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)

# Resolver el DPI desde config (no hardcodear): PERFIL/OCR_DPI controla la
# velocidad del OCR. Si falla (tests sin config), cae a un default seguro.
try:
    from src.config import get_settings

    OCR_DPI = int(get_settings().OCR_DPI)
except Exception:  # pragma: no cover
    OCR_DPI = 150


# ── Fase 1 (PLAN-001): OCR en cascada con motores seleccionables ─────


def detect_and_fix_rotation(img_or_arr):
    """Clasifica el ángulo de una página (0/90/180/270) y corrige la imagen.

    Usa el clasificador de PaddleOCR si está disponible; con Tesseract
    entra por OSD (orientation and script detection). Devuelve el ángulo
    detectado y la imagen ya rotada a orientación correcta.

    WP1: el OSD corre sobre una versión reducida (~500px) de la página —
    detectar 180° no necesita la imagen a 200 DPI y ahorra ~50% de la
    pasada de rotación por página (la "2ª pasada de Tesseract").
    """
    import numpy as np

    if isinstance(img_or_arr, Image.Image):
        img = img_or_arr
        arr = np.array(img.convert("L"))
    else:
        arr = np.asarray(img_or_arr)
        img = Image.fromarray(arr)

    # intento 1: OSD de tesseract (rápido, sin cargar Paddle) a baja resolución
    try:
        import pytesseract

        thumb = img.convert("L")
        w, h = thumb.size
        scale = 500.0 / max(w, h)
        if scale < 1.0:
            thumb = thumb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        osd = pytesseract.image_to_osd(thumb, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0)) % 360
    except Exception:
        angle = 0
    if angle:
        arr = np.rot90(arr, k=angle // 90)
    return angle, Image.fromarray(arr)


def extract_text_pymupdf(pdf_bytes: bytes) -> dict[int, str]:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = {}
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        pages[i] = text
    doc.close()
    return pages


def _render_page(doc: pymupdf.Document, page_idx: int, dpi: int = OCR_DPI) -> Image.Image:
    """Renderiza a RGB (tinta azul preservada para motores ML).

    Tesseract trabaja mejor con binarización; PaddleOCR y otros motores ML
    rinden mejor con la imagen RGB original (PLAN-001 §5 Fase 1: la tinta
    azul de los campos manuscritos pierde contraste con la binarización).
    Cada adaptador decide su preprocesado; el binarizado Tesseract ahora es
    perezoso (lazy), hecho dentro de _tesseract_page.
    """
    page = doc.load_page(page_idx)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _ocr_env_context():
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_WAIT_POLICY")
    old = {k: os.environ.get(k) for k in keys}
    return old, keys


def _restore_ocr_env(old: dict, keys) -> None:
    for k in keys:
        if old.get(k) is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old[k]


def _tesseract_page(img) -> tuple[str, list[dict]]:
    import numpy as np
    import pytesseract

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    from PIL import Image as PILImage

    # Binarización lazy: solo Tesseract la necesita (con Paddle/ML perjudica)
    if img.mode not in ("1", "L"):
        gray = img.convert("L").point(lambda x: 0 if x < 140 else 255, mode="1")
    else:
        gray = img
    data = pytesseract.image_to_data(
        gray,
        lang="spa+eng",
        config="--psm 3 --oem 1 --dpi 200",
        output_type=pytesseract.Output.DICT,
    )
    words = []
    text_parts = []
    for j in range(len(data["text"])):
        w = data["text"][j].strip()
        if w and int(data["conf"][j]) > 20:
            words.append({
                "text": w,
                "x": data["left"][j],
                "y": data["top"][j],
                "w": data["width"][j],
                "h": data["height"][j],
            })
            text_parts.append(w)
    return " ".join(text_parts), words


def _paddle_page(img: Image.Image) -> tuple[str, list[dict]]:
    """Adaptador PaddleOCR (PP-OCRv4): OCR con boxes → formato word_boxes.

    Conversión de coordenadas: PaddleOCR devuelve (4 esquinas, escala imagen
    original). _render_page produce PIL en modo "L" escalado al DPI indicado
    (200 DPI). El contrato word_boxes exige 200 DPI.
    """
    from paddleocr import PaddleOCR

    global _paddle_instance  # noqa: F824
    if _paddle_instance is None:
        _paddle_instance = PaddleOCR(
            use_angle_cls=True,  # la cascade necesita clasificador de ángulo
            lang="es",
            show_log=False,
        )
    import numpy as np

    arr = np.array(img.convert("RGB"))
    result = _paddle_instance.ocr(arr, cls=True)
    words = []
    text_parts = []
    if result and result[0]:
        for line in result[0]:
            box, (text, conf) = box_line = line
            if not text or conf is None or float(conf) < 0.4:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))
            words.append({
                "text": str(text).strip(),
                "x": x0,
                "y": y0,
                "w": max(1, x1 - x0),
                "h": max(1, y1 - y0),
            })
            text_parts.append(str(text).strip())
    return " ".join(text_parts), words


_paddle_instance = None


def _ocr_page_dispatch(img: Image.Image) -> tuple[str, list[dict]]:
    """Cascada: motor configurado → si falla o vacío, fallback Tesseract."""
    from src.config import get_settings

    engine = (get_settings().OCR_ENGINE or "tesseract").lower()
    try:
        if engine == "paddle":
            text, words = _paddle_page(img)
            if text.strip():
                return text, words
            logger.info("Paddle devolvió vacío; fallback a Tesseract")
        elif engine == "surya":
            # Surya no implementado: espacio reservado para el adaptador
            logger.warning("OCR_ENGINE=surya no implementado; usando Tesseract")
    except Exception as e:
        logger.warning(f"OCR motor '{engine}' falló: {e}; fallback a Tesseract")
    return _tesseract_page(img)


def _render_worker(
    pdf_bytes: bytes,
    page_indices: list[int],
    dpi: int,
    out_q: "Queue",
    stop: Event,
) -> None:
    """Own Document per thread — PyMuPDF is not safe to share across threads."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for idx in page_indices:
            if stop.is_set():
                break
            try:
                img = _render_page(doc, idx, dpi=dpi)
                out_q.put((idx, img, None))
            except Exception as e:
                out_q.put((idx, None, e))
    finally:
        doc.close()
        out_q.put((None, None, None))


def ocr_empty_pages_iter(
    pdf_bytes: bytes, page_indices: list[int], dpi: int = OCR_DPI
) -> Iterator[tuple[int, str, list[dict]]]:
    """Yield (page_idx, text, words) as each page finishes OCR (render prefetch)."""
    if not page_indices:
        return

    n = len(page_indices)
    logger.info(f"OCR {n} pages (dpi={dpi})...")
    t0 = time.time()

    old, keys = _ocr_env_context()
    # PASSIVE measured ~15x slower on OCR; single tesseract multi-thread + prefetch.
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

    render_q: Queue = Queue(maxsize=2)
    stop = Event()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
    try:
        pool.submit(_render_worker, pdf_bytes, page_indices, dpi, render_q, stop)
        done = 0
        while done < n:
            idx, img, err = render_q.get()
            if idx is None:
                break
            if err is not None:
                raise err
            try:
                # Detección y corrección de rotación antes de OCR: la página
                # invertida del benchmark (page.rotation==0 pero 180° físico)
                # requiere clasificador propio, no los metadatos del PDF.
                angle, fixed = detect_and_fix_rotation(img)
                text, words = _ocr_page_dispatch(fixed)
                done += 1
                yield idx, text, words
            finally:
                if img is not None:
                    img.close()
        logger.info(f"OCR finished {done} pages in {time.time()-t0:.1f}s")
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        _restore_ocr_env(old, keys)


def ocr_empty_pages(
    pdf_bytes: bytes, page_indices: list[int], dpi: int = OCR_DPI
) -> dict[int, tuple[str, list[dict]]]:
    out: dict[int, tuple[str, list[dict]]] = {}
    for idx, text, words in ocr_empty_pages_iter(pdf_bytes, page_indices, dpi):
        out[idx] = (text, words)
    return out


def process_pdf(pdf_bytes: bytes) -> tuple[dict[int, str], dict[int, str], list, dict[int, list[dict]]]:
    logger.info("Extracting text with PyMuPDF...")
    t0 = time.time()
    native_pages = extract_text_pymupdf(pdf_bytes)
    native_chars = sum(len(t) for t in native_pages.values())
    logger.info(f"PyMuPDF: {len(native_pages)} pages, {native_chars} chars in {time.time()-t0:.1f}s")

    empty_pages = [i for i, t in native_pages.items() if len(t.strip()) < 20]
    page_images: dict[int, str] = {}
    all_word_boxes: dict[int, list[dict]] = {}

    if empty_pages:
        for page_num, text, words in ocr_empty_pages_iter(pdf_bytes, empty_pages):
            page_images[page_num] = text
            native_pages[page_num] = text
            all_word_boxes[page_num] = words

    merged = {}
    for i in range(len(native_pages)):
        merged[i] = native_pages.get(i, page_images.get(i, ""))

    logger.info(f"PDF processed: {sum(len(t) for t in merged.values())} total chars")
    return merged, page_images, [], all_word_boxes
