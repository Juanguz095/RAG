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

OCR_DPI = 200


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
    page = doc.load_page(page_idx)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = img.convert("L")
    img = img.point(lambda x: 0 if x < 140 else 255, mode="1")
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


def _tesseract_page(img: Image.Image) -> tuple[str, list[dict]]:
    import pytesseract

    data = pytesseract.image_to_data(
        img,
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
                text, words = _tesseract_page(img)
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
