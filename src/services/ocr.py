from __future__ import annotations

import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    OCR_LANG = get_settings().OCR_LANG or "spa"
    OCR_DETECT_ROTATION = bool(get_settings().OCR_DETECT_ROTATION)
    OCR_AUTO_MIN_SCORE = float(getattr(get_settings(), "OCR_AUTO_MIN_SCORE", 60))
    OCR_LOW_CONF_THRESHOLD = float(getattr(get_settings(), "OCR_LOW_CONF_THRESHOLD", 45))
    OCR_ROTATION_MIN_SCORE = float(getattr(get_settings(), "OCR_ROTATION_MIN_SCORE", 60))
    OCR_WORKERS = int(getattr(get_settings(), "OCR_WORKERS", 3))
except Exception:  # pragma: no cover
    OCR_DPI = 150
    OCR_LANG = "spa"
    OCR_DETECT_ROTATION = False
    OCR_AUTO_MIN_SCORE = 60.0
    OCR_LOW_CONF_THRESHOLD = 45.0
    OCR_ROTATION_MIN_SCORE = 60.0
    OCR_WORKERS = 3


# ── Fase 1 (PLAN-001): OCR en cascada con motores seleccionables ─────


def _quick_ocr_score(img) -> float:
    """Confianza media de Tesseract en una orientación (para elegir 0° vs 180°).

    Corre a resolución reducida (~700px): solo se usa como desempate cuando el
    OSD no es concluyente, así que no penaliza el caso normal.
    """
    try:
        import pytesseract

        thumb = img.convert("L")
        w, h = thumb.size
        scale = 700.0 / max(w, h)
        if scale < 1.0:
            thumb = thumb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        data = pytesseract.image_to_data(
            thumb, lang=OCR_LANG, config="--psm 3 --oem 1",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return 0.0
    confs = []
    for c, t in zip(data.get("conf", []), data.get("text", [])):
        try:
            ci = int(c)
        except (TypeError, ValueError):
            continue
        if str(t).strip() and ci > 0:
            confs.append(ci)
    return (sum(confs) / len(confs)) if confs else 0.0


def detect_and_fix_rotation(img_or_arr):
    """Clasifica el ángulo de una página (0/90/180/270) y corrige la imagen.

    Usa el clasificador de PaddleOCR si está disponible; con Tesseract
    entra por OSD (orientation and script detection). Devuelve el ángulo
    detectado y la imagen ya rotada a orientación correcta.

    WP1: el OSD corre sobre una versión reducida (~500px) de la página —
    detectar 180° no necesita la imagen a 200 DPI y ahorra ~50% de la
    pasada de rotación por página (la "2ª pasada de Tesseract").

    WP3 (PLAN-009 Fase A): si el OSD falla (o da 0° con poca confianza), se
    aplica un fallback determinista que compara el OCR a 0° y 180° y se queda
    con el de mayor confianza. Esto recupera las páginas que hoy se indexaban
    como basura invertida (p. ej. `VIONY3438…`). El fallo deja de ser silencioso.
    """
    import numpy as np

    if isinstance(img_or_arr, Image.Image):
        img = img_or_arr
        arr = np.array(img.convert("L"))
    else:
        arr = np.asarray(img_or_arr)
        img = Image.fromarray(arr)

    # intento 1: OSD de tesseract (rápido, sin cargar Paddle) a baja resolución
    angle = 0
    osd_conf = None
    osd_failed = False
    try:
        import pytesseract

        thumb = img.convert("L")
        w, h = thumb.size
        scale = 500.0 / max(w, h)
        if scale < 1.0:
            thumb = thumb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        osd = pytesseract.image_to_osd(thumb, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0)) % 360
        try:
            osd_conf = float(osd.get("orientation_conf", 0.0))
        except (TypeError, ValueError):
            osd_conf = None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"OSD de rotacion fallo ({e}); probando 0/180 determinista")
        osd_failed = True

    # WP3: fallback determinista cuando OSD no concluye (excepción o 0° con
    # confianza baja). Solo en esos casos paga el doble OCR rápido.
    if osd_failed or (angle == 0 and osd_conf is not None and osd_conf < 1.0):
        s0 = _quick_ocr_score(img)
        rot180 = img.rotate(180, expand=True)
        s180 = _quick_ocr_score(rot180)
        # Exigir además un score ABSOLUTO decente: en páginas con poco texto
        # (manuscritos, casi en blanco) el score es bajo y ruidoso, y el 15%
        # relativo daba falsos positivos que invertían la página (y con ella las
        # coordenadas del resaltado). Solo giramos si la versión girada se lee
        # claramente mejor Y bien.
        if s180 >= OCR_ROTATION_MIN_SCORE and s180 > s0 * 1.15:
            logger.info(f"Rotacion 180 detectada por fallback (score {s180:.1f} > {s0:.1f})")
            return 180, rot180
        return 0, img

    if angle:
        arr = np.rot90(arr, k=angle // 90)
    return angle, Image.fromarray(arr)


def _remap_boxes_to_original(
    words: list[dict], angle: int, orig_w: int, orig_h: int
) -> list[dict]:
    """Mapea word_boxes del espacio ROTADO al de la imagen ORIGINAL.

    El OCR corre sobre la página ya enderezada (para leerla bien), pero el visor
    muestra la página ORIGINAL. Sin este mapeo el resaltado caería en la posición
    equivocada (p. ej. un código de cabecera aparecía al pie). `angle` es el giro
    aplicado (0/90/180/270 en sentido antihorario, igual que `np.rot90`).
    """
    if not angle or not words:
        return words
    out = []
    for wd in words:
        x, y, bw, bh = wd["x"], wd["y"], wd["w"], wd["h"]
        xs, ys = [], []
        for cx, cy in ((x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)):
            if angle == 180:
                ox, oy = orig_w - 1 - cx, orig_h - 1 - cy
            elif angle == 90:
                ox, oy = orig_w - 1 - cy, cx
            elif angle == 270:
                ox, oy = cy, orig_h - 1 - cx
            else:
                ox, oy = cx, cy
            xs.append(ox)
            ys.append(oy)
        nx0, ny0 = int(min(xs)), int(min(ys))
        nx1, ny1 = int(max(xs)), int(max(ys))
        out.append({**wd, "x": nx0, "y": ny0, "w": max(1, nx1 - nx0), "h": max(1, ny1 - ny0)})
    return out


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
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_THREAD_LIMIT")
    old = {k: os.environ.get(k) for k in keys}
    return old, keys


def _restore_ocr_env(old: dict, keys) -> None:
    for k in keys:
        if old.get(k) is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old[k]


def _tesseract_page_scored(img) -> tuple[str, list[dict], float]:
    """OCR Tesseract devolviendo además la confianza media (0-100) de la página.

    El score se calcula de la MISMA pasada de `image_to_data` (sin coste extra):
    es el criterio que usa la cascada `auto` para decidir si re-OCR con RapidOCR.
    """
    import numpy as np
    import pytesseract

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))

    # Binarización lazy: solo Tesseract la necesita (con Paddle/ML perjudica).
    # LUT de 256 entradas (C) en vez de un lambda Python por pixel (mucho más lento).
    if img.mode not in ("1", "L"):
        lut = [0] * 140 + [255] * (256 - 140)
        gray = img.convert("L").point(lut, mode="1")
    else:
        gray = img
    data = pytesseract.image_to_data(
        gray,
        lang=OCR_LANG,
        config=f"--psm 3 --oem 1 --dpi {OCR_DPI}",
        output_type=pytesseract.Output.DICT,
    )
    words = []
    text_parts = []
    confs: list[int] = []
    for j in range(len(data["text"])):
        w = data["text"][j].strip()
        if not w:
            continue
        try:
            c = int(data["conf"][j])
        except (TypeError, ValueError):
            c = -1
        if c > 20:
            words.append({
                "text": w,
                "x": data["left"][j],
                "y": data["top"][j],
                "w": data["width"][j],
                "h": data["height"][j],
            })
            text_parts.append(w)
            confs.append(c)
    score = (sum(confs) / len(confs)) if confs else 0.0
    return " ".join(text_parts), words, score


def _tesseract_page(img) -> tuple[str, list[dict]]:
    """Contrato heredado (2-tuple). Usa `_tesseract_page_scored` internamente."""
    text, words, _score = _tesseract_page_scored(img)
    return text, words


def _paddle_page(img: Image.Image) -> tuple[str, list[dict], float]:
    """Adaptador PaddleOCR (PP-OCRv4): OCR con boxes → formato word_boxes.

    Conversión de coordenadas: PaddleOCR devuelve (4 esquinas, escala imagen
    original). _render_page produce PIL escalado al DPI indicado. El score
    devuelto es la confianza media (0-100) de las líneas aceptadas.
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
    confs: list[float] = []
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
            confs.append(float(conf))
    score = (sum(confs) / len(confs) * 100.0) if confs else 0.0
    return " ".join(text_parts), words, score


_paddle_instance = None


# ── Fase B (PLAN-009): RapidOCR (ONNX, PP-OCRv5 latin) para manuscritos ──
# Instancia POR HILO: el OCR corre en paralelo (varias páginas a la vez) y
# RapidOCR no es seguro con una única instancia compartida entre hilos.
import threading as _threading

_rapidocr_local = _threading.local()


def _get_rapidocr():
    """Instancia RapidOCR perezosa, una por hilo (segura con OCR paralelo).

    Los modelos (PP-OCRv5 latin mobile) se pre-descargaron en el build; con los
    mismos params que aquí, RapidOCR los encuentra en la caché local (offline).
    """
    inst = getattr(_rapidocr_local, "instance", None)
    if inst is not None:
        return inst
    from rapidocr import RapidOCR

    from src.services.rapidocr_params import build_params

    inst = RapidOCR(params=build_params())
    _rapidocr_local.instance = inst
    return inst


def _rapidocr_page(img: Image.Image) -> tuple[str, list[dict], float]:
    """Adaptador RapidOCR (PP-OCRv5 latin): OCR con boxes → word_boxes.

    RapidOCR acepta la imagen en el mismo render que Tesseract/Paddle (RGB al
    OCR_DPI), así que las coordenadas son directamente compatibles. El score
    (0-1 en RapidOCR) se escala a 0-100 para homogeneizar con Tesseract.
    """
    import numpy as np

    engine = _get_rapidocr()
    arr = np.array(img.convert("RGB"))
    result = engine(arr)

    words = []
    text_parts = []
    scores: list[float] = []
    txts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    res_scores = getattr(result, "scores", None)
    if txts:
        for i, text in enumerate(txts):
            try:
                conf = float(res_scores[i]) if res_scores is not None else 1.0
            except (TypeError, ValueError, IndexError):
                conf = 1.0
            if not text or conf < 0.4:
                continue
            box = boxes[i] if boxes is not None and i < len(boxes) else None
            if box is None:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
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
            scores.append(conf)
    score = (sum(scores) / len(scores) * 100.0) if scores else 0.0
    return " ".join(text_parts), words, score


def _auto_cascade(img: Image.Image) -> tuple[str, list[dict], dict]:
    """Cascada adaptativa por página: Tesseract → (si baja calidad) RapidOCR.

    El caso normal (impreso) se queda en Tesseract (rápido, ~2 s/pág). Solo las
    páginas con confianza baja o vacías (manuscritas, giradas, degradadas) pagan
    el coste de RapidOCR. Si RapidOCR no está disponible, se conserva Tesseract.
    """
    from src.config import get_settings

    min_score = float(getattr(get_settings(), "OCR_AUTO_MIN_SCORE", OCR_AUTO_MIN_SCORE))
    text, words, tscore = _tesseract_page_scored(img)
    if text.strip() and tscore >= min_score:
        return text, words, {"ocr_engine": "tesseract", "ocr_score": tscore}
    try:
        rtext, rwords, rscore = _rapidocr_page(img)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"RapidOCR no disponible ({e}); se conserva Tesseract")
        return text, words, {"ocr_engine": "tesseract", "ocr_score": tscore}
    if rtext.strip() and (rscore >= tscore or not text.strip()):
        return rtext, rwords, {"ocr_engine": "rapidocr", "ocr_score": rscore}
    return text, words, {"ocr_engine": "tesseract", "ocr_score": tscore}


def _ocr_page_dispatch(img: Image.Image) -> tuple[str, list[dict], dict]:
    """Cascada: motor configurado → si falla o vacío, fallback Tesseract.

    Devuelve `(text, words, meta)` con `meta = {"ocr_engine", "ocr_score"}`
    para propagar la calidad del OCR hasta los chunks (PLAN-009 Fase C).
    """
    from src.config import get_settings

    engine = (get_settings().OCR_ENGINE or "tesseract").lower()

    if engine == "auto":
        return _auto_cascade(img)

    if engine == "rapidocr":
        try:
            text, words, score = _rapidocr_page(img)
            if text.strip():
                return text, words, {"ocr_engine": "rapidocr", "ocr_score": score}
            logger.info("RapidOCR devolvió vacío; fallback a Tesseract")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"OCR motor 'rapidocr' falló: {e}; fallback a Tesseract")
        text, words, score = _tesseract_page_scored(img)
        return text, words, {"ocr_engine": "tesseract", "ocr_score": score}

    if engine == "paddle":
        try:
            text, words, score = _paddle_page(img)
            if text.strip():
                return text, words, {"ocr_engine": "paddle", "ocr_score": score}
            logger.info("Paddle devolvió vacío; fallback a Tesseract")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"OCR motor 'paddle' falló: {e}; fallback a Tesseract")
    elif engine == "surya":
        # Surya no implementado: espacio reservado para el adaptador
        logger.warning("OCR_ENGINE=surya no implementado; usando Tesseract")

    text, words, score = _tesseract_page_scored(img)
    return text, words, {"ocr_engine": "tesseract", "ocr_score": score}


def aggregate_ocr_confidence(
    page_numbers: list[int], page_meta: dict[int, dict], threshold: float = OCR_LOW_CONF_THRESHOLD
) -> tuple[float | None, str | None, bool]:
    """Peor confianza OCR entre las páginas de un chunk (y su motor).

    Un chunk que cruza una página manuscrita mal leída queda marcado
    `low_confidence` para que el visor lo señale y no se confíe en él.
    """
    entries = [page_meta[p] for p in page_numbers if p in page_meta]
    if not entries:
        return None, None, False
    worst = min(entries, key=lambda m: m.get("ocr_score", 100.0))
    score = float(worst.get("ocr_score", 100.0))
    engine = worst.get("ocr_engine")
    return score, engine, score < threshold


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


def _ocr_page_full(idx: int, img: Image.Image) -> tuple[int, str, list[dict], dict]:
    """OCR completo de una página: rotación (opcional) + motor + remapeo.

    Cierra la imagen al terminar. Pensado para correr en un pool de hilos.
    """
    try:
        if OCR_DETECT_ROTATION:
            angle, fixed = detect_and_fix_rotation(img)
        else:
            angle, fixed = 0, img
        text, words, meta = _ocr_page_dispatch(fixed)
        # Las coordenadas vienen del espacio ROTADO; el visor muestra la página
        # original → remapearlas para que el resaltado coincida.
        if angle:
            words = _remap_boxes_to_original(words, angle, img.width, img.height)
        return idx, text, words, meta
    finally:
        try:
            img.close()
        except Exception:  # noqa: BLE001
            pass


def ocr_empty_pages_iter(
    pdf_bytes: bytes, page_indices: list[int], dpi: int = OCR_DPI
) -> Iterator[tuple[int, str, list[dict], dict]]:
    """Yield (page_idx, text, words, meta) as each page finishes OCR.

    `meta` = {"ocr_engine": str, "ocr_score": float} de la página (PLAN-009 Fase C).
    Las páginas se OCR-ean en paralelo (OCR_WORKERS) y se van entregando a medida
    que terminan; el llamador tolera el orden (bufferiza por índice).
    """
    if not page_indices:
        return

    n = len(page_indices)
    workers = max(1, int(OCR_WORKERS))
    logger.info(f"OCR {n} pages (dpi={dpi}, workers={workers})...")
    t0 = time.time()

    old, keys = _ocr_env_context()
    # PASSIVE measured ~15x slower on OCR; single tesseract multi-thread + prefetch.
    os.environ.pop("OMP_NUM_THREADS", None)
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    # Con varias páginas en paralelo, limitar cada Tesseract a 1 hilo evita
    # oversubscription (3 workers × N hilos de OpenMP saturan los cores).
    if workers > 1:
        os.environ["OMP_THREAD_LIMIT"] = "1"

    render_q: Queue = Queue(maxsize=2)
    stop = Event()
    render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
    ocr_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr")
    try:
        render_pool.submit(_render_worker, pdf_bytes, page_indices, dpi, render_q, stop)
        pending: set = set()
        render_done = False
        done = 0
        max_pending = workers + 2  # cola acotada: no acumular imágenes en RAM
        while done < n:
            while not render_done and len(pending) < max_pending:
                idx, img, err = render_q.get()
                if idx is None:
                    render_done = True
                    break
                if err is not None:
                    raise err
                pending.add(ocr_pool.submit(_ocr_page_full, idx, img))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                idx2, text, words, meta = fut.result()
                done += 1
                yield idx2, text, words, meta
        logger.info(f"OCR finished {done} pages in {time.time()-t0:.1f}s")
    finally:
        stop.set()
        render_pool.shutdown(wait=False, cancel_futures=True)
        ocr_pool.shutdown(wait=False, cancel_futures=True)
        _restore_ocr_env(old, keys)


def ocr_empty_pages(
    pdf_bytes: bytes, page_indices: list[int], dpi: int = OCR_DPI
) -> dict[int, tuple[str, list[dict], dict]]:
    out: dict[int, tuple[str, list[dict], dict]] = {}
    for idx, text, words, meta in ocr_empty_pages_iter(pdf_bytes, page_indices, dpi):
        out[idx] = (text, words, meta)
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
        for page_num, text, words, _meta in ocr_empty_pages_iter(pdf_bytes, empty_pages):
            page_images[page_num] = text
            native_pages[page_num] = text
            all_word_boxes[page_num] = words

    merged = {}
    for i in range(len(native_pages)):
        merged[i] = native_pages.get(i, page_images.get(i, ""))

    logger.info(f"PDF processed: {sum(len(t) for t in merged.values())} total chars")
    return merged, page_images, [], all_word_boxes
