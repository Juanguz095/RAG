import gc
import logging
from dataclasses import dataclass, field

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class OCRResult:
    text: str
    confidence: float = 0.0
    bbox: dict | None = None
    page_num: int = 0
    engine: str = ""


@dataclass
class PageOCRResult:
    page_num: int
    text: str
    lines: list[OCRResult] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    layout: list[dict] = field(default_factory=list)
    engine: str = ""
    confidence: float = 0.0


_surya_det_model = None
_surya_det_processor = None
_surya_rec_model = None
_surya_rec_processor = None
_paddle_ocr = None


def _get_surya_detection():
    global _surya_det_model, _surya_det_processor
    if _surya_det_model is None:
        try:
            from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
            _surya_det_model = load_det_model()
            _surya_det_processor = load_det_processor()
            logger.info("Surya detection model loaded")
        except Exception as e:
            logger.warning(f"Failed to load Surya detection: {e}")
    return _surya_det_model, _surya_det_processor


def _get_surya_recognition():
    global _surya_rec_model, _surya_rec_processor
    if _surya_rec_model is None:
        try:
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            _surya_rec_model = load_rec_model()
            _surya_rec_processor = load_rec_processor()
            logger.info("Surya recognition model loaded")
        except Exception as e:
            logger.warning(f"Failed to load Surya recognition: {e}")
    return _surya_rec_model, _surya_rec_processor


def _get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="es", show_log=False)
            logger.info("PaddleOCR loaded")
        except Exception as e:
            logger.warning(f"Failed to load PaddleOCR: {e}")
    return _paddle_ocr


def ocr_page_surya(image: Image.Image, page_num: int = 0) -> PageOCRResult:
    det_model, det_processor = _get_surya_detection()
    rec_model, rec_processor = _get_surya_recognition()

    if det_model is None or rec_model is None:
        logger.warning("Surya not available, falling back to PaddleOCR")
        return ocr_page_paddle(image, page_num)

    try:
        from surya.ocr import run_ocr

        pil_image = image.convert("RGB")
        predictions = run_ocr(
            [pil_image], [["es"]],
            det_model, det_processor,
            rec_model, rec_processor,
        )
        del pil_image
        gc.collect()

        full_text_parts = []
        lines = []
        if predictions and len(predictions) > 0:
            for text_line in predictions[0].text_lines:
                text = text_line.text.strip()
                if text:
                    full_text_parts.append(text)
                    lines.append(OCRResult(
                        text=text,
                        confidence=getattr(text_line, "confidence", 0.9),
                        bbox=None,
                        page_num=page_num,
                        engine="surya",
                    ))

        full_text = "\n".join(full_text_parts)
        avg_conf = sum(line.confidence for line in lines) / max(len(lines), 1)

        return PageOCRResult(
            page_num=page_num,
            text=full_text,
            lines=lines,
            engine="surya",
            confidence=avg_conf,
        )
    except Exception as e:
        logger.error(f"Surya OCR failed on page {page_num}: {e}")
        return ocr_page_paddle(image, page_num)


def ocr_page_paddle(image: Image.Image, page_num: int = 0) -> PageOCRResult:
    paddle = _get_paddle_ocr()
    if paddle is None:
        logger.error("PaddleOCR not available")
        return PageOCRResult(page_num=page_num, text="", engine="none", confidence=0.0)

    try:
        numpy = __import__("numpy")
        img_array = numpy.array(image)
        result = paddle.ocr(img_array, cls=True)

        full_text_parts = []
        lines = []
        if result and result[0]:
            for line in result[0]:
                coords = line[0]
                text = line[1][0]
                conf = line[1][1]
                full_text_parts.append(text)

                bbox = {
                    "x0": min(c[0] for c in coords),
                    "y0": min(c[1] for c in coords),
                    "x1": max(c[0] for c in coords),
                    "y1": max(c[1] for c in coords),
                } if coords else None

                lines.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=bbox,
                    page_num=page_num,
                    engine="paddleocr",
                ))

        full_text = "\n".join(full_text_parts)
        avg_conf = sum(line.confidence for line in lines) / max(len(lines), 1)

        return PageOCRResult(
            page_num=page_num,
            text=full_text,
            lines=lines,
            engine="paddleocr",
            confidence=avg_conf,
        )
    except Exception as e:
        logger.error(f"PaddleOCR failed on page {page_num}: {e}")
        return PageOCRResult(page_num=page_num, text="", engine="error", confidence=0.0)


def ocr_page(image: Image.Image, page_num: int = 0) -> PageOCRResult:
    """OCR a scanned page using the bounded local engine."""
    tesseract_result = ocr_page_tesseract(image, page_num)
    if tesseract_result.text.strip() and tesseract_result.confidence > 0.3:
        return tesseract_result

    # Try PaddleOCR as fallback (lighter than Surya, no large model download)
    paddle_result = ocr_page_paddle(image, page_num)
    if paddle_result.text.strip():
        logger.info(f"PaddleOCR succeeded on page {page_num}")
        return paddle_result

    # Try Surya as final fallback for best quality
    surya_result = ocr_page_surya(image, page_num)
    if surya_result.text.strip():
        logger.info(f"Surya OCR succeeded on page {page_num}")
        return surya_result

    return tesseract_result


def _prepare_tesseract_image(image: Image.Image) -> Image.Image:
    """Enhance scanned image for better OCR quality."""
    # Convert to grayscale
    gray = ImageOps.grayscale(image)

    # Resize if too small (OCR works better on larger images)
    width, height = gray.size
    if width < 1000:
        scale = 1000 / width
        gray = gray.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    # Auto contrast
    gray = ImageOps.autocontrast(gray, cutoff=2)

    # Sharpen
    gray = gray.filter(ImageFilter.SHARPEN)

    # Enhance contrast
    gray = ImageEnhance.Contrast(gray).enhance(1.5)

    # Enhance brightness slightly
    gray = ImageEnhance.Brightness(gray).enhance(1.1)

    return gray


def _ocr_tesseract_once(image: Image.Image, page_num: int, config: str) -> PageOCRResult:
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        lang="spa+eng",
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    lines: list[OCRResult] = []
    for i, value in enumerate(data.get("text", [])):
        text = value.strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i]) / 100.0
        except (KeyError, ValueError, TypeError):
            confidence = 0.0
        if confidence < 0:
            continue
        bbox = {
            "x0": int(data["left"][i]),
            "y0": int(data["top"][i]),
            "x1": int(data["left"][i] + data["width"][i]),
            "y1": int(data["top"][i] + data["height"][i]),
        }
        lines.append(OCRResult(text, confidence, bbox, page_num, "tesseract"))
    return PageOCRResult(
        page_num=page_num,
        text=" ".join(line.text for line in lines),
        lines=lines,
        engine="tesseract",
        confidence=sum(line.confidence for line in lines) / max(len(lines), 1),
    )


def ocr_page_tesseract(image: Image.Image, page_num: int = 0) -> PageOCRResult:
    """Run two local Tesseract layouts over an enhanced scan and keep the best result."""
    try:
        prepared = _prepare_tesseract_image(image)
        result = _ocr_tesseract_once(prepared, page_num, "--oem 3 --psm 6")
        if result.text.strip():
            return result
        return _ocr_tesseract_once(prepared, page_num, "--oem 3 --psm 11")
    except Exception as exc:
        logger.warning("Tesseract OCR unavailable on page %d: %s", page_num, exc)
        return PageOCRResult(page_num=page_num, text="", engine="none", confidence=0.0)


def ocr_document(images: list[Image.Image]) -> list[PageOCRResult]:
    results = []
    for i, img in enumerate(images):
        logger.info(f"OCR processing page {i + 1}/{len(images)}")
        result = ocr_page(img, page_num=i)
        results.append(result)
        del img
        gc.collect()
    return results
