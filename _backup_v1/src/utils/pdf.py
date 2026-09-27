
import fitz  # PyMuPDF
from PIL import Image


class PDFInfo:
    def __init__(self, page_count: int, metadata: dict, is_scanned: bool = False):
        self.page_count = page_count
        self.metadata = metadata
        self.is_scanned = is_scanned


def validate_pdf(content: bytes) -> PDFInfo:
    doc = fitz.open(stream=content, filetype="pdf")
    metadata = doc.metadata or {}
    page_count = len(doc)

    is_scanned = _detect_if_scanned(doc)
    doc.close()

    return PDFInfo(page_count=page_count, metadata=metadata, is_scanned=is_scanned)


def _detect_if_scanned(doc: fitz.Document) -> bool:
    text_chars = 0
    sample_pages = min(3, len(doc))
    for i in range(sample_pages):
        page = doc[i]
        text_chars += len(page.get_text("text").strip())
    avg_chars = text_chars / max(sample_pages, 1)
    return avg_chars < 50


def extract_page_images(content: bytes, dpi: int = 300) -> list[Image.Image]:
    doc = fitz.open(stream=content, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


def extract_page_text(content: bytes, page_num: int) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    if page_num < 0 or page_num >= len(doc):
        doc.close()
        return ""
    text = str(doc[page_num].get_text("text"))
    doc.close()
    return text


def get_page_count(content: bytes) -> int:
    doc = fitz.open(stream=content, filetype="pdf")
    count = len(doc)
    doc.close()
    return count
