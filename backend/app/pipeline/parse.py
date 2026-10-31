"""Stage 1 — Parse: extract text from PDF pages with OCR fallback."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import structlog

from app.core.config import get_settings
from app.core.errors import InvalidPDFError

log = structlog.get_logger(__name__)


@dataclass
class PageContent:
    page_num: int  # 1-indexed
    text: str
    used_ocr: bool


def _ocr_page(page: fitz.Page) -> str:
    try:
        import pytesseract
        from PIL import Image

        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as exc:
        log.warning("OCR failed for page", page=page.number + 1, error=str(exc))
        return ""


def parse_pdf(file_path: Path) -> list[PageContent]:
    """Parse all pages of a PDF; apply OCR to pages with sparse text.

    Raises InvalidPDFError on corrupt/encrypted PDFs.
    """
    settings = get_settings()

    with file_path.open("rb") as fh:
        header = fh.read(5)
    if header != b"%PDF-":
        raise InvalidPDFError("File is not a valid PDF (bad magic bytes)")

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        raise InvalidPDFError(f"Could not open PDF: {exc}") from exc

    if doc.is_encrypted:
        raise InvalidPDFError("PDF is encrypted/password-protected")

    page_count = doc.page_count
    if page_count > settings.max_pages:
        raise InvalidPDFError(
            f"PDF has {page_count} pages; maximum is {settings.max_pages}"
        )

    results: list[PageContent] = []
    min_chars = settings.min_text_chars_per_page

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        used_ocr = False

        if len(text) < min_chars:
            log.debug("Sparse page, running OCR", page=i + 1, char_count=len(text))
            ocr_text = _ocr_page(page)
            if ocr_text:
                text = ocr_text
                used_ocr = True

        try:
            tables = page.find_tables()
            for table in tables.tables:
                for row in table.extract():
                    row_text = " | ".join(str(c) for c in row if c)
                    if row_text.strip():
                        text += "\n" + row_text
        except Exception:
            pass

        results.append(PageContent(page_num=i + 1, text=text, used_ocr=used_ocr))

    doc.close()
    log.info("PDF parsed", path=str(file_path), pages=page_count,
             ocr_pages=sum(1 for p in results if p.used_ocr))
    return results
