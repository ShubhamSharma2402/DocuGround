"""Unit tests for upload validation: magic bytes, PDF validity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.core.errors import InvalidPDFError
from app.pipeline.parse import parse_pdf


class TestPDFValidation:
    def _write_temp(self, content: bytes) -> Path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(content)
            return Path(f.name)

    def test_wrong_magic_bytes_rejected(self):
        path = self._write_temp(b"This is not a PDF file at all.")
        try:
            with pytest.raises(InvalidPDFError, match="magic bytes"):
                parse_pdf(path)
        finally:
            path.unlink(missing_ok=True)

    def test_png_magic_bytes_rejected(self):
        path = self._write_temp(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        try:
            with pytest.raises(InvalidPDFError, match="magic bytes"):
                parse_pdf(path)
        finally:
            path.unlink(missing_ok=True)

    def test_truncated_pdf_rejected(self):
        path = self._write_temp(b"%PDF-1.4 truncated content")
        try:
            with pytest.raises(InvalidPDFError):
                parse_pdf(path)
        finally:
            path.unlink(missing_ok=True)
