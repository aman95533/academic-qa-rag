import io
from pathlib import Path

import pymupdf as fitz
import pytest

from ingest import IngestionError, extract_text_from_pdf, validate_upload


def _make_pdf(text_by_page: list[str]) -> bytes:
    doc = fitz.open()
    for text in text_by_page:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_validate_upload_rejects_non_pdf():
    with pytest.raises(IngestionError):
        validate_upload("notes.txt", 100)


def test_extract_text_from_pdf(tmp_path: Path):
    pdf_bytes = _make_pdf(["Hello academic world.", "Page two content."])
    path = tmp_path / "test.pdf"
    path.write_bytes(pdf_bytes)
    doc = extract_text_from_pdf(path, "test.pdf")
    assert not doc.is_likely_scanned
    assert doc.total_chars > 0
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Hello" in doc.pages[0].text


def test_empty_pdf_detected_as_scanned():
    pdf_bytes = _make_pdf(["", ""])
    doc = extract_text_from_pdf(io.BytesIO(pdf_bytes), "empty.pdf")
    assert doc.is_likely_scanned
    assert doc.total_chars == 0
