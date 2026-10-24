"""PDF extraction, validation, and text chunking with citation metadata."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pymupdf as fitz  # PyMuPDF

import pytesseract
from PIL import Image

from config import ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE, MAX_UPLOAD_BYTES

# import pytesseract
# from pathlib import Path

TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


@dataclass
class ExtractedDocument:
    filename: str
    pages: list[PageText]
    is_likely_scanned: bool
    total_chars: int


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    content_hash: str = field(default="")

    def to_metadata(self) -> dict:
        return {
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
        }


class IngestionError(Exception):
    """Raised when a file cannot be processed."""


def validate_upload(filename: str, file_size: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '{ext}'. Only PDF files are allowed.")
    if file_size <= 0:
        raise IngestionError("File is empty.")
    if file_size > MAX_UPLOAD_BYTES:
        raise IngestionError(
            f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

def _needs_ocr(text: str) -> bool:
    non_space = [c for c in text if not c.isspace()]

    if len(non_space) < 40:
        return True

    letters = sum(c.isalpha() for c in non_space)
    return letters / len(non_space) < 0.30


def _ocr_page(page) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )
    return pytesseract.image_to_string(image, lang="eng")

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_text_from_pdf(
    source: BinaryIO | Path,
    filename: str,
) -> ExtractedDocument:
    """Extract readable text, using OCR when PDF text is garbled."""
    try:
        if isinstance(source, Path):
            doc = fitz.open(source)
        else:
            data = source.read()
            if not data:
                raise IngestionError("File is empty or could not be read.")
            doc = fitz.open(stream=data, filetype="pdf")
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Could not open PDF '{filename}': {exc}"
        ) from exc

    pages: list[PageText] = []
    total_chars = 0

    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = normalize_whitespace(page.get_text("text") or "")

            if _needs_ocr(text):
                try:
                    ocr_text = normalize_whitespace(_ocr_page(page))
                    if ocr_text and not _needs_ocr(ocr_text):
                        text = ocr_text
                except Exception as exc:
                    raise IngestionError(
                        f"OCR failed on page {i + 1} of '{filename}': {exc}"
                    ) from exc

            pages.append(PageText(page_number=i + 1, text=text))
            total_chars += len(text)
    finally:
        doc.close()

    return ExtractedDocument(
        filename=filename,
        pages=pages,
        is_likely_scanned=(total_chars == 0),
        total_chars=total_chars,
    )

def chunk_pages(
    document: ExtractedDocument,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split document text into overlapping chunks; metadata stays per page."""
    if document.is_likely_scanned or document.total_chars == 0:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    seen_hashes: set[str] = set()
    global_index = 0

    for page in document.pages:
        if not page.text:
            continue

        page_chunks = _split_text_with_overlap(
            page.text, chunk_size, chunk_overlap
        )

        for piece in page_chunks:
            content_hash = hashlib.sha256(
                piece.encode("utf-8")
            ).hexdigest()[:16]

            if content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)
            chunk_id = (
                f"{document.filename}::p{page.page_number}::c{global_index}"
            )

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source_filename=document.filename,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    content_hash=content_hash,
                )
            )
            global_index += 1

    return chunks


def _split_text_with_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    parts: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()

        if piece:
            parts.append(piece)

        if end >= len(text):
            break

        start = max(0, end - chunk_overlap)

    return parts