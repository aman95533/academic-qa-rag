"""
Tests for text chunking logic in ingest.py.
All tests use in-memory data only (no API calls, no filesystem).
"""
from ingest import (
    ExtractedDocument,
    PageText,
    TextChunk,
    _split_text_with_overlap,
    chunk_pages,
)


def test_split_overlap_produces_multiple_chunks():
    text = "a" * 1000
    parts = _split_text_with_overlap(text, chunk_size=400, chunk_overlap=50)
    assert len(parts) >= 2
    assert all(len(p) <= 400 for p in parts)


def test_split_overlap_short_text_single_chunk():
    text = "short text"
    parts = _split_text_with_overlap(text, chunk_size=400, chunk_overlap=50)
    assert len(parts) == 1
    assert parts[0] == text


def test_chunk_pages_preserves_page_metadata():
    doc = ExtractedDocument(
        filename="sample.pdf",
        pages=[
            PageText(page_number=1, text="Photosynthesis uses chlorophyll."),
            PageText(page_number=2, text="Newton's first law describes inertia."),
        ],
        is_likely_scanned=False,
        total_chars=80,
    )
    chunks = chunk_pages(doc, chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 2
    pages = {c.page_number for c in chunks}
    assert pages == {1, 2}
    assert all(c.source_filename == "sample.pdf" for c in chunks)
    assert len({c.content_hash for c in chunks}) == 2


def test_scanned_pdf_yields_no_chunks():
    doc = ExtractedDocument(
        filename="scan.pdf",
        pages=[PageText(page_number=1, text="")],
        is_likely_scanned=True,
        total_chars=0,
    )
    assert chunk_pages(doc) == []


def test_chunk_ids_unique():
    long_text = "word " * 500
    doc = ExtractedDocument(
        filename="multi.pdf",
        pages=[
            PageText(page_number=1, text=long_text),
            PageText(page_number=2, text=long_text + " extra"),
        ],
        is_likely_scanned=False,
        total_chars=len(long_text) * 2 + 6,
    )
    chunks = chunk_pages(doc, chunk_size=300, chunk_overlap=30)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "All chunk IDs must be unique"


def test_chunk_overlap_exceeds_size_raises():
    """chunk_overlap >= chunk_size should raise ValueError."""
    doc = ExtractedDocument(
        filename="x.pdf",
        pages=[PageText(page_number=1, text="a" * 500)],
        is_likely_scanned=False,
        total_chars=500,
    )
    try:
        chunk_pages(doc, chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError was not raised"
    except ValueError:
        pass


def test_text_chunk_to_metadata():
    chunk = TextChunk(
        chunk_id="doc.pdf::p1::c0",
        text="some text",
        source_filename="doc.pdf",
        page_number=1,
        chunk_index=0,
        content_hash="abc123",
    )
    meta = chunk.to_metadata()
    assert meta["source_filename"] == "doc.pdf"
    assert meta["page_number"] == 1
    assert meta["chunk_index"] == 0
    assert meta["content_hash"] == "abc123"


def test_empty_page_skipped():
    doc = ExtractedDocument(
        filename="sparse.pdf",
        pages=[
            PageText(page_number=1, text=""),
            PageText(page_number=2, text="Real content here."),
        ],
        is_likely_scanned=False,
        total_chars=18,
    )
    chunks = chunk_pages(doc, chunk_size=100, chunk_overlap=10)
    assert all(c.page_number == 2 for c in chunks)
