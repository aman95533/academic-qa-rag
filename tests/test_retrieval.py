"""
Tests for retrieval.py and the evaluation helper retrieval_hit_at_k.
Uses mocks so no real Cohere/ChromaDB calls are made.
"""
from unittest.mock import MagicMock, patch

import pytest

from evaluation import retrieval_hit_at_k as eval_hit
from ingest import TextChunk
from retrieval import RetrievalError, RetrievedChunk, VectorStore


# ── retrieval_hit_at_k ────────────────────────────────────────────────────────

def test_retrieval_hit_at_k_found():
    chunks = [
        RetrievedChunk("1", "Photosynthesis in plants", "a.pdf", 1, 0.1),
        RetrievedChunk("2", "Other topic", "a.pdf", 2, 0.2),
    ]
    assert eval_hit(chunks, ["photosynthesis"])


def test_retrieval_hit_at_k_not_found():
    chunks = [
        RetrievedChunk("1", "Photosynthesis in plants", "a.pdf", 1, 0.1),
    ]
    assert not eval_hit(chunks, ["quantum"])


def test_retrieval_hit_at_k_empty_substrings():
    chunks = [RetrievedChunk("1", "anything", "a.pdf", 1, 0.1)]
    assert not eval_hit(chunks, [])


def test_retrieval_hit_at_k_empty_chunks():
    assert not eval_hit([], ["photosynthesis"])


# ── VectorStore (mocked) ──────────────────────────────────────────────────────

@patch("retrieval.get_cohere_api_key", return_value="test-key")
@patch("retrieval.chromadb.PersistentClient")
def test_index_chunks_skips_duplicates(mock_client_cls, mock_key):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.get.return_value = {"ids": []}
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_cls.return_value = mock_client

    store = VectorStore(persist_directory="dummy-path")
    with patch.object(store, "embed_texts", return_value=[[0.1] * 8, [0.2] * 8]):
        chunks = [
            TextChunk("id1", "text one", "f.pdf", 1, 0),
            TextChunk("id2", "text two", "f.pdf", 1, 1),
        ]
        n = store.index_chunks(chunks)
        assert n == 2

        # Re-index same chunks — should be skipped (count > 0, ids already present)
        mock_collection.get.return_value = {"ids": ["id1", "id2"]}
        mock_collection.count.return_value = 2
        n2 = store.index_chunks(chunks)
        assert n2 == 0


@patch("retrieval.get_cohere_api_key", return_value="test-key")
@patch("retrieval.chromadb.PersistentClient")
def test_index_empty_chunks_returns_zero(mock_client_cls, mock_key):
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = MagicMock()
    mock_client_cls.return_value = mock_client

    store = VectorStore(persist_directory="dummy-path")
    n = store.index_chunks([])
    assert n == 0


@patch("retrieval.get_cohere_api_key", return_value="test-key")
@patch("retrieval.chromadb.PersistentClient")
def test_retrieve_raises_when_index_empty(mock_client_cls, mock_key):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_cls.return_value = mock_client

    store = VectorStore(persist_directory="dummy-path")
    with pytest.raises(RetrievalError, match="No documents indexed"):
        store.retrieve("some question", top_k=5)


@patch("retrieval.get_cohere_api_key", return_value="test-key")
@patch("retrieval.chromadb.PersistentClient")
def test_retrieve_raises_on_empty_query(mock_client_cls, mock_key):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 5
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_cls.return_value = mock_client

    store = VectorStore(persist_directory="dummy-path")
    with pytest.raises(RetrievalError, match="Query is empty"):
        store.retrieve("   ", top_k=5)
