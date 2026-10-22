"""Cohere embeddings and ChromaDB semantic retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COHERE_EMBED_MODEL,
    COHERE_RERANK_MODEL,
    DEFAULT_TOP_K,
    ENABLE_RERANK,
    get_cohere_api_key,
)
from ingest import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    distance: float | None
    score: float | None = None


class RetrievalError(Exception):
    pass


class VectorStore:
    """Persistent ChromaDB store with Cohere embeddings."""

    def __init__(self, persist_directory: str | None = None) -> None:
        self.persist_directory = str(persist_directory or CHROMA_DIR)
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._cohere = None

    @property
    def cohere_client(self):
        if self._cohere is None:
            import cohere

            self._cohere = cohere.ClientV2(api_key=get_cohere_api_key())
        return self._cohere

    def close(self) -> None:
        """Release ChromaDB file handles (helps on Windows during tests)."""
        self._collection = None  # type: ignore[assignment]
        self._client = None  # type: ignore[assignment]

    def reset_index(self) -> None:
        """Delete and recreate the collection (safe rebuild)."""
        name = self._collection.name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        names = set()
        for meta in result.get("metadatas") or []:
            if meta and "source_filename" in meta:
                names.add(meta["source_filename"])
        return sorted(names)

    def embed_texts(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.cohere_client.embed(
                texts=list(texts),
                model=COHERE_EMBED_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )
            emb = response.embeddings
            # Cohere SDK exposes float vectors as `.float` (alias for float_).
            vectors = getattr(emb, "float", None) or getattr(emb, "float_", None)
            if vectors is None:
                raise RetrievalError("Unexpected Cohere embed response format.")
            return vectors
        except ValueError as exc:
            raise exc
        except Exception as exc:
            raise RetrievalError(f"Embedding failed: {exc}") from exc

    def index_chunks(self, chunks: Sequence[TextChunk]) -> int:
        if not chunks:
            return 0

        existing_ids = set()
        if self.count() > 0:
            stored = self._collection.get(include=[])
            existing_ids = set(stored.get("ids") or [])

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return 0

        texts = [c.text for c in new_chunks]
        embeddings = self.embed_texts(texts, input_type="search_document")
        if len(embeddings) != len(new_chunks):
            raise RetrievalError("Embedding count mismatch during indexing.")

        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.to_metadata() for c in new_chunks],
        )
        return len(new_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_filter: str | None = None,
        use_rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise RetrievalError("Query is empty.")
        if self.count() == 0:
            raise RetrievalError(
                "No documents indexed yet. Upload and index PDFs first."
            )

        query_emb = self.embed_texts([query], input_type="search_query")
        if not query_emb:
            raise RetrievalError("Could not embed query.")

        where = None
        if source_filter:
            where = {"source_filename": source_filter}

        n_results = min(max(top_k * 2, top_k), self.count()) if ENABLE_RERANK else top_k
        n_results = min(n_results, self.count())

        try:
            results = self._collection.query(
                query_embeddings=query_emb,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        retrieved: list[RetrievedChunk] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            if not doc or not meta:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    source_filename=str(meta.get("source_filename", "unknown")),
                    page_number=int(meta.get("page_number", 0)),
                    distance=float(dist) if dist is not None else None,
                )
            )

        do_rerank = ENABLE_RERANK if use_rerank is None else use_rerank
        if do_rerank and len(retrieved) > 1:
            retrieved = rerank_chunks(query, retrieved, top_k=top_k)
        else:
            retrieved = retrieved[:top_k]

        return retrieved


def rerank_chunks(
    query: str, chunks: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Optional Cohere Rerank to reorder retrieved passages."""
    try:
        import cohere

        client = cohere.ClientV2(api_key=get_cohere_api_key())
        docs = [c.text for c in chunks]
        response = client.rerank(
            model=COHERE_RERANK_MODEL,
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
        )
        reranked: list[RetrievedChunk] = []
        for item in response.results:
            base = chunks[item.index]
            reranked.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    text=base.text,
                    source_filename=base.source_filename,
                    page_number=base.page_number,
                    distance=base.distance,
                    score=float(item.relevance_score),
                )
            )
        return reranked
    except Exception as exc:
        logger.warning("Rerank failed, using vector order: %s", exc)
        return chunks[:top_k]
