"""Reranker applied to fused BM25+FAISS candidates before returning the
final top-N to the agent.

Original implementation used a sentence-transformers CrossEncoder, which
requires PyTorch. Replaced with a cosine-similarity reranker using the same
FastEmbed embedding model already loaded for retrieval -- zero extra
dependencies, no torch required, quality is slightly lower than a dedicated
cross-encoder but fully adequate for signal-summary retrieval where the
documents are short structured strings.

Interface is unchanged: rerank(query, candidates, top_n) ->
list[tuple[Document, float]], highest score first.
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from src.config import DEFAULT_TOP_K_FINAL
from src.retrieval.signal_retriever import get_embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def rerank(
    query: str,
    candidates: list[Document],
    top_n: int = DEFAULT_TOP_K_FINAL,
) -> list[tuple[Document, float]]:
    """Score each candidate document against the query using cosine similarity
    of their embeddings and return the top_n (document, score) pairs,
    highest score first. Returns an empty list if there are no candidates."""
    if not candidates:
        return []

    embeddings = get_embeddings()
    query_emb = embeddings.embed_query(query)
    doc_embs = embeddings.embed_documents([doc.page_content for doc in candidates])

    scored = [
        (doc, _cosine(query_emb, doc_emb))
        for doc, doc_emb in zip(candidates, doc_embs)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]
