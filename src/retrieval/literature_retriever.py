"""Hybrid retriever over ingested literature/abstract PDF chunks, reusing
the same HybridRetriever (BM25 + FAISS + RRF) pattern as signal_retriever.py.

Chunks whose source filename or body mentions the query tokens are promoted
(and unrelated PDFs are dropped when a matching source exists) so a query
like "DEPO-PROVERA MENINGIOMA" does not return a different product's PDF.
"""

from __future__ import annotations

import re

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import DEFAULT_TOP_K_FUSED, DEFAULT_TOP_K_SPARSE, LITERATURE_INDEX_DIR
from src.retrieval.signal_retriever import HybridRetriever, get_embeddings

_STOP = {"WITH", "FROM", "THIS", "THAT", "HAVE", "BEEN", "DRUG", "EVENT", "SIGNAL", "QUERY"}

# FAERS uses British spelling; many PDFs use US spelling or abbreviations.
_TOKEN_VARIANTS = {
    "HAEMORRHAGE": ["HAEMORRHAGE", "HEMORRHAGE", "HAEMORRHAGIC", "HEMORRHAGIC", "ICH"],
    "HEMORRHAGE": ["HAEMORRHAGE", "HEMORRHAGE", "HAEMORRHAGIC", "HEMORRHAGIC", "ICH"],
    "INTRACRANIAL": ["INTRACRANIAL", "INTRACEREBRAL", "ICH"],
    "MENINGIOMA": ["MENINGIOMA", "MENINGIOMAS"],
}


def _query_tokens(query: str) -> list[str]:
    return [
        t for t in re.split(r"[^A-Za-z0-9]+", query.upper())
        if len(t) >= 4 and t not in _STOP
    ]


def _expand_token(token: str) -> list[str]:
    return _TOKEN_VARIANTS.get(token, [token])


def _source_key(doc: Document) -> str:
    return (
        str(doc.metadata.get("source_file", ""))
        .upper()
        .replace("-", " ")
        .replace("_", " ")
    )


def _literature_post_fuse(all_docs: list[Document]):
    """Prefer chunks from PDFs (or text) that actually mention the query."""

    def _fn(query: str, fused: list[Document]) -> list[Document]:
        tokens = _query_tokens(query)
        if not tokens:
            return fused

        def hits(doc: Document) -> tuple[int, int, int]:
            src = _source_key(doc)
            body = doc.page_content.upper()
            src_n = sum(1 for t in tokens if any(v in src for v in _expand_token(t)))
            body_n = sum(1 for t in tokens if any(v in body for v in _expand_token(t)))
            event_tokens = [t for t in tokens if t in _TOKEN_VARIANTS or t in {"INTRACRANIAL", "MENINGIOMA"}]
            event_n = sum(
                1 for t in event_tokens if any(v in body for v in _expand_token(t))
            )
            return (event_n, body_n, src_n)

        src_hits = [d for d in all_docs if hits(d)[2] > 0]
        if src_hits:
            src_hits.sort(key=hits, reverse=True)
            with_event_body = [d for d in src_hits if hits(d)[0] > 0]
            chosen = with_event_body or src_hits
            return chosen[:DEFAULT_TOP_K_FUSED]

        body_hits = [d for d in fused if hits(d)[1] > 0]
        if body_hits:
            body_hits.sort(key=hits, reverse=True)
            return body_hits

        extras = [d for d in all_docs if hits(d)[1] > 0]
        if extras:
            extras.sort(key=hits, reverse=True)
            return extras[:DEFAULT_TOP_K_FUSED]
        return fused

    return _fn


def build_literature_retriever() -> HybridRetriever | None:
    """Load the literature FAISS index if it exists. Returns None if no
    literature has been ingested yet -- build_agent.py skips registering
    the search_literature tool in that case rather than erroring."""
    embeddings = get_embeddings()
    index_file = LITERATURE_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return None

    vector_store = FAISS.load_local(
        str(LITERATURE_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    documents: list[Document] = list(vector_store.docstore._dict.values())
    if not documents:
        return None

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = DEFAULT_TOP_K_SPARSE

    return HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        exact_match_fn=None,
        post_fuse_fn=_literature_post_fuse(documents),
    )
