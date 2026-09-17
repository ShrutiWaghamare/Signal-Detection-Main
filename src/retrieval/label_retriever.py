"""Hybrid retriever over ingested FDA drug label PDF chunks, reusing the
same HybridRetriever (BM25 + FAISS + RRF) pattern as signal_retriever.py.
Mirrors literature_retriever.py exactly except for which FAISS index it
loads -- label and literature corpora are kept in separate indexes because
their retrieval behavior differs enough from each other (and from the
structured signal summaries) that merging any of the three hurts recall on
the other two.
"""

from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import DEFAULT_TOP_K_SPARSE, LABELS_INDEX_DIR
from src.retrieval.signal_retriever import HybridRetriever, get_embeddings


def build_label_retriever() -> HybridRetriever | None:
    """Load the drug-label FAISS index if it exists. Returns None if no
    label PDFs have been ingested yet -- build_agent.py skips registering
    the search_drug_label tool in that case rather than erroring."""
    embeddings = get_embeddings()
    index_file = LABELS_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return None

    vector_store = FAISS.load_local(
        str(LABELS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
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
    )
