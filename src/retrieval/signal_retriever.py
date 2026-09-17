"""Hybrid retriever (BM25 sparse + FAISS dense, fused with Reciprocal Rank
Fusion) over signal-report evidence text, with an exact-match override: if
the query names an exact/near-exact drug+event pair that exists in the
corpus, that document is forced to rank 1 instead of relying purely on
embedding similarity.

`HybridRetriever` here is intentionally generic (query text in, ranked
Documents out) so literature_retriever.py and label_retriever.py reuse the
exact same fusion + override logic over their own corpora instead of
duplicating it.
"""

from __future__ import annotations

import re
from functools import lru_cache

from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K_DENSE,
    DEFAULT_TOP_K_FUSED,
    DEFAULT_TOP_K_SPARSE,
    MIN_SIGNAL_COOCCURRENCE,
    RRF_K,
    SIGNALS_INDEX_DIR,
)
from src.ingestion.faers_ingest import load_signal_summary


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Shared embedding model instance (loaded once, reused by every
    retriever/index in the project).
    Uses FastEmbedEmbeddings (ONNX Runtime backend) -- no PyTorch required,
    no DLL issues on Windows. Model is downloaded on first call (~50MB) and
    cached in ~/.cache/fastembed/."""
    return FastEmbedEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = RRF_K
) -> list[Document]:
    """Fuse multiple ranked document lists into one ranking using
    Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_in_list)).
    Documents are matched by page_content since BM25 and FAISS return
    separate Document instances for the same underlying text."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_by_key.setdefault(key, doc)

    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in fused_keys]


_DRUG_EVENT_PATTERN = re.compile(
    r"([A-Za-z0-9\-\s]+?)\s+(?:and|,|&)\s+([A-Za-z0-9\-\s]+)", re.IGNORECASE
)


class HybridRetriever(BaseRetriever):
    """BM25 + FAISS retriever fused with RRF, with an optional exact-match
    override hook. Subclass corpora provide `documents`, a prebuilt
    `vector_store`, and an optional `exact_match_fn(query) -> Document|None`.

    `vector_store` and `bm25_retriever` are typed as `Any` rather than the
    concrete FAISS/BM25Retriever classes: production code always passes the
    real classes, but keeping the field structurally typed (only
    `.similarity_search(...)` / `.invoke(...)` are actually required) means
    this class is also swap-friendly (e.g. FAISS -> PineconeVectorStore)
    and testable with lightweight fakes, without pydantic rejecting a
    duck-typed double that implements the same interface.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document] = Field(default_factory=list)
    vector_store: Any = Field(...)
    bm25_retriever: Any = Field(...)
    exact_match_fn: Any = Field(default=None)
    post_fuse_fn: Any = Field(default=None)
    top_k_dense: int = DEFAULT_TOP_K_DENSE
    top_k_sparse: int = DEFAULT_TOP_K_SPARSE
    top_k_fused: int = DEFAULT_TOP_K_FUSED

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        dense_hits = self.vector_store.similarity_search(query, k=self.top_k_dense)
        sparse_hits = self.bm25_retriever.invoke(query)[: self.top_k_sparse]

        fused = _reciprocal_rank_fusion([dense_hits, sparse_hits])[: self.top_k_fused]

        if self.post_fuse_fn is not None:
            fused = self.post_fuse_fn(query, fused)

        if self.exact_match_fn is not None:
            override = self.exact_match_fn(query)
            if override is not None:
                # Exact/named-pair hits replace the fused list so a different
                # drug-event pair cannot leak into the agent's evidence.
                return [override]

        return fused


def _row_to_signal_text(row) -> str:
    prr = row["prr"] if not (row["prr"] != row["prr"]) else float("nan")  # nan check
    ror = row["ror"] if not (row["ror"] != row["ror"]) else float("nan")
    return (
        f"Drug: {row['drugname']} | Event: {row['pt']} | "
        f"Total co-reports: {row['a_drug_and_event']} | "
        f"PRR: {prr:.3f} | ROR: {ror:.3f} | "
        f"Serious reports: {row['serious_reports']}"
    )


def _build_signal_documents() -> list[Document]:
    """Read the pre-aggregated signal summary (one row per drug-event pair,
    produced at ingestion time) and convert each row into a LangChain Document
    for FAISS/BM25 indexing.

    The signal summary is a small file (100k-500k rows) that fits in RAM,
    unlike the full flat parquet (143M+ rows). PRR/ROR in these documents
    were computed by the ingestion pipeline from real FAERS data; the stats
    code tool provides fresh on-demand lookup against the same summary."""
    summary_df = load_signal_summary()
    summary_df = summary_df[summary_df["a_drug_and_event"] >= MIN_SIGNAL_COOCCURRENCE]
    documents = []
    for _, row in summary_df.iterrows():
        text = _row_to_signal_text(row)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "drug": row["drugname"],
                    "event": row["pt"],
                    "source": "faers_signal",
                },
            )
        )
    return documents


def _longest_contained(names, text: str) -> str | None:
    """Return the longest vocabulary string contained in `text`, or None."""
    hits = [n for n in names if n and n in text]
    if not hits:
        return None
    return max(hits, key=lambda s: (len(s), s))


def _not_indexed_signal_doc(drug: str, event: str) -> Document:
    return Document(
        page_content=(
            f"No indexed signal-summary document for {drug} / {event}. "
            f"Pairs with fewer than {MIN_SIGNAL_COOCCURRENCE} co-reports are omitted "
            "from this semantic index. Use calculate_pv_statistics on signal_df "
            "to look up PRR, ROR, case counts, and serious reports for this pair."
        ),
        metadata={"drug": drug, "event": event, "source": "faers_signal"},
    )


def _exact_match_for_signals(documents: list[Document]):
    """Returns a closure: given a query, if it names a drug+event pair,
    return that indexed document (or a not-indexed message) so a different
    pair cannot be treated as evidence."""
    by_pair = {(doc.metadata["drug"], doc.metadata["event"]): doc for doc in documents}
    try:
        summary_df = load_signal_summary()
        drug_names = set(summary_df["drugname"].dropna().astype(str).str.upper())
        event_names = set(summary_df["pt"].dropna().astype(str).str.upper())
    except Exception:
        drug_names = {drug for drug, _ in by_pair}
        event_names = {event for _, event in by_pair}

    def _match(query: str) -> Document | None:
        query_upper = query.upper()
        found_drug = _longest_contained(drug_names, query_upper)
        found_event = _longest_contained(event_names, query_upper)
        if found_drug and found_event:
            return by_pair.get((found_drug, found_event)) or _not_indexed_signal_doc(
                found_drug, found_event
            )
        if found_drug:
            same_drug = [d for d in documents if d.metadata.get("drug") == found_drug]
            if not same_drug:
                return _not_indexed_signal_doc(found_drug, found_event or "(unresolved event)")
        return None

    return _match


def build_signal_retriever() -> HybridRetriever:
    """Build (or load, if already saved) the FAISS index over per-pair
    signal-summary documents, plus a BM25 index over the same documents,
    and wrap both in a HybridRetriever with the exact-match override."""
    embeddings = get_embeddings()
    index_file = SIGNALS_INDEX_DIR / "index.faiss"

    if index_file.exists():
        vector_store = FAISS.load_local(
            str(SIGNALS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        documents = list(vector_store.docstore._dict.values())
    else:
        documents = _build_signal_documents()
        if not documents:
            raise RuntimeError(
                "No signal documents to index. Run `python -m "
                "src.ingestion.faers_ingest` first so there is FAERS data "
                "to summarize."
            )
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(str(SIGNALS_INDEX_DIR))

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = DEFAULT_TOP_K_SPARSE

    return HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        exact_match_fn=_exact_match_for_signals(documents),
    )
