"""One-time/refreshable script: builds and saves all three FAISS indexes to
disk (signals, labels, literature). Run this after faers_ingest.py, and
again any time you add new label/literature PDFs.

Usage:
    python -m src.ingestion.build_vector_stores
"""

from __future__ import annotations

import logging

from langchain_community.vectorstores import FAISS

from src.config import LABELS_INDEX_DIR, LITERATURE_INDEX_DIR, SIGNALS_INDEX_DIR
from src.ingestion.pdf_ingest import ingest_labels, ingest_literature
from src.retrieval.signal_retriever import _build_signal_documents, get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_signals_index() -> None:
    logger.info("Loading embedding model...")
    embeddings = get_embeddings()
    logger.info("Building signal documents from summary...")
    documents = _build_signal_documents()
    if not documents:
        logger.warning("No signal documents to index -- run faers_ingest.py first.")
        return
    logger.info("Embedding %d signal documents into FAISS (this takes a few minutes)...", len(documents))
    store = FAISS.from_documents(documents, embeddings)
    logger.info("Saving signals index to disk...")
    store.save_local(str(SIGNALS_INDEX_DIR))
    logger.info("Saved signals index (%d documents) to %s", len(documents), SIGNALS_INDEX_DIR)


def build_labels_index() -> None:
    logger.info("Ingesting label PDFs...")
    embeddings = get_embeddings()
    documents = ingest_labels()
    if not documents:
        logger.warning("No label PDFs found -- skipping labels index.")
        return
    logger.info("Embedding %d label chunks into FAISS...", len(documents))
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(str(LABELS_INDEX_DIR))
    logger.info("Saved labels index (%d chunks) to %s", len(documents), LABELS_INDEX_DIR)


def build_literature_index() -> None:
    logger.info("Ingesting literature PDFs...")
    embeddings = get_embeddings()
    documents = ingest_literature()
    if not documents:
        logger.warning("No literature PDFs found -- skipping literature index.")
        return
    logger.info("Embedding %d literature chunks into FAISS...", len(documents))
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(str(LITERATURE_INDEX_DIR))
    logger.info("Saved literature index (%d chunks) to %s", len(documents), LITERATURE_INDEX_DIR)


def build_all() -> None:
    build_signals_index()
    build_labels_index()
    build_literature_index()


if __name__ == "__main__":
    build_all()
