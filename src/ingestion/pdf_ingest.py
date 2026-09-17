"""PyPDFLoader + RecursiveCharacterTextSplitter for label and literature
PDFs. Splits each PDF into overlapping text chunks with source metadata
(file name, page number) so retrieved chunks can be traced back to the
original document.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, LABELS_PDF_DIR, LITERATURE_PDF_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_and_chunk_pdf(pdf_path: Path, doc_type: str) -> list[Document]:
    """Load one PDF and split it into chunked Documents, tagging each with
    doc_type ('label' or 'literature') and the source file name so results
    can cite exactly where they came from."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    chunks = _splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["source_file"] = pdf_path.name
    return chunks


def ingest_pdf_directory(directory: Path, doc_type: str) -> list[Document]:
    """Load + chunk every PDF in a directory. Returns an empty list (with a
    warning, not an error) if the directory has no PDFs yet, so callers can
    treat 'no documents ingested' as a normal, skippable state."""
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s -- skipping %s ingestion.", directory, doc_type)
        return []

    all_chunks: list[Document] = []
    for pdf_path in pdf_paths:
        logger.info("Chunking %s (%s)", pdf_path.name, doc_type)
        all_chunks.extend(load_and_chunk_pdf(pdf_path, doc_type))
    return all_chunks


def ingest_labels() -> list[Document]:
    return ingest_pdf_directory(LABELS_PDF_DIR, doc_type="label")


def ingest_literature() -> list[Document]:
    return ingest_pdf_directory(LITERATURE_PDF_DIR, doc_type="literature")
