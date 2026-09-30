"""Paths, constants, and supported model/quarter lists shared across the
project. Every other module reads paths and defaults from here instead of
hardcoding strings, so moving data or swapping a default model is a
one-file change.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

EXTERNAL_DOCS_DIR = DATA_DIR / "external_docs"
LABELS_PDF_DIR = EXTERNAL_DOCS_DIR / "labels"
LITERATURE_PDF_DIR = EXTERNAL_DOCS_DIR / "literature"

VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
SIGNALS_INDEX_DIR = VECTOR_STORE_DIR / "signals"
LABELS_INDEX_DIR = VECTOR_STORE_DIR / "labels"
LITERATURE_INDEX_DIR = VECTOR_STORE_DIR / "literature"

for _dir in (
    RAW_DIR,
    PROCESSED_DIR,
    LABELS_PDF_DIR,
    LITERATURE_PDF_DIR,
    SIGNALS_INDEX_DIR,
    LABELS_INDEX_DIR,
    LITERATURE_INDEX_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# Flattened FAERS table produced once by ingestion/faers_ingest.py
FLAT_PARQUET_PATH = PROCESSED_DIR / "faers_flat.parquet"
DEMO_PARQUET_PATH = PROCESSED_DIR / "demo.parquet"
DRUG_PARQUET_PATH = PROCESSED_DIR / "drug.parquet"
REAC_PARQUET_PATH = PROCESSED_DIR / "reac.parquet"
OUTC_PARQUET_PATH = PROCESSED_DIR / "outc.parquet"

# Pre-aggregated signal summary: one row per (drug, event) pair with PRR/ROR
# pre-computed at ingestion time. Much smaller than flat_df (100k-500k rows vs
# 143M rows) -- used by the signal retriever and the stats code tool so neither
# needs to load the full flat parquet into RAM.
SIGNAL_SUMMARY_PATH = PROCESSED_DIR / "signal_summary.parquet"

# Minimum co-occurrence threshold: pairs with fewer unique reports than this
# are excluded from the signal summary and FAISS index (too sparse to be useful).
MIN_SIGNAL_COOCCURRENCE = 500

# --------------------------------------------------------------------------
# FAERS quarters currently ingested (edit as you add more raw files)
# e.g. "2023Q1", "2023Q2" -- matches the FDA FAERS ASCII file naming
# --------------------------------------------------------------------------
SUPPORTED_QUARTERS: list[str] = [
    "2026Q1",
    "2026Q2",
]

# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
# Default provider: Groq (hosted, fast, first-class LangChain tool-calling
# support via langchain-groq -- inference runs on Groq's hardware, not the
# local GPU, which matters on a low-VRAM laptop). "ollama" and "vllm"
# remain available -- see src/agent/llm_provider.py::build_llm.
DEFAULT_LLM_PROVIDER = "groq"

OLLAMA_BASE_URL = "http://localhost:11434"

# OpenAI-compatible external providers (each uses ChatOpenAI with a different base_url).
OPENAI_COMPAT_BASE_URLS: dict[str, str] = {
    "sambanova": "https://api.sambanova.ai/v1",
    "cerebras":  "https://api.cerebras.ai/v1",
    "mistral":   "https://api.mistral.ai/v1",
}
# Env-var name for each provider's API key.
OPENAI_COMPAT_KEY_ENV: dict[str, str] = {
    "sambanova": "SAMBANOVA_API_KEY",
    "cerebras":  "CEREBRAS_API_KEY",
    "mistral":   "MISTRAL_API_KEY",
}

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"

# Hosted models shown in the Streamlit dropdown.
# Add a model here and it will appear automatically — no other changes needed.
# Gemini requires a full LangGraph migration; see llm_provider.py for details.
MODEL_PROVIDERS: dict[str, str] = {
    "openai/gpt-oss-120b":              "groq",
    "qwen/qwen3.8-27b":                 "groq",
    "openai/gpt-oss-20b":               "groq",
    "ministral-8b-latest":              "mistral",     # free tier, fast, tool-calling verified
    "ministral-14b-latest":             "mistral",     # free tier, stronger, tool-calling verified
    # SambaNova + Cerebras require adding a payment method on their console first:
    # "Meta-Llama-3.3-70B-Instruct":   "sambanova",
    # "DeepSeek-V3.2":                 "sambanova",
    # "qwen-3.8-27b":                  "cerebras",
}
CANDIDATE_LLM_MODELS: list[str] = list(MODEL_PROVIDERS.keys())


def provider_for_model(model: str) -> str:
    return MODEL_PROVIDERS.get(model, DEFAULT_LLM_PROVIDER)

# Extra local tags (not in the UI dropdown). Pull with `ollama pull <tag>`.
LOCAL_CANDIDATE_LLM_MODELS: list[str] = [
    "qwen2.5:3b",
    "llama3.1:8b",
    "qwen2.5:7b-instruct",
    "mistral:7b-instruct-v0.3",
    "phi3.5:3.8b-mini-instruct",
]

LLM_TEMPERATURE = 0.0

# --------------------------------------------------------------------------
# Token budget — controlled by DEV_MODE in .env
# --------------------------------------------------------------------------
# DEV_MODE=true  → 300 tokens   quick test: did the pipeline call tools correctly?
# DEV_MODE=false → 1024 tokens  real use:   full structured PV answers
# RAGAS eval always uses 2048 regardless (set directly in ragas_eval.py)
import os as _os
_dev = _os.getenv("DEV_MODE", "false").strip().lower() in ("1", "true", "yes")
LLM_MAX_TOKENS: int = 300 if _dev else 1024
DEV_MODE: bool = _dev
del _os, _dev

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------
# fastembed model name -- BAAI/bge-small-en-v1.5 is natively supported by fastembed
# and slightly stronger than all-MiniLM-L6-v2 for retrieval tasks.
# fastembed downloads the ONNX model on first use (~50MB, cached in ~/.cache/fastembed).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --------------------------------------------------------------------------
# Reranker (cross-encoder)
# --------------------------------------------------------------------------
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------------------------------------
# Retrieval tuning
# --------------------------------------------------------------------------
DEFAULT_TOP_K_DENSE = 10        # candidates pulled from FAISS before fusion
DEFAULT_TOP_K_SPARSE = 10       # candidates pulled from BM25 before fusion
DEFAULT_TOP_K_FUSED = 10        # candidates kept after RRF fusion, pre-rerank
DEFAULT_TOP_K_FINAL = 3         # final results shown to the agent/UI after rerank
RRF_K = 60                      # standard Reciprocal Rank Fusion constant

# --------------------------------------------------------------------------
# Chunking (PDF ingestion)
# --------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------
DISCLAIMER = (
    "This tool surfaces reporting patterns in spontaneous adverse-event data "
    "(e.g. FAERS). Disproportionality signals such as PRR/ROR indicate a "
    "statistical association in reporting rates, not a confirmed causal "
    "relationship. All outputs require review by a qualified pharmacovigilance "
    "professional before any regulatory or clinical action."
)
