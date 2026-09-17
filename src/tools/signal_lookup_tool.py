"""Exact/fuzzy drug and event name resolution against the known FAERS
vocabulary, so a query like "Depo Provera" or "meningioma" resolves to the
exact stored strings ("DEPO-PROVERA", "MENINGIOMA") before the stats tool
or retrievers are called. Prevents silent zero-result lookups caused by
casing/punctuation mismatches rather than a genuinely absent drug/event.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool
from rapidfuzz import process, fuzz

from src.ingestion.faers_ingest import load_signal_summary

_FUZZY_SCORE_CUTOFF = 80  # 0-100; below this we report "no confident match"


@lru_cache(maxsize=1)
def _vocab() -> tuple[list[str], list[str]]:
    """Cached distinct drug names and event terms from the signal summary.
    Uses the summary (small, fits in RAM) rather than the full flat table."""
    summary_df = load_signal_summary()
    drugs = sorted(summary_df["drugname"].dropna().unique().tolist())
    events = sorted(summary_df["pt"].dropna().unique().tolist())
    return drugs, events


def _resolve(name: str, vocabulary: list[str]) -> dict:
    name_norm = name.strip().upper()
    if name_norm in vocabulary:
        return {"query": name, "resolved": name_norm, "match_type": "exact", "score": 100}

    match = process.extractOne(name_norm, vocabulary, scorer=fuzz.WRatio)
    if match is None:
        return {"query": name, "resolved": None, "match_type": "none", "score": 0}

    candidate, score, _ = match
    if score >= _FUZZY_SCORE_CUTOFF:
        return {"query": name, "resolved": candidate, "match_type": "fuzzy", "score": score}
    return {"query": name, "resolved": None, "match_type": "none", "score": score}


@tool
def resolve_drug_name(name: str) -> str:
    """Resolve a user-provided drug name to the exact drug name string used
    in the FAERS data (e.g. 'depo provera' -> 'DEPO-PROVERA'). Call this
    before searching or calculating statistics for a drug whose exact FAERS
    spelling you are not certain of."""
    drugs, _ = _vocab()
    result = _resolve(name, drugs)
    if result["resolved"] is None:
        return f"No confident match for drug '{name}' (best score {result['score']}). Ask the user to confirm spelling."
    return f"Resolved '{name}' -> '{result['resolved']}' ({result['match_type']} match, score {result['score']})."


@tool
def resolve_event_name(name: str) -> str:
    """Resolve a user-provided adverse-event term to the exact MedDRA
    preferred term string used in the FAERS data (e.g. 'meningioma' ->
    'MENINGIOMA'). Call this before searching or calculating statistics for
    an event whose exact FAERS spelling you are not certain of."""
    _, events = _vocab()
    result = _resolve(name, events)
    if result["resolved"] is None:
        return f"No confident match for event '{name}' (best score {result['score']}). Ask the user to confirm spelling."
    return f"Resolved '{name}' -> '{result['resolved']}' ({result['match_type']} match, score {result['score']})."
