"""PV Signal Console — Streamlit UI (simplified layout)

Three-column layout:
  LEFT   — Query panel (drug + event inputs, free query, session history)
  CENTRE — Results
             Signal lookup : status chips, 4 KPI cards, screening checks,
                             evidence cards, 3-bullet summary
             Query         : short answer, source cards, scope note,
                             "Run full signal analysis" button
  RIGHT  — Evaluation verdict + collapsed audit trail

Design rules
  * Only numbers the stats tool really returns are shown (no 2x2 table,
    no CI / chi-square).
  * Neutral wording — no "STRONG SIGNAL" banner, no causal language.
  * One disclaimer, in the page footer.
  * Evidence cards show only complete, readable, event-related sentences.
"""

from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
from rapidfuzz import process as fuzzy_process

from src.agent.build_agent import build_pv_agent
from src.config import (
    DEFAULT_LLM_MODEL,
    provider_for_model,
)
from src.guardrails.llm_judge import run_llm_judge
from src.guardrails.validators import run_guardrails, extract_numbers, _numbers_match
from src.ingestion.faers_ingest import load_signal_summary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODE_SIGNAL = "Signal lookup (drug + event)"
MODE_QUERY = "Query"
MAX_EVIDENCE_CARDS = 3
UI_LLM_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "ministral-14b-latest",
]
FOOTER_TEXT = (
    "Spontaneous-report association only; it does not establish causality. "
    "All outputs require qualified PV review before any regulatory or clinical action."
)

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PV Signal Console",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; }

/* Top bar */
.top-bar {
    background:#0d1b4b; color:#b3d1ff; padding:8px 20px; font-size:13px;
    font-weight:600; display:flex; align-items:center;
    justify-content:space-between; border-radius:0 0 6px 6px; margin-bottom:14px;
}
.top-bar-title { color:#fff; font-size:14px; }
.top-bar-meta  { font-size:11px; color:#7da8e8; }

/* Left panel */
.panel-title {
    font-size:11px; font-weight:700; color:#7da8e8; text-transform:uppercase;
    letter-spacing:1.2px; margin-bottom:6px; border-bottom:1px solid #e8eaf6;
    padding-bottom:4px;
}
.section-label {
    font-size:11px; color:#aaa; text-transform:uppercase;
    letter-spacing:.9px; margin:10px 0 3px;
}

/* Titles */
.result-title { font-size:22px; font-weight:800; color:#0d1b4b; margin:4px 0 2px; }
.result-query { font-size:12px; color:#888; margin-bottom:10px; }
.section-header {
    font-size:13px; font-weight:700; color:#5c6785; margin:18px 0 8px;
}

/* Status chips */
.chip-row { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 4px; }
.chip {
    font-size:12px; padding:3px 11px; border-radius:99px; font-weight:600;
}
.chip-info    { background:#e3f0ff; color:#0d47a1; }
.chip-ok      { background:#e6f4ea; color:#1b5e20; }
.chip-neutral { background:#eef0f4; color:#4a5568; }

/* Signal-strength banner (traffic light) */
.str-banner {
    display:flex; align-items:baseline; justify-content:space-between; gap:12px;
    padding:10px 14px; border-radius:8px; margin:8px 0 10px;
    font-weight:800; font-size:16px; letter-spacing:.02em;
}
.str-sub { font-weight:600; font-size:12px; opacity:.9; }
.str-high { background:#fdecea; color:#b71c1c; border:1px solid #ef9a9a; }
.str-mid  { background:#fff3e0; color:#e65100; border:1px solid #ffcc80; }
.str-low  { background:#fffde7; color:#f57f17; border:1px solid #ffe082; }
.str-none { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }
.str-val-high { color:#b71c1c; font-weight:700; }
.str-val-mid  { color:#e65100; font-weight:700; }
.str-val-low  { color:#f57f17; font-weight:700; }
.str-val-none { color:#2e7d32; font-weight:700; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:10px; }
.kpi { background:#f5f7fb; border-radius:8px; padding:10px 12px; }
.kpi-label { font-size:12px; color:#667; }
.kpi-value { font-size:22px; font-weight:700; color:#0d1b4b; margin-top:2px; }
.kpi-sub   { font-size:11px; color:#99a; }

/* Screening rows */
.chk-row {
    display:flex; justify-content:space-between; font-size:13px;
    padding:7px 0; border-top:1px solid #eef0f4;
}
.chk-ok  { color:#2e7d32; font-weight:700; }
.chk-bad { color:#c62828; font-weight:700; }

/* Evidence cards */
.ev-card {
    border:1px solid #e3e8f5; border-radius:8px; padding:10px 14px;
    margin-bottom:8px; font-size:13.5px; background:#fff;
}
.ev-head { font-size:12px; color:#667; margin-bottom:4px; font-weight:600; }
.ev-links { font-size:11px; margin-top:6px; }
.ev-links a { color:#1565c0; text-decoration:none; }
.ev-card mark { background:#fff3b0; color:inherit; padding:0 2px; border-radius:2px; }

/* Answer + notes */
.answer-box {
    font-size:14.5px; line-height:1.65; color:#1a2238;
    background:#fff; border:1px solid #e3e8f5; border-radius:8px; padding:12px 16px;
}
.note-box {
    background:#f5f7fb; border-radius:8px; padding:9px 14px;
    font-size:12.5px; color:#4a5568; margin:12px 0 8px;
}

/* Guardrail failure (shown only when a check fails) */
.gr-fail {
    background:#fce4ec; border:1px solid #ef9a9a; border-radius:6px;
    padding:9px 16px; font-size:12.5px; color:#b71c1c; margin-top:12px;
}

/* Audit trail */
.audit-step {
    border-left:3px solid #1565c0; padding:8px 10px 8px 12px;
    margin-bottom:10px; background:#f8f9ff; border-radius:0 6px 6px 0;
}
.audit-tool { font-weight:700; font-size:12px; color:#1565c0; }
.audit-time { font-size:10px; color:#aaa; float:right; }
.audit-io   { font-size:11px; color:#555; margin-top:4px; word-break:break-word; }

/* Footer (single disclaimer) */
.foot {
    border-top:1px solid #e8eaf6; margin-top:28px; padding-top:10px;
    font-size:12px; color:#7b849a;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent(model: str):
    return build_pv_agent(model=model, provider=provider_for_model(model))


@st.cache_data
def get_signal_df() -> pd.DataFrame:
    return load_signal_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — lookup and stats
# ─────────────────────────────────────────────────────────────────────────────
def lookup_pair(df, drug_q, event_q):
    dm = fuzzy_process.extractOne(drug_q.upper(), df["drugname"].unique(), score_cutoff=70)
    em = fuzzy_process.extractOne(event_q.upper(), df["pt"].unique(), score_cutoff=70)
    if dm is None or em is None:
        return None, dm, em
    row = df[(df["drugname"] == dm[0]) & (df["pt"] == em[0])]
    return (row.iloc[0] if not row.empty else None), dm, em


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _stats_from_row(row) -> dict | None:
    if row is None:
        return None
    try:
        return {
            "prr": float(row["prr"]),
            "ror": float(row["ror"]),
            "a": int(row["a_drug_and_event"]),
            "serious": int(row["serious_reports"]),
            "total": int(row["total_reports"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _stats_from_steps(steps) -> dict | None:
    """Fallback: read the stats tool output when the pair is not in the summary table."""
    for action, obs in steps or []:
        if getattr(action, "tool", None) != "calculate_pv_statistics":
            continue
        lines = [ln.strip() for ln in str(obs).splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        header = re.split(r"\s+", lines[0].lower())
        vals = re.split(r"\s+", lines[-1])
        offset = max(len(vals) - len(header), 0)

        def _col(name: str) -> str | None:
            if name not in header:
                return None
            j = header.index(name) + offset
            return vals[j] if 0 <= j < len(vals) else None

        try:
            return {
                "prr": float(_col("prr")),
                "ror": float(_col("ror")),
                "a": int(float(_col("a_drug_and_event"))),
                "serious": int(float(_col("serious_reports"))),
                "total": None,
            }
        except (TypeError, ValueError):
            continue
    return None


def screening(stats: dict) -> dict:
    """Neutral screening result: N >= 3 and PRR >= 2."""
    n_ok = stats["a"] >= 3
    prr_ok = stats["prr"] >= 2
    return {"n_ok": n_ok, "prr_ok": prr_ok, "met": n_ok and prr_ok}


def signal_strength(stats: dict) -> dict:
    """PRR band as reporting strength (not causality). Evans-style cutoffs."""
    prr = stats["prr"]
    if prr >= 10:
        return {"label": "High", "band": "PRR ≥ 10", "banner": "str-high", "val": "str-val-high"}
    if prr >= 5:
        return {"label": "Moderate", "band": "PRR 5–10", "banner": "str-mid", "val": "str-val-mid"}
    if prr >= 2:
        return {"label": "Low", "band": "PRR 2–5", "banner": "str-low", "val": "str-val-low"}
    return {"label": "Not elevated", "band": "PRR < 2", "banner": "str-none", "val": "str-val-none"}


# Fixed FAERS → DailyMed search aliases (demo drugs only). Not LLM-generated.
DAILYMED_MAP = {
    "DEPO-PROVERA": "medroxyprogesterone",
    "WARFARIN": "warfarin",
    "COUMADIN": "warfarin",
    "AMIODARONE": "amiodarone",
    "CORDARONE": "amiodarone",
}


def dailymed_url(drug: str) -> str:
    term = DAILYMED_MAP.get(drug.upper(), drug)
    return f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={quote(term)}"


def pubmed_url(drug: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={quote(drug)}+adverse+event"


def _cite_links_html(drug: str) -> str:
    dm = html.escape(dailymed_url(drug), quote=True)
    pm = html.escape(pubmed_url(drug), quote=True)
    return (
        f'<div class="ev-links">'
        f'<a href="{dm}" target="_blank" rel="noopener">DailyMed</a> · '
        f'<a href="{pm}" target="_blank" rel="noopener">PubMed</a>'
        f"</div>"
    )


def _drug_from_text(text: str | None) -> str | None:
    """Match a known demo drug in free text (query or tool output). Not LLM-generated."""
    if not text:
        return None
    folded = _fold_med(text)
    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for brand, generic in DAILYMED_MAP.items():
        for raw in (brand, generic):
            key = _fold_med(raw)
            if len(key) >= 5 and key in folded and brand not in seen:
                hits.append((len(key), brand))
                seen.add(brand)
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][1]


def _drug_from_steps(steps) -> str | None:
    for action, obs in steps or []:
        if action.tool == "resolve_drug_name":
            m = re.search(r"->\s*'([^']+)'", str(obs))
            if m:
                return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — evidence
# ─────────────────────────────────────────────────────────────────────────────
_SECTION_RE = re.compile(r"^\(?(\d+(?:\.\d+)+)\)?\s*[•\-–:]?\s*")
_LEADING_SEC_RE = re.compile(r"^\(?(\d+\.\d+)\)?\s*[•\-–:]?\s*")
_TRAILING_SEC_RE = re.compile(r"\((\d+\.\d+)\)\s*$")
_BOILERPLATE_RE = re.compile(
    r"these highlights do not include|see full prescribing information|"
    r"this label may not be the latest|initial u\.s\. approval|"
    r"reference id:|dosage forms? and strengths|single-scored tablets|"
    r"recent major changes|highlights of prescribing information|"
    r"indications and usage|dosage and administration|"
    r"warnings and precautions\s*,",
    flags=re.I,
)
# Instruction / finding language — TOC headers do not match this.
_CLINICAL_RE = re.compile(
    r"\b("
    r"discontinue|monitor|reported|report|should|must|may not|may cause|"
    r"avoid|contraindicat|diagnos|symptom|patients?|fatal|serious|"
    r"occur|observ|recommend|advise|do not|cases? of|associated|"
    r"increas|impair|death|toxic|pneumonitis|bleed|h[ae]morrhag|"
    r"interstitial|hypersensitiv"
    r")\b",
    flags=re.I,
)
# Label wording that should still count as the MedDRA PT.
_EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "HAEMORRHAGE": ("HEMORRHAGE", "HEMORRHAGIC", "HAEMORRHAGIC", "BLEEDING", "BLEED"),
    "HEMORRHAGE": ("HAEMORRHAGE", "HEMORRHAGIC", "HAEMORRHAGIC", "BLEEDING", "BLEED"),
    "PULMONARYTOXICITY": ("PULMONARY", "PNEUMONITIS", "INTERSTITIALLUNG"),
}


def parse_evidence_items(raw: str) -> list[dict]:
    """Parse the '[relevance X.XXX] ...' blocks returned by retriever tools."""
    items = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        rel = None
        if block.startswith("[relevance"):
            try:
                rel = float(block[block.index(" ") + 1: block.index("]")])
                block = block[block.index("]") + 1:].strip()
            except Exception:
                pass
        block = re.sub(r"^\[(?:[^\]]+\.pdf)\]\s*", "", block, flags=re.IGNORECASE)
        items.append({"text": block, "relevance": rel})
    return items


def _split_section(text: str) -> tuple[str | None, str]:
    """'5.4 Meningioma Cases of ...' -> ('5.4', 'Meningioma Cases of ...')."""
    text = re.sub(r"\s+", " ", text).strip()
    m = _SECTION_RE.match(text)
    if m:
        return m.group(1), text[m.end():]
    return None, text


def _section_for_sentence(block: str, sent: str, event: str | None) -> str | None:
    """Section that belongs to this event sentence, not a neighbouring bullet.

    Two FDA layouts appear in chunks:
      W&P body:     '5.4 Meningioma Cases of ...'
      Highlights:   'Meningioma: Discontinue … (5.4) • Ectopic …'
    A parenthesised number *before* the event name is often the previous
    item's citation, e.g. 'cancer carefully. (5.3) • Meningioma'.
    """
    block_n = re.sub(r"\s+", " ", block)
    if event:
        m_wp = re.search(
            rf"(?<![\(\d])(\d+\.\d+)\s+{re.escape(event)}\b",
            block_n,
            flags=re.I,
        )
        if m_wp:
            return m_wp.group(1)
        m_ev = re.search(re.escape(event), block_n, flags=re.I)
        if m_ev:
            rest = block_n[m_ev.end():]
            m_par = re.search(r"\((\d+\.\d+)\)", rest)
            if m_par and m_par.start() < 280:
                return m_par.group(1)
    loc = block_n.lower().find(sent[:48].lower())
    if loc >= 0:
        m_after = re.search(r"\((\d+\.\d+)\)", block_n[loc + len(sent): loc + len(sent) + 24])
        if m_after:
            return m_after.group(1)
    return None


def _clean_evidence_sentence(sent: str) -> str:
    sent = _LEADING_SEC_RE.sub("", sent).strip(" •-\t")
    sent = _TRAILING_SEC_RE.sub("", sent).strip(" •-\t")
    return sent


def _fold_med(text: str) -> str:
    """Uppercase letters/digits only, AE/OE folded so HAEMORRHAGE == HEMORRHAGE."""
    return re.sub(r"[^A-Z0-9]+", "", text.upper()).replace("AE", "E").replace("OE", "E")


def _stems(token: str) -> list[str]:
    folded = _fold_med(token)
    if len(folded) < 4:
        return []
    out = {folded}
    for suf in ("ING", "TIONS", "TION", "SION", "ITY", "IES", "ED", "ES", "S", "AGE", "IA"):
        if len(folded) > len(suf) + 4 and folded.endswith(suf):
            out.add(folded[: -len(suf)])
    return [s for s in out if len(s) >= 4]


def _needles(*terms: str | None) -> list[str]:
    found: set[str] = set()
    for term in terms:
        if not term:
            continue
        parts = [term, *re.split(r"[^A-Za-z0-9]+", term)]
        for part in parts:
            if len(part) < 4:
                continue
            found.update(_stems(part))
            found.add(_fold_med(part))
            found.update(_EVENT_ALIASES.get(_fold_med(part), ()))
            found.update(_EVENT_ALIASES.get(_fold_med(term), ()))
    return sorted({n for n in found if len(n) >= 4}, key=len, reverse=True)


def _related(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    blob = _fold_med(text)
    return any(n in blob for n in needles)


def _is_boilerplate(text: str) -> bool:
    if _BOILERPLATE_RE.search(text):
        return True
    if text.count("-") >= 8 or re.search(r"-{4,}", text):
        return True
    if re.search(r"\b\d{1,2}/\d{4}\b", text) and re.search(r"\(\d+\.\d+\)", text):
        return True
    return False


def _is_heading_only(text: str) -> bool:
    t = re.sub(r"\s+", " ", text).strip(" •-\t:")
    if len(t) < 80 and re.fullmatch(
        r"(?:\d+\.\d+\s+)?[A-Za-z][A-Za-z0-9 \-/]*(?:\s*\(\d+\.\d+\))?",
        t,
    ):
        return True
    letters = re.sub(r"[^A-Za-z]", "", t)
    if letters and len(t) < 120:
        upper = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper > 0.85:
            return True
    return False


def _is_displayable(text: str) -> bool:
    """True only for a complete clinical sentence, not a TOC / header hit."""
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) < 28 or _is_boilerplate(t) or _is_heading_only(t):
        return False
    if len(re.findall(r"[A-Za-z]{2,}", t)) < 6:
        return False
    return bool(_CLINICAL_RE.search(t))


def _usefulness(text: str) -> int:
    score = 0
    if re.search(
        r"\b(discontinue|monitor|contraindicat|fatal|diagnos|do not)\b",
        text,
        flags=re.I,
    ):
        score += 4
    if re.search(r"\b(reported|cases? of|risk|should|avoid|toxic)\b", text, flags=re.I):
        score += 2
    if text.rstrip().endswith((".", "!", "?")):
        score += 1
    return score


def _snippets(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(
        r"(?<=[.!?])\s+|\s*[•●▪]\s+|(?<=\))\s+(?=[A-Z•])|\s*-{3,}\s*",
        text,
    )
    return [p.strip(" \t-–") for p in parts if p.strip()]


def _excerpt_around(text: str, needles: list[str], width: int = 520) -> str:
    """If no clean sentence survived, keep a readable window around the first related term."""
    text_n = re.sub(r"\s+", " ", text)
    for n in needles:
        flex = re.escape(n).replace("AE", "[AE]?").replace("OE", "[OE]?")
        m = re.search(flex + r"\w*", text_n, flags=re.I)
        if not m:
            continue
        start = max(0, m.start() - 80)
        end = min(len(text_n), m.end() + width)
        # Prefer full sentences inside the window.
        window = text_n[start:end]
        first_dot = window.find(". ")
        if start > 0 and 0 <= first_dot < 80:
            window = window[first_dot + 2 :]
        last_dot = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if last_dot >= 40:
            window = window[: last_dot + 1]
        chunk = window.strip(" •-\t")
        if start > 0:
            chunk = "…" + chunk
        if end < len(text_n) and not chunk.endswith((".", "?", "!")):
            chunk = chunk.rstrip(".,;:") + "…"
        return _clean_evidence_sentence(chunk)
    return ""


def _key_sentences(text: str, event: str | None, extra: list[str] | None = None) -> list[str]:
    """Keep retrieved PDF text that is related to the event/drug, including fragments."""
    text = re.sub(r"\s+", " ", text).strip()
    event_needles = _needles(event)
    extra_needles = _needles(*(extra or []))
    kept: list[str] = []
    parts = [_clean_evidence_sentence(s) for s in _snippets(text)]
    i = 0
    while i < len(parts):
        sent = parts[i]
        if len(sent) < 24 or _is_boilerplate(sent):
            i += 1
            continue
        if event_needles:
            if not _related(sent, event_needles):
                i += 1
                continue
        elif extra_needles and not _related(sent, extra_needles):
            i += 1
            continue
        piece = sent
        j = i + 1
        while j < len(parts) and len(piece) < 480:
            nxt = parts[j]
            if len(nxt) < 20 or _is_boilerplate(nxt) or _is_heading_only(nxt):
                break
            follow_on = (
                (event_needles and _related(nxt, event_needles) and _is_displayable(nxt))
                or nxt[:1].islower()
                or nxt.startswith(("If ", "Do not", "Patients ", "Monitor ", "Discontinue "))
            )
            if not follow_on:
                break
            piece = (piece.rstrip(". ") + ". " + nxt).strip()
            j += 1
        if _is_displayable(piece):
            kept.append(piece)
        i = j
    kept.sort(key=_usefulness, reverse=True)
    if kept:
        return kept
    fallback_needles = event_needles or extra_needles
    if fallback_needles and _related(text, fallback_needles) and not _is_boilerplate(text):
        excerpt = _excerpt_around(text, fallback_needles)
        if excerpt and _is_displayable(excerpt):
            return [excerpt]
    return []


def _summarize_evidence_items(
    items: list[dict],
    event: str | None,
    extra: list[str] | None = None,
    limit: int = MAX_EVIDENCE_CARDS,
) -> list[dict]:
    """De-duplicate overlapping chunks -> up to `limit` clinical evidence cards."""
    seen: list[str] = []
    candidates: list[dict] = []
    for item in items:
        _fallback_sec, body = _split_section(item["text"])
        if event:
            body = re.sub(rf"^{re.escape(event)}\s*:?\s+(?=[A-Z])", "", body, flags=re.I)
        for sent in _key_sentences(body, event, extra):
            sent = _clean_evidence_sentence(sent)
            if not _is_displayable(sent):
                continue
            fp = re.sub(r"[^A-Z0-9]+", "", sent.upper())[:72]
            if any(fp[:50] in prev or prev[:50] in fp for prev in seen):
                continue
            seen.append(fp)
            section = _section_for_sentence(item["text"], sent, event)
            if not section and event and _fallback_sec:
                head = re.sub(r"\s+", " ", item["text"]).strip()
                if re.match(
                    rf"^{re.escape(_fallback_sec)}\s+{re.escape(event)}\b",
                    head,
                    flags=re.I,
                ):
                    section = _fallback_sec
            candidates.append(
                {
                    "text": sent,
                    "section": section,
                    "relevance": item.get("relevance"),
                    "score": _usefulness(sent),
                }
            )
    candidates.sort(
        key=lambda c: (c["score"], c.get("relevance") or 0),
        reverse=True,
    )
    out = []
    for c in candidates[:limit]:
        c.pop("score", None)
        out.append(c)
    return out


def _highlight(text: str, event: str | None) -> str:
    safe = html.escape(text)
    tokens = [t for t in _needles(event) if t.isalpha() and len(t) >= 4]
    for t in tokens:
        safe = re.sub(rf"({re.escape(t)}\w*)", r"<mark>\1</mark>", safe, flags=re.I)
    # US/UK: highlight "hemorrhag..." when the PT is HAEMORRHAGE
    if event and "HEMORRH" in _fold_med(event):
        safe = re.sub(r"(h[ae]?morrhag\w*|bleed\w*)", r"<mark>\1</mark>", safe, flags=re.I)
    return safe


def _query_terms(query: str | None) -> list[str]:
    stop = {
        "WHAT", "DOES", "THE", "LABEL", "SAY", "ABOUT", "AND", "FOR", "WITH",
        "FROM", "THIS", "THAT", "HAVE", "DOES", "TELL", "GIVE",
    }
    return [
        t for t in re.split(r"[^A-Za-z0-9]+", (query or "").upper())
        if len(t) >= 5 and t not in stop
    ]


def _evidence_from_steps(
    steps,
    event: str | None,
    drug: str | None = None,
    query: str | None = None,
) -> list[dict]:
    items: list[dict] = []
    for action, obs in steps or []:
        if action.tool in ("search_signal_evidence", "search_literature", "search_drug_label"):
            if "no matching" in str(obs).lower():
                continue
            items.extend(parse_evidence_items(str(obs)))
    extra = [x for x in (drug, *_query_terms(query)) if x]
    return _summarize_evidence_items(items, event, extra=extra)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — answer text
# ─────────────────────────────────────────────────────────────────────────────
def _strip_disclaimer(text: str) -> str:
    if not text:
        return ""
    idx = text.lower().find("this tool surfaces reporting patterns")
    if idx >= 0:
        text = text[:idx]
    return text.strip()


def _clean_bullet(text: str, prefixes: tuple[str, ...]) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip(" \t-•*"))
    for prefix in prefixes:
        text = re.sub(rf"^\*?\*?{re.escape(prefix)}\*?\*?:?\s*", "", text, flags=re.I)
    return text.strip(" —:-").replace("**", "").strip()


def _extract_next_step(answer: str) -> str:
    text = _strip_disclaimer(answer)
    m = re.search(r"\(c\)\s*(.+)$", text, flags=re.I | re.S)
    if m:
        return _clean_bullet(m.group(1), ("Next step", "Recommended review"))
    return ""


def signal_summary_bullets(stats: dict | None, evidence: list[dict], answer: str) -> list[str]:
    """Three short bullets. No numbers — those are already in the cards."""
    if stats is None:
        first = "No statistics were returned for this pair."
    elif screening(stats)["met"]:
        first = "Reporting of this pair is disproportionately high; the screening criteria are met."
    else:
        first = "Reporting of this pair does not meet the screening criteria."

    second = (
        "The event is described in the retrieved literature."
        if evidence
        else "No literature text was retrieved for this event."
    )

    third = _extract_next_step(answer)
    if not third or re.search(r"\d", third) or len(third) > 220:
        third = (
            "A qualified PV professional should review the case reports and product "
            "information before any regulatory or clinical action."
        )
    return [first, second, third]


def detect_pair(steps) -> tuple[str, str] | None:
    """Find a resolved drug + event in the tool trace of a free query."""
    drug = event = None
    for action, obs in steps or []:
        m = re.search(r"->\s*'([^']+)'", str(obs))
        if not m:
            continue
        if action.tool == "resolve_drug_name":
            drug = m.group(1)
        elif action.tool == "resolve_event_name":
            event = m.group(1)
    return (drug, event) if drug and event else None


# ─────────────────────────────────────────────────────────────────────────────
# Agent run + guardrails
# ─────────────────────────────────────────────────────────────────────────────
def run_and_collect(
    executor,
    query: str,
    *,
    expected_drug: str | None = None,
    expected_event: str | None = None,
) -> dict:
    """Run the agent, collect intermediate steps, run both guardrail checks."""
    t0 = time.time()
    result = executor.invoke({"input": query})
    elapsed = time.time() - t0

    answer = result.get("output", "")
    steps = result.get("intermediate_steps", [])

    tool_outs = [str(obs) for _, obs in steps]
    tool_text = "\n\n".join(tool_outs)

    guardrail = run_guardrails(
        answer, tool_text, tool_outs,
        expected_drug=expected_drug,
        expected_event=expected_event,
    )
    llm_judge = run_llm_judge(answer, tool_outs)

    def _countable(n: str) -> bool:
        try:
            return len(n) > 1 and float(n) > 10
        except ValueError:
            return False

    ans_nums = {n for n in extract_numbers(answer) if _countable(n)}
    ctx_nums = extract_numbers(tool_text) if tool_outs else set()
    if ans_nums:
        verified = sum(1 for n in ans_nums if any(_numbers_match(n, c) for c in ctx_nums))
        traceability = int(verified / len(ans_nums) * 100)
    else:
        traceability = 100

    logger.info("=" * 60)
    logger.info("REGEX GUARDRAIL   passed=%s  traceability=%d%%", guardrail["passed"], traceability)
    logger.info("LLM JUDGE         verdict=%s  reasoning=%s",
                llm_judge.get("overall_verdict"), llm_judge.get("reasoning"))
    for issue in guardrail.get("issues", []):
        logger.warning("  REGEX ISSUE: %s", issue)
    for issue in llm_judge.get("issues", []):
        logger.warning("  JUDGE ISSUE: %s", issue)
    logger.info("TOOL TRACE  %d step(s)  elapsed=%.1fs", len(steps), elapsed)
    for i, (action, obs) in enumerate(steps, 1):
        logger.info("  [%d] %-26s | %s", i, action.tool, str(action.tool_input)[:120])
        logger.info("       → %s", str(obs)[:200])
    logger.info("=" * 60)

    return dict(answer=answer, steps=steps, tool_outs=tool_outs,
                guardrail=guardrail, llm_judge=llm_judge,
                traceability=traceability, elapsed=elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Renderers (centre panel)
# ─────────────────────────────────────────────────────────────────────────────
def render_evidence_cards(
    evidence: list[dict], event: str | None, drug: str | None = None
) -> None:
    links = _cite_links_html(drug) if drug else ""
    for item in evidence:
        head = f"§{item['section']} · Literature" if item.get("section") else "Literature"
        st.markdown(
            f'<div class="ev-card"><div class="ev-head">{html.escape(head)}</div>'
            f'{_highlight(item["text"], event)}{links}</div>',
            unsafe_allow_html=True,
        )


def render_signal(ctx: dict, res: dict) -> None:
    drug_name, event_name, row = ctx["drug"], ctx["event"], ctx.get("row")
    steps = res["steps"]

    st.markdown(
        f'<div class="result-title">{html.escape(drug_name)} &nbsp;×&nbsp; {html.escape(event_name)}</div>'
        f'<div class="result-query">FAERS 2026Q1–2026Q2 · generated {ctx.get("ts", "")} · '
        f'drug match {ctx.get("dm_score", 100):.0f}% · event match {ctx.get("em_score", 100):.0f}%</div>',
        unsafe_allow_html=True,
    )

    stats = _stats_from_row(row) or _stats_from_steps(steps)
    evidence = _evidence_from_steps(steps, event_name, drug_name)

    if stats is None:
        st.warning("No statistics were found for this drug and event pair.")
    else:
        scr = screening(stats)
        strength = signal_strength(stats)
        ser_pct = stats["serious"] / stats["a"] * 100 if stats["a"] > 0 else 0.0

        st.markdown(
            f'<div class="str-banner {strength["banner"]}">'
            f'<span>Signal strength: {html.escape(strength["label"])}</span>'
            f'<span class="str-sub">{html.escape(strength["band"])}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Status chips
        crit_chip = (
            '<span class="chip chip-info">Disproportionality: screening criteria met</span>'
            if scr["met"]
            else '<span class="chip chip-neutral">Screening criteria not met</span>'
        )
        lit_chip = (
            '<span class="chip chip-ok">In retrieved literature</span>'
            if evidence
            else '<span class="chip chip-neutral">No literature retrieved</span>'
        )
        st.markdown(
            f'<div class="chip-row">{crit_chip}{lit_chip}</div>',
            unsafe_allow_html=True,
        )

        # KPI cards
        total_sub = f"of {stats['total']:,} reports" if stats.get("total") else "&nbsp;"
        st.markdown('<div class="section-header">Statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-label">Cases</div><div class="kpi-value">{stats['a']:,}</div><div class="kpi-sub">{total_sub}</div></div>
  <div class="kpi"><div class="kpi-label">PRR</div><div class="kpi-value">{stats['prr']:,.2f}</div><div class="kpi-sub">&nbsp;</div></div>
  <div class="kpi"><div class="kpi-label">ROR</div><div class="kpi-value">{stats['ror']:,.2f}</div><div class="kpi-sub">&nbsp;</div></div>
  <div class="kpi"><div class="kpi-label">Serious</div><div class="kpi-value">{ser_pct:.1f}%</div><div class="kpi-sub">{stats['serious']:,} reports</div></div>
</div>
""", unsafe_allow_html=True)

        # Screening checks
        n_mark = '<span class="chk-ok">✓</span>' if scr["n_ok"] else '<span class="chk-bad">✗</span>'
        p_mark = '<span class="chk-ok">✓</span>' if scr["prr_ok"] else '<span class="chk-bad">✗</span>'
        st.markdown('<div class="section-header">Screening criteria</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="chk-row"><span>Cases ≥ 3</span><span>{stats['a']:,} &nbsp;{n_mark}</span></div>
<div class="chk-row"><span>PRR ≥ 2</span><span>{stats['prr']:,.2f} &nbsp;{p_mark}</span></div>
<div class="chk-row"><span>Signal strength</span><span class="{strength['val']}">{html.escape(strength["label"])} · {html.escape(strength["band"])}</span></div>
""", unsafe_allow_html=True)

    # Evidence
    st.markdown('<div class="section-header">Literature</div>', unsafe_allow_html=True)
    if evidence:
        render_evidence_cards(evidence, event_name, drug=drug_name)
    else:
        st.caption("No readable literature passage was retrieved for this event.")
        st.markdown(_cite_links_html(drug_name), unsafe_allow_html=True)

    # Summary (3 bullets, no repeated numbers)
    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
    for b in signal_summary_bullets(stats, evidence, res.get("answer") or ""):
        st.markdown(f"- {b}")


def render_query(ctx: dict, res: dict) -> None:
    steps = res["steps"]
    st.markdown(
        f'<div class="result-title" style="font-size:17px;">Query result</div>'
        f'<div class="result-query">"{html.escape(ctx["query"])}" &nbsp;·&nbsp; {ctx.get("ts", "")}</div>',
        unsafe_allow_html=True,
    )

    # Answer
    answer = _strip_disclaimer(res.get("answer") or "")
    st.markdown('<div class="section-header">Answer</div>', unsafe_allow_html=True)
    if answer:
        with st.container():
            st.markdown(answer)
    else:
        st.caption("No answer was returned. Re-run the query.")

    # Sources — same PDF passages the agent retrieved, if they relate to the query
    pair = detect_pair(steps)
    ev = pair[1] if pair else None
    dr = (
        (pair[0] if pair else None)
        or _drug_from_steps(steps)
        or _drug_from_text(ctx.get("query"))
    )
    evidence = _evidence_from_steps(steps, ev, dr, query=ctx.get("query"))
    if evidence or dr:
        st.markdown('<div class="section-header">Sources</div>', unsafe_allow_html=True)
        if evidence:
            render_evidence_cards(evidence, ev or dr, drug=dr)
        elif dr:
            st.caption("No readable literature passage was retrieved for this query.")
            st.markdown(_cite_links_html(dr), unsafe_allow_html=True)

    # Scope note
    used_stats = any(a.tool == "calculate_pv_statistics" for a, _ in steps)
    scope = (
        "Includes FAERS statistics from the stats tool."
        if used_stats
        else "No FAERS statistics were used for this answer."
    )
    st.markdown(
        f'<div class="note-box">Answered from {len(evidence)} passage(s). {scope}</div>',
        unsafe_allow_html=True,
    )

    # Offer the full analysis when a drug + event pair was resolved
    if pair:
        if st.button(f"Run full signal analysis: {pair[0]} × {pair[1]}", key=f"full_{ctx.get('ts', '')}"):
            st.session_state["pending_pair"] = pair
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "current" not in st.session_state:
    st.session_state.current = None
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_LLM_MODEL

# A "Run full signal analysis" click from Query mode lands here on the rerun.
_pending = st.session_state.pop("pending_pair", None)
if _pending:
    st.session_state["mode"] = MODE_SIGNAL
    st.session_state["l_drug"], st.session_state["l_event"] = _pending
    st.session_state["auto_run"] = True

# Optional deep-link: /?autorun=1&drug=DEPO-PROVERA&event=MENINGIOMA  or  /?autorun=1&q=...
if (st.query_params.get("autorun") or "").strip().lower() in ("1", "true"):
    qp_drug = (st.query_params.get("drug") or "").strip()
    qp_event = (st.query_params.get("event") or "").strip()
    qp_q = (st.query_params.get("q") or "").strip()
    st.query_params.clear()
    if qp_drug and qp_event:
        st.session_state["mode"] = MODE_SIGNAL
        st.session_state["l_drug"] = qp_drug
        st.session_state["l_event"] = qp_event
        st.session_state["auto_run"] = True
    elif qp_q:
        st.session_state["mode"] = MODE_QUERY
        st.session_state["l_free"] = qp_q
        st.session_state["auto_run"] = True

# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading agent…"):
    agent_executor = get_agent(st.session_state.model_name)
    signal_df = get_signal_df()

# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
  <span class="top-bar-title">🔬 PV Signal Console</span>
  <span class="top-bar-meta">FAERS 2026Q1–2026Q2 &nbsp;·&nbsp; {html.escape(st.session_state.model_name)}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Three-column layout
# ─────────────────────────────────────────────────────────────────────────────
left, centre, right = st.columns([2, 5, 2])

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Query
# ═══════════════════════════════════════════════════════════════════════════════
drug_in = event_in = free_q = ""

with left:
    st.markdown('<div class="panel-title">Ask</div>', unsafe_allow_html=True)

    mode = st.radio("Query mode", [MODE_SIGNAL, MODE_QUERY],
                    label_visibility="collapsed", key="mode")

    if mode == MODE_SIGNAL:
        st.markdown('<div class="section-label">Drug name</div>', unsafe_allow_html=True)
        drug_in = st.text_input("Drug name", placeholder="e.g. DEPO-PROVERA",
                                label_visibility="collapsed", key="l_drug")
        st.markdown('<div class="section-label">Adverse event (MedDRA PT)</div>', unsafe_allow_html=True)
        event_in = st.text_input("Adverse event", placeholder="e.g. MENINGIOMA",
                                 label_visibility="collapsed", key="l_event")
        run_btn = st.button("🔬 Run analysis", type="primary", use_container_width=True, key="b_signal")
    else:
        st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
        free_q = st.text_area("Your question", height=100,
                              placeholder="e.g. What is written about Depo-Provera and meningioma?",
                              label_visibility="collapsed", key="l_free")
        run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True, key="b_free")

    st.markdown("---")
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    try:
        model_index = UI_LLM_MODELS.index(st.session_state.model_name)
    except ValueError:
        st.session_state.model_name = DEFAULT_LLM_MODEL
        model_index = 0
    new_model = st.selectbox(
        "LLM model",
        UI_LLM_MODELS,
        index=model_index,
        format_func=lambda m: f"{m}  ({provider_for_model(m)})",
        label_visibility="collapsed",
    )
    if new_model != st.session_state.model_name:
        st.session_state.model_name = new_model
        st.rerun()

    st.markdown('<div class="section-label">Data quarters</div>', unsafe_allow_html=True)
    st.caption("FAERS 2026Q1–2026Q2 (loaded)")

    if st.session_state.history:
        st.markdown("---")
        st.markdown('<div class="panel-title">Session history</div>', unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            if st.button(f"{h['label']}", key=f"hist_{i}", use_container_width=True):
                st.session_state.current = h["context"]

# ═══════════════════════════════════════════════════════════════════════════════
# Trigger analysis
# ═══════════════════════════════════════════════════════════════════════════════
auto_run = st.session_state.pop("auto_run", False)

if run_btn or auto_run:
    if mode == MODE_SIGNAL:
        drug_q, event_q = drug_in.strip(), event_in.strip()
        if not drug_q or not event_q:
            with centre:
                st.warning("Enter both drug name and adverse event.")
        else:
            row, dm, em = lookup_pair(signal_df, drug_q, event_q)
            drug_name = dm[0] if dm else drug_q.upper()
            event_name = em[0] if em else event_q.upper()

            agent_q = (
                f"Analyze FAERS pair {drug_name} / {event_name}. "
                f"Spellings are exact — skip resolve tools. "
                f"In one step call both: calculate_pv_statistics "
                f"(pandas: drugname=='{drug_name}', pt=='{event_name}', "
                f"print prr,ror,a_drug_and_event,serious_reports) "
                f"and search_literature('{drug_name} {event_name}'). "
                f"Numbers only from the stats tool. Ignore unrelated literature. "
                f"Then write the final answer using the FINAL ANSWER FORMAT "
                f"from the system prompt (Statistics, Literature, then bullets "
                f"(a)(b)(c)). No causal claims, no tables, no PDF names."
            )

            with centre:
                with st.spinner(f"Running analysis for {drug_name} × {event_name}…"):
                    res = run_and_collect(
                        agent_executor, agent_q,
                        expected_drug=drug_name,
                        expected_event=event_name,
                    )

            ctx = dict(mode="signal", drug=drug_name, event=event_name,
                       row=row, res=res, ts=_ts(), query=agent_q,
                       dm_score=dm[1] if dm else 0, em_score=em[1] if em else 0)
            st.session_state.current = ctx
            st.session_state.history.append(
                dict(label=f"{drug_name} × {event_name}", ts=_ts(), result=res, context=ctx)
            )
    else:
        fq = free_q.strip()
        if not fq:
            with centre:
                st.warning("Enter a question.")
        else:
            with centre:
                with st.spinner("Agent is reasoning…"):
                    res = run_and_collect(agent_executor, fq)
            ctx = dict(mode="free", query=fq, res=res, ts=_ts())
            st.session_state.current = ctx
            st.session_state.history.append(
                dict(label=fq[:48] + ("…" if len(fq) > 48 else ""), ts=_ts(), result=res, context=ctx)
            )

# ═══════════════════════════════════════════════════════════════════════════════
# CENTRE PANEL — Results
# ═══════════════════════════════════════════════════════════════════════════════
with centre:
    ctx = st.session_state.current

    if ctx is None:
        st.markdown("""
<div style="text-align:center; color:#b0bec5; margin-top:80px;">
  <div style="font-size:40px;">🔬</div>
  <div style="font-size:16px; margin-top:10px; font-weight:600; color:#90a4ae;">PV Signal Console</div>
  <div style="font-size:13px; margin-top:6px;">
    Enter a drug name and adverse event on the left, then click <b>Run analysis</b>.
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        res = ctx.get("res") or ctx.get("result")
        if not res:
            st.warning("Result data is not available for this history item. Re-run the query.")
        else:
            if ctx["mode"] == "signal":
                render_signal(ctx, res)
            else:
                render_query(ctx, res)

            # Show a banner only when a guardrail check fails
            gr = res["guardrail"]
            if not gr["passed"]:
                issues_txt = " · ".join(gr["issues"])
                st.markdown(
                    f'<div class="gr-fail">⚠️ Guardrail issues: {html.escape(issues_txt)}'
                    f' · traceability {res["traceability"]}%</div>',
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Evaluation + collapsed audit trail
# ═══════════════════════════════════════════════════════════════════════════════
with right:
    ctx = st.session_state.current
    st.markdown('<div class="panel-title">Evaluation</div>', unsafe_allow_html=True)

    if ctx is None or not (ctx.get("res") or ctx.get("result")):
        st.caption("No run yet.")
    else:
        res = ctx.get("res") or ctx.get("result")
        gr = res["guardrail"]
        judge = res.get("llm_judge", {})
        trac = res["traceability"]
        elap = res.get("elapsed", 0)
        steps = res["steps"]
        j_verdict = judge.get("overall_verdict", "error")
        has_evid = any("no matching" not in o.lower() for o in res["tool_outs"])

        regex_ok = gr["passed"]
        judge_ok = j_verdict == "pass"
        judge_err = j_verdict == "error"

        if judge_err:
            icon = "✅" if regex_ok else "❌"
            color = "#2e7d32" if regex_ok else "#c62828"
            label = "Verified" if regex_ok else "Issues found"
            note = judge.get("reasoning") or "LLM judge could not finish. Re-run the query."
        elif regex_ok and judge_ok:
            icon, color, label = "✅", "#2e7d32", "Verified"
            note = "Numerically grounded and worded safely."
        else:
            icon, color, label = "❌", "#c62828", "Review needed"
            note = judge.get("reasoning", "") or " · ".join(gr.get("issues", []))

        st.markdown(f"""
<div style="background:#f8f9ff; border-left:4px solid {color};
            padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:8px;">
  <span style="font-weight:700; color:{color}; font-size:14px;">{icon} {label}</span>
  <div style="font-size:11px; color:#666; margin-top:4px; line-height:1.5;">{html.escape(note[:120])}</div>
</div>
<div style="font-size:12px; color:#555; line-height:1.9; margin-bottom:6px;">
  Tool calls: {len(steps)} · Time: {elap:.1f}s<br>
  Traceability: {trac}% · Evidence: {'Yes' if has_evid else 'No'}
</div>""", unsafe_allow_html=True)

        # Audit trail — collapsed by default
        ICONS = {
            "resolve_drug_name": "💊",
            "resolve_event_name": "🏥",
            "calculate_pv_statistics": "📊",
            "search_signal_evidence": "🔍",
            "search_literature": "📚",
            "search_drug_label": "🏷️",
        }
        with st.expander(f"Audit trail ({len(steps)} tool calls)", expanded=False):
            if not steps:
                st.caption("No tool calls in last run.")
            for action, obs in steps:
                inp = action.tool_input
                inp_str = (
                    inp.get("python_code", inp).strip()
                    if isinstance(inp, dict) and "python_code" in inp
                    else str(inp)
                )
                out_str = str(obs)[:220]
                st.markdown(f"""
<div class="audit-step">
  <span class="audit-tool">{ICONS.get(action.tool, "🔧")} {action.tool}</span>
  <span class="audit-time">{ctx['ts']}</span>
  <div class="audit-io">
    <b>in:</b> {html.escape(str(inp_str)[:120])}<br>
    <b>out:</b> {html.escape(out_str)}{"…" if len(str(obs)) > 220 else ""}
  </div>
</div>""", unsafe_allow_html=True)

        # Check details — collapsed by default
        with st.expander("Check details", expanded=False):
            st.markdown("**LLM judge**")
            checks = [
                ("Numeric faithfulness", judge.get("numeric_faithfulness")),
                ("No causal language",
                 not judge.get("causal_language_used", True)
                 if judge.get("causal_language_used") is not None else None),
                ("Evidence grounded", judge.get("evidence_grounded")),
            ]
            for label_txt, val in checks:
                mark = "✅" if val is True else ("❌" if val is False else "–")
                st.markdown(f'<span style="font-size:12px;">{mark} {label_txt}</span>',
                            unsafe_allow_html=True)
            for issue in judge.get("issues", []):
                st.caption(f"⚠️ {issue}")

            st.markdown("**Regex guardrail**")
            st.markdown(f'<span style="font-size:12px;">{"✅ Pass" if regex_ok else "❌ Fail"}</span>',
                        unsafe_allow_html=True)
            for issue in gr.get("issues", []):
                st.caption(f"• {issue[:120]}")

# ─────────────────────────────────────────────────────────────────────────────
# Footer — the only disclaimer on the page
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="foot">{FOOTER_TEXT}</div>', unsafe_allow_html=True)
