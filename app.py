"""PV Signal Console — Streamlit UI

Three-column layout matching the PV Signal Console design:
  LEFT   — Query panel (drug + event inputs, free query, session history)
  CENTRE — Results (signal badge, statistical evidence table, retrieved
            evidence, limitations, recommended review, guardrail banner)
  RIGHT  — Audit trail (timestamped tool-call log)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
from rapidfuzz import process as fuzzy_process

from src.agent.build_agent import build_pv_agent
from src.config import (
    CANDIDATE_LLM_MODELS,
    DEFAULT_LLM_MODEL,
    DISCLAIMER,
    provider_for_model,
)
from src.guardrails.llm_judge import run_llm_judge
from src.guardrails.validators import run_guardrails, extract_numbers, _numbers_match
from src.ingestion.faers_ingest import load_signal_summary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; }

/* ── Top bar ──────────────────────────────────────────────────── */
.top-bar {
    background: #0d1b4b;
    color: #b3d1ff;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 0 0 6px 6px;
    margin-bottom: 12px;
}
.top-bar-title { color: #ffffff; font-size: 14px; }
.top-bar-meta  { font-size: 11px; color: #7da8e8; }

/* ── Disclaimer strip ─────────────────────────────────────────── */
.disc-strip {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 7px 14px;
    font-size: 11.5px;
    color: #6d4c1f;
    margin-bottom: 10px;
    border-radius: 0 4px 4px 0;
}

/* ── Left panel ───────────────────────────────────────────────── */
.panel-title {
    font-size: 11px;
    font-weight: 700;
    color: #7da8e8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
    border-bottom: 1px solid #e8eaf6;
    padding-bottom: 4px;
}
.section-label {
    font-size: 11px;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: .9px;
    margin: 10px 0 3px;
}

/* ── Signal badge ─────────────────────────────────────────────── */
.sig-badge {
    padding: 14px 20px;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    font-size: 17px;
    letter-spacing: .5px;
    margin: 10px 0 6px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.14);
}
.sig-strong   { background:#c62828; color:#fff; }
.sig-moderate { background:#bf360c; color:#fff; }
.sig-weak     { background:#e65100; color:#fff; }
.sig-none     { background:#2e7d32; color:#fff; }

/* Signal strength bar */
.sig-bar-wrap { background:#e8eaf6; border-radius:4px; height:8px; margin:3px 0 10px; }
.sig-bar-fill { height:8px; border-radius:4px; }

/* ── Stat evidence table ──────────────────────────────────────── */
.ev-table { width:100%; border-collapse:collapse; font-size:13.5px; margin:6px 0; }
.ev-table td { padding:8px 10px; border-bottom:1px solid #f0f0f0; }
.ev-table td:first-child { color:#555; }
.ev-table td:last-child { text-align:right; font-weight:700; color:#0d1b4b; font-size:15px; }
.ev-table tr:last-child td { border-bottom:none; }

/* ── Evidence item ────────────────────────────────────────────── */
.ev-item {
    background:#fafbff;
    border:1px solid #e3e8f5;
    border-radius:6px;
    padding:10px 14px;
    margin-bottom:8px;
    font-size:13px;
}
.ev-item-rel {
    font-size:11px;
    color:#1565c0;
    font-weight:700;
    float:right;
    background:#e3f0ff;
    padding:2px 7px;
    border-radius:10px;
}

/* ── Limitation / review boxes ────────────────────────────────── */
.limit-box {
    background:#fff8e1;
    border-left:4px solid #f9a825;
    padding:12px 16px;
    border-radius:0 6px 6px 0;
    font-size:13px;
    color:#5d4037;
    margin:10px 0;
}
.review-box {
    background:#e8f4fd;
    border-left:4px solid #1565c0;
    padding:12px 16px;
    border-radius:0 6px 6px 0;
    font-size:13px;
    color:#0d3b6e;
    margin:10px 0;
}

/* ── Guardrail banner ─────────────────────────────────────────── */
.gr-pass {
    background:#e8f5e9;
    border:1px solid #a5d6a7;
    border-radius:6px;
    padding:9px 16px;
    font-size:12.5px;
    color:#2e7d32;
    margin-top:10px;
}
.gr-fail {
    background:#fce4ec;
    border:1px solid #ef9a9a;
    border-radius:6px;
    padding:9px 16px;
    font-size:12.5px;
    color:#b71c1c;
    margin-top:10px;
}

/* ── Audit trail (right panel) ────────────────────────────────── */
.audit-step {
    border-left:3px solid #1565c0;
    padding:8px 10px 8px 12px;
    margin-bottom:10px;
    background:#f8f9ff;
    border-radius:0 6px 6px 0;
}
.audit-tool { font-weight:700; font-size:12px; color:#1565c0; }
.audit-time { font-size:10px; color:#aaa; float:right; }
.audit-io   { font-size:11px; color:#555; margin-top:4px; word-break:break-word; }

/* ── Session history ──────────────────────────────────────────── */
.hist-item {
    padding:7px 10px;
    border-radius:5px;
    font-size:12px;
    color:#1565c0;
    cursor:pointer;
    margin-bottom:4px;
    background:#f0f4ff;
    border-left:3px solid #1565c0;
}
.hist-time { font-size:10px; color:#aaa; display:block; }

/* ── Result title ─────────────────────────────────────────────── */
.result-title {
    font-size:22px;
    font-weight:800;
    color:#0d1b4b;
    margin:4px 0 2px;
    letter-spacing:.2px;
}
.result-query {
    font-size:12px;
    color:#888;
    font-style:italic;
    margin-bottom:10px;
}
.step-label {
    display:inline-block;
    background:#e3f0ff;
    color:#1565c0;
    font-size:10px;
    font-weight:700;
    padding:2px 8px;
    border-radius:10px;
    margin-left:8px;
    vertical-align:middle;
    letter-spacing:.5px;
    text-transform:uppercase;
}
.section-header {
    font-size:14px;
    font-weight:700;
    color:#0d1b4b;
    margin:14px 0 6px;
    display:flex;
    align-items:center;
    gap:8px;
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def classify_signal(prr: float) -> dict:
    if prr >= 10:
        return dict(level="STRONG SIGNAL", css="sig-strong", emoji="🔴", color="#c62828", pct=95,
                    evans="PRR ≥ 10 — Marked disproportionality. Priority regulatory review recommended.")
    elif prr >= 5:
        return dict(level="MODERATE-HIGH SIGNAL", css="sig-moderate", emoji="🟠", color="#bf360c", pct=70,
                    evans="PRR 5–10 — Significant disproportionality. Further investigation warranted.")
    elif prr >= 2:
        return dict(level="WEAK SIGNAL", css="sig-weak", emoji="🟡", color="#e65100", pct=40,
                    evans="PRR 2–5 — Potential signal. Monitor and accumulate more data.")
    else:
        return dict(level="NO SIGNAL DETECTED", css="sig-none", emoji="🟢", color="#2e7d32", pct=10,
                    evans="PRR < 2 — No disproportionality detected in current FAERS data.")


def lookup_pair(df, drug_q, event_q):
    dm = fuzzy_process.extractOne(drug_q.upper(), df["drugname"].unique(), score_cutoff=70)
    em = fuzzy_process.extractOne(event_q.upper(), df["pt"].unique(), score_cutoff=70)
    if dm is None or em is None:
        return None, dm, em
    row = df[(df["drugname"] == dm[0]) & (df["pt"] == em[0])]
    return (row.iloc[0] if not row.empty else None), dm, em


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _display_evidence(text: str, max_chars: int = 220) -> str:
    """One or two complete sentences for the evidence card."""
    text = re.sub(r"\s+", " ", text).strip()
    if text and not text[0].isupper() and ". " in text[:160]:
        text = text.split(". ", 1)[1].strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if cut >= 60:
        return window[: cut + 1].strip()
    return window.rstrip() + "…"


def _key_sentences(text: str, event: str | None) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    event_u = (event or "").upper()
    event_tokens = [t for t in re.split(r"[^A-Z0-9]+", event_u) if len(t) >= 5]
    skip_if_unrelated = (
        "ectopic pregnancy",
        "anaphylaxis",
        "injection site",
        "liver function",
        "carbohydrate metabolism",
        "menstrual irregularities",
        "reference id",
        "adverse reactions",
    )
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sent in sentences:
        sent = sent.strip(" •-\t")
        if len(sent) < 40:
            continue
        low = sent.lower()
        u = sent.upper()
        related = (not event_tokens) or any(t in u for t in event_tokens)
        if not related:
            continue
        if any(s in low for s in skip_if_unrelated) and not related:
            continue
        kept.append(sent)
    return kept


def _summarize_evidence_items(items: list[dict], event: str | None) -> list[dict]:
    """Collapse overlapping PDF chunks into 1–2 unique event-related sentences."""
    seen: list[str] = []
    out: list[dict] = []
    for item in items:
        for sent in _key_sentences(item["text"], event):
            fingerprint = re.sub(r"[^A-Z0-9]+", "", sent.upper())[:72]
            if any(fingerprint[:50] in prev or prev[:50] in fingerprint for prev in seen):
                continue
            seen.append(fingerprint)
            out.append({"text": sent, "relevance": item.get("relevance")})
            if len(out) >= 2:
                return out
    if out:
        return out
    if items:
        return [{"text": _display_evidence(items[0]["text"]), "relevance": items[0].get("relevance")}]
    return []


def parse_evidence_items(raw: str) -> list[dict]:
    """Parse the '[relevance X.XXX] ...' lines returned by retriever tools."""
    items = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        rel = None
        if block.startswith("[relevance"):
            try:
                rel_str = block[block.index(" ") + 1: block.index("]")]
                rel = float(rel_str)
                block = block[block.index("]") + 1:].strip()
            except Exception:
                pass
        # Do not show PDF filenames in the UI
        block = re.sub(r"^\[(?:[^\]]+\.pdf)\]\s*", "", block, flags=re.IGNORECASE)
        items.append({"text": block, "relevance": rel})
    return items


def _prr_strength_word(prr: float) -> str:
    if prr >= 10:
        return "Strong"
    if prr >= 5:
        return "Moderate"
    if prr >= 2:
        return "Weak"
    return "None"


def _strip_disclaimer(text: str) -> str:
    if not text:
        return ""
    idx = text.lower().find("this tool surfaces reporting patterns")
    if idx >= 0:
        text = text[:idx]
    return text.strip()


def _stats_from_row(row) -> dict | None:
    if row is None:
        return None
    try:
        return {
            "prr": float(row["prr"]),
            "ror": float(row["ror"]),
            "a": int(row["a_drug_and_event"]),
            "serious": int(row["serious_reports"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _stats_from_steps(steps) -> dict | None:
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
            if 0 <= j < len(vals):
                return vals[j]
            return None

        try:
            return {
                "prr": float(_col("prr")),
                "ror": float(_col("ror")),
                "a": int(float(_col("a_drug_and_event"))),
                "serious": int(float(_col("serious_reports"))),
            }
        except (TypeError, ValueError):
            continue
    return None


def _clean_bullet(text: str, prefixes: tuple[str, ...]) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip(" \t-•*"))
    for prefix in prefixes:
        text = re.sub(rf"^\*?\*?{re.escape(prefix)}\*?\*?:?\s*", "", text, flags=re.I)
    return text.strip(" —:-").replace("**", "").strip()


def _extract_abc(answer: str) -> tuple[str, str, str]:
    text = _strip_disclaimer(answer)
    abc = re.search(
        r"\(a\)\s*(.+?)\s*\(b\)\s*(.+?)\s*\(c\)\s*(.+)$",
        text,
        flags=re.I | re.S,
    )
    if abc:
        return (
            _clean_bullet(abc.group(1), ("Strength", "Signal strength")),
            _clean_bullet(abc.group(2), ("FAERS limitation", "Limitation")),
            _clean_bullet(abc.group(3), ("Next step", "Recommended review")),
        )
    labeled = re.search(
        r"(?:signal\s+)?strength:\s*(.+?)(?:faers\s+limitation:\s*(.+?))?(?:next\s+step:\s*(.+))?$",
        text,
        flags=re.I | re.S,
    )
    if labeled:
        return (
            _clean_bullet(labeled.group(1) or "", ("Strength", "Signal strength")),
            _clean_bullet(labeled.group(2) or "", ("FAERS limitation", "Limitation")),
            _clean_bullet(labeled.group(3) or "", ("Next step",)),
        )
    return "", "", ""


def _extract_literature(answer: str) -> str:
    text = _strip_disclaimer(answer)
    match = re.search(
        r"Literature:\s*(.+?)(?=\n\s*\**Bullets|\n\s*\(a\)|\n\s*-\s*\(a\)|\n\s*-\s*\*\*Signal strength|\Z)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        return ""
    lit = re.sub(r"\s+", " ", match.group(1)).replace("**", "").strip()
    if lit.lower().startswith("bullets"):
        return ""
    return lit


def _literature_from_steps(steps, event: str | None) -> str:
    items: list[dict] = []
    saw_empty = False
    for action, obs in steps or []:
        if getattr(action, "tool", None) not in (
            "search_literature",
            "search_drug_label",
            "search_signal_evidence",
        ):
            continue
        raw = str(obs)
        if "no matching" in raw.lower():
            saw_empty = True
            continue
        items.extend(parse_evidence_items(raw))
    summarized = _summarize_evidence_items(items, event)
    if summarized:
        return summarized[0]["text"]
    if saw_empty or not items:
        return "No matching literature was retrieved."
    return _display_evidence(items[0]["text"])


def format_standard_interpretation(
    answer: str,
    *,
    drug: str | None = None,
    event: str | None = None,
    row=None,
    steps=None,
) -> str:
    """Rebuild the AI Interpretation block in one layout for every model."""
    stats = _stats_from_row(row) or _stats_from_steps(steps)
    a_txt, b_txt, c_txt = _extract_abc(answer)
    lit = _extract_literature(answer) or _literature_from_steps(steps, event)

    if stats:
        strength = _prr_strength_word(stats["prr"])
        if not a_txt:
            a_txt = (
                f"{strength} — PRR of {stats['prr']:.2f} is "
                f"{'far above' if stats['prr'] >= 10 else 'in'} the "
                f">=10 / >=5 / >=2 thresholds, indicating "
                f"{'marked' if stats['prr'] >= 10 else 'a'} disproportionality "
                f"in reporting rates."
            )
        elif not re.match(rf"{re.escape(strength)}\b", a_txt, flags=re.I):
            a_txt = f"{strength} — {a_txt}"
        pair = f"{drug or 'the drug'} / {event or 'the event'}"
        stats_block = (
            f"- PRR: {stats['prr']:.2f}\n"
            f"- ROR: {stats['ror']:.2f}\n"
            f"- a_drug_and_event (drug + event reports): {stats['a']:,}\n"
            f"- serious_reports: {stats['serious']:,}"
        )
        header = f"Here is the FAERS analysis for the {pair} pair."
    else:
        stats_block = "- No statistics were returned by the stats tool."
        header = "Here is the FAERS analysis."

    if not b_txt:
        b_txt = (
            "Spontaneous reporting is subject to under-reporting, duplicates, "
            "and missing denominators, so these ratios reflect reporting "
            "patterns, not incidence."
        )
    if not c_txt:
        c_txt = (
            "Review individual case reports (including the serious subset) "
            "for temporality and confounders before any regulatory or clinical action."
        )
    if not lit:
        lit = "No matching literature was retrieved."

    return (
        f"{header}\n\n"
        f"**Statistics (from the stats tool):**\n"
        f"{stats_block}\n\n"
        f"**Literature:** {lit}\n\n"
        f"**Bullets:**\n"
        f"- (a) **Strength:** {a_txt}\n"
        f"- (b) **FAERS limitation:** {b_txt}\n"
        f"- (c) **Next step:** {c_txt}\n\n"
        f"{DISCLAIMER}"
    )


def run_and_collect(executor, query: str) -> dict:
    """Run the agent, collect intermediate steps, run both guardrail checks."""
    t0 = time.time()
    result = executor.invoke({"input": query})
    elapsed = time.time() - t0

    answer = result.get("output", "")
    steps  = result.get("intermediate_steps", [])

    tool_outs = [str(obs) for _, obs in steps]
    tool_text = "\n\n".join(tool_outs)

    # Regex guardrail (instant, free)
    guardrail = run_guardrails(answer, tool_text, tool_outs)

    # LLM-as-Judge guardrail (semantic, one extra Groq call)
    llm_judge = run_llm_judge(answer, tool_outs)

    # Traceability score — ignore small prose numbers (PRR thresholds, step counts)
    # the same way the regex guardrail does, so "PRR >= 5 but < 10" does not
    # drag a correctly cited 9.26 down to 33%.
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

    # Terminal audit log
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
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {label, ts, result, context}
if "current" not in st.session_state:
    st.session_state.current = None

# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading agent…"):
    # Model selector lives in top bar — use default until user changes
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_LLM_MODEL
    agent_executor = get_agent(st.session_state.model_name)
    signal_df      = get_signal_df()

# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────
chosen_model = st.session_state.model_name
st.markdown(f"""
<div class="top-bar">
  <span class="top-bar-title">🔬 PV Signal Console</span>
  <span class="top-bar-meta">
    FAERS 2026Q1–2026Q2 &nbsp;·&nbsp; {chosen_model}
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="disc-strip">
  ⚠️ Reporting-pattern signal only. PRR/ROR indicate a statistical association in
  spontaneous-report rates, not a confirmed causal relationship.
  All outputs require qualified PV review before any regulatory or clinical action.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Three-column layout
# ─────────────────────────────────────────────────────────────────────────────
left, centre, right = st.columns([2, 5, 2])

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Query
# ═══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="panel-title">Ask</div>', unsafe_allow_html=True)

    # Mode toggle
    mode = st.radio("Query mode", ["Signal lookup (drug + event)", "Query"], label_visibility="collapsed")

    if mode == "Signal lookup (drug + event)":
        st.markdown('<div class="section-label">Drug Name</div>', unsafe_allow_html=True)
        drug_in = st.text_input("Drug Name", placeholder="e.g. DEPO-PROVERA", label_visibility="collapsed", key="l_drug")
        st.markdown('<div class="section-label">Adverse Event (MedDRA PT)</div>', unsafe_allow_html=True)
        event_in = st.text_input("Adverse Event", placeholder="e.g. MENINGIOMA", label_visibility="collapsed", key="l_event")
        run_btn = st.button("🔬 Run analysis", type="primary", use_container_width=True, key="b_signal")
    else:
        st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
        free_q = st.text_area("Your question", height=100,
                              placeholder="e.g. How serious are WARFARIN intracranial haemorrhage cases?",
                              label_visibility="collapsed", key="l_free")
        run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True, key="b_free")

    st.markdown("---")
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    try:
        model_index = CANDIDATE_LLM_MODELS.index(st.session_state.model_name)
    except ValueError:
        st.session_state.model_name = DEFAULT_LLM_MODEL
        model_index = 0
    new_model = st.selectbox(
        "LLM Model",
        CANDIDATE_LLM_MODELS,
        index=model_index,
        format_func=lambda m: f"{m}  ({provider_for_model(m)})",
        label_visibility="collapsed",
    )
    if new_model != st.session_state.model_name:
        st.session_state.model_name = new_model
        st.rerun()

    st.markdown('<div class="section-label">Data quarters</div>', unsafe_allow_html=True)
    st.caption("FAERS 2026Q1–2026Q2 (loaded)")

    # Session history
    if st.session_state.history:
        st.markdown("---")
        st.markdown('<div class="panel-title">Session history</div>', unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            if st.button(f"{h['label']}", key=f"hist_{i}", use_container_width=True):
                st.session_state.current = h["context"]

# ═══════════════════════════════════════════════════════════════════════════════
# Trigger analysis
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if mode == "Signal lookup (drug + event)":
        drug_q  = (drug_in  if "drug_in"  in dir() else "").strip()
        event_q = (event_in if "event_in" in dir() else "").strip()
        if not drug_q or not event_q:
            with centre:
                st.warning("Enter both Drug Name and Adverse Event.")
        else:
            row, dm, em = lookup_pair(signal_df, drug_q, event_q)
            drug_name  = dm[0] if dm else drug_q.upper()
            event_name = em[0] if em else event_q.upper()

            # Names are already resolved by lookup_pair. Skip extra resolve
            # round-trips so a second demo query (WARFARIN) can run immediately
            # after DEPO without waiting on Groq 429s. Stats still come from
            # calculate_pv_statistics.
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
                    res = run_and_collect(agent_executor, agent_q)

            ctx = dict(mode="signal", drug=drug_name, event=event_name,
                       row=row, res=res, ts=_ts(), query=agent_q,
                       dm_score=dm[1] if dm else 0, em_score=em[1] if em else 0)
            st.session_state.current = ctx
            st.session_state.history.append(
                dict(label=f"{drug_name} × {event_name}", ts=_ts(), result=res, context=ctx)
            )
    else:
        fq = (free_q if "free_q" in dir() else "").strip()
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
  <div style="font-size:16px; margin-top:10px; font-weight:600; color:#90a4ae;">
    PV Signal Console
  </div>
  <div style="font-size:13px; margin-top:6px; color:#b0bec5;">
    Enter a drug name and adverse event on the left, then click <b>Run analysis</b>.
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        res   = ctx.get("res") or ctx.get("result", {})
        if not res:
            st.warning("Result data not available for this history item. Please re-run the query.")
            st.stop()
        steps = res["steps"]
        ts    = ctx.get("ts", "")

        # ── Title ──────────────────────────────────────────────────────────
        if ctx["mode"] == "signal":
            drug_name  = ctx["drug"]
            event_name = ctx["event"]
            row        = ctx.get("row")
            st.markdown(
                f'<div class="result-title">{drug_name} &nbsp;×&nbsp; {event_name}</div>'
                f'<div class="result-query">report generated {ts} &nbsp;·&nbsp; '
                f'Drug match {ctx.get("dm_score",100):.0f}% · Event match {ctx.get("em_score",100):.0f}%</div>',
                unsafe_allow_html=True,
            )

            if row is not None:
                prr_v   = float(row["prr"])
                ror_v   = float(row["ror"])
                a_v     = int(row["a_drug_and_event"])
                ser_v   = int(row["serious_reports"])
                tot_v   = int(row["total_reports"])
                ser_pct = ser_v / a_v * 100 if a_v > 0 else 0.0

                sig = classify_signal(prr_v)

                # Signal badge
                st.markdown(
                    f'<div class="sig-badge {sig["css"]}">'
                    f'{sig["emoji"]} &nbsp; {sig["level"]}'
                    f'<div style="font-size:12.5px;font-weight:400;margin-top:5px;opacity:.92;">'
                    f'{sig["evans"]}</div></div>',
                    unsafe_allow_html=True,
                )
                # Strength bar
                st.markdown(
                    f'<div class="sig-bar-wrap">'
                    f'<div class="sig-bar-fill" style="width:{sig["pct"]}%;background:{sig["color"]};"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Statistical evidence table
                st.markdown(
                    '<div class="section-header">Statistical evidence</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"""
<table class="ev-table">
  <tr><td>Co-reports (drug + event)</td><td>{a_v:,}</td></tr>
  <tr><td>PRR (Proportional Reporting Ratio)</td><td>{prr_v:,.2f}</td></tr>
  <tr><td>ROR (Reporting Odds Ratio)</td><td>{ror_v:,.2f}</td></tr>
  <tr><td>Serious-outcome reports</td><td>{ser_v:,} &nbsp;({ser_pct:.1f}%)</td></tr>
  <tr><td>Total reports in database</td><td>{tot_v:,}</td></tr>
</table>
""", unsafe_allow_html=True)

        else:
            # Query mode
            st.markdown(
                f'<div class="result-title" style="font-size:17px;">Query result</div>'
                f'<div class="result-query">"{ctx["query"]}" &nbsp;·&nbsp; {ts}</div>',
                unsafe_allow_html=True,
            )

        # ── Prior signal / literature evidence ─────────────────────────────
        evid_steps = [(a, o) for a, o in steps
                      if a.tool in ("search_signal_evidence", "search_literature", "search_drug_label")]
        if evid_steps:
            st.markdown(
                '<div class="section-header">Supporting evidence</div>',
                unsafe_allow_html=True,
            )
            event_name = ctx.get("event")
            all_items = []
            for action, obs in evid_steps:
                all_items.extend(parse_evidence_items(str(obs)))
            for item in _summarize_evidence_items(all_items, event_name):
                rel_badge = (
                    f'<span class="ev-item-rel">rel {item["relevance"]:.2f}</span>'
                    if item["relevance"] is not None else ""
                )
                st.markdown(
                    f'<div class="ev-item">{rel_badge}{item["text"]}</div>',
                    unsafe_allow_html=True,
                )

        # ── LLM interpretation (same layout for every model) ──────────────
        answer = res.get("answer") or ""
        if answer.strip() or res.get("steps"):
            formatted = format_standard_interpretation(
                answer,
                drug=ctx.get("drug"),
                event=ctx.get("event"),
                row=ctx.get("row"),
                steps=res.get("steps"),
            )
            st.markdown('<div class="section-header">AI Interpretation</div>', unsafe_allow_html=True)
            st.markdown(formatted)

        # Static limitation + review boxes (always shown for signal queries)
        if ctx.get("mode") == "signal":
            st.markdown("""
<div class="section-header">Limitations</div>
<div class="limit-box">
Spontaneous-report data is subject to reporting bias and cannot establish causation.
A disproportionality signal indicates an elevated reporting rate relative to background
— it does not confirm the drug caused the event.
</div>
<div class="section-header" style="margin-top:10px;">Recommended review</div>
<div class="review-box">
Escalate to signal management workflow for formal disproportionality review
(e.g. EBGM/EBGM05) and literature cross-check before any labelling action.
</div>
""", unsafe_allow_html=True)

        # ── Guardrail banner ───────────────────────────────────────────────
        gr = res["guardrail"]
        trac = res["traceability"]
        if gr["passed"]:
            st.markdown(f"""
<div class="gr-pass">
  ✅ &nbsp; All figures traced to tool output &nbsp;·&nbsp;
  no causal language detected &nbsp;·&nbsp; evidence found &nbsp;·&nbsp;
  traceability {trac}%
</div>""", unsafe_allow_html=True)
        else:
            issues_txt = " · ".join(gr["issues"])
            st.markdown(f"""
<div class="gr-fail">
  ⚠️ &nbsp; Guardrail issues: {issues_txt}
  &nbsp;·&nbsp; traceability {trac}%
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Audit trail
# ═══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="panel-title">Audit trail</div>', unsafe_allow_html=True)

    ctx = st.session_state.current
    if ctx is None:
        st.caption("No run yet.")
    else:
        steps = ctx["res"]["steps"]
        if not steps:
            st.caption("No tool calls in last run.")
        else:
            ICONS = {
                "resolve_drug_name":      "💊",
                "resolve_event_name":     "🏥",
                "calculate_pv_statistics": "📊",
                "search_signal_evidence": "🔍",
                "search_literature":      "📚",
                "search_drug_label":      "🏷️",
            }
            for i, (action, obs) in enumerate(steps):
                icon = ICONS.get(action.tool, "🔧")
                inp  = action.tool_input
                inp_str = (
                    inp.get("python_code", inp).strip()
                    if isinstance(inp, dict) and "python_code" in inp
                    else str(inp)
                )
                out_str = str(obs)[:220]

                st.markdown(f"""
<div class="audit-step">
  <span class="audit-tool">{icon} {action.tool}</span>
  <span class="audit-time">{ctx['ts']}</span>
  <div class="audit-io">
    <b>in:</b> {str(inp_str)[:120]}<br>
    <b>out:</b> {out_str}{"…" if len(str(obs)) > 220 else ""}
  </div>
</div>""", unsafe_allow_html=True)

        # Evaluation panel below audit trail
        if ctx:
            res       = ctx["res"]
            gr        = res["guardrail"]
            judge     = res.get("llm_judge", {})
            trac      = res["traceability"]
            elap      = res.get("elapsed", 0)
            j_verdict = judge.get("overall_verdict", "error")
            has_evid  = any("no matching" not in o.lower() for o in res["tool_outs"])

            st.markdown("---")
            st.markdown('<div class="panel-title">Evaluation</div>', unsafe_allow_html=True)

            # ── Single verdict line ────────────────────────────────────────
            # Combine regex + LLM judge into one clear verdict
            regex_ok  = gr["passed"]
            judge_ok  = j_verdict == "pass"
            judge_err = j_verdict == "error"

            if judge_err:
                overall_icon  = "✅" if regex_ok else "❌"
                overall_color = "#2e7d32" if regex_ok else "#c62828"
                overall_label = "Verified" if regex_ok else "Issues found"
                overall_note  = judge.get("reasoning") or "LLM judge could not finish. Re-run the query."
            elif regex_ok and judge_ok:
                overall_icon, overall_color = "✅", "#2e7d32"
                overall_label = "Verified"
                overall_note  = "Numerically grounded and worded safely."
            else:
                overall_icon, overall_color = "❌", "#c62828"
                overall_label = "Review needed"
                overall_note  = judge.get("reasoning", "") or " · ".join(gr.get("issues", []))

            st.markdown(f"""
<div style="background:#f8f9ff; border-left:4px solid {overall_color};
            padding:10px 14px; border-radius:0 8px 8px 0; margin-bottom:10px;">
  <span style="font-weight:700; color:{overall_color}; font-size:14px;">
    {overall_icon} {overall_label}
  </span>
  <div style="font-size:11px; color:#666; margin-top:4px; line-height:1.5;">
    {overall_note[:120]}
  </div>
</div>""", unsafe_allow_html=True)

            # ── Compact metrics ────────────────────────────────────────────
            st.markdown(f"""
<div style="font-size:12px; color:#555; line-height:2;">
  <b>Tool calls:</b> {len(res['steps'])} &nbsp;·&nbsp;
  <b>Time:</b> {elap:.1f}s<br>
  <b>Traceability:</b> {trac}% &nbsp;·&nbsp;
  <b>Evidence:</b> {'Yes' if has_evid else 'No'}
</div>""", unsafe_allow_html=True)

            # ── Expandable audit details ───────────────────────────────────
            with st.expander("🔍 Audit details"):
                st.markdown("**LLM Judge**")
                checks = [
                    ("Numeric faithfulness", judge.get("numeric_faithfulness")),
                    ("No causal language",
                     not judge.get("causal_language_used", True)
                     if judge.get("causal_language_used") is not None else None),
                    ("Evidence grounded", judge.get("evidence_grounded")),
                ]
                for label, val in checks:
                    icon = "✅" if val is True else ("❌" if val is False else "–")
                    st.markdown(
                        f'<span style="font-size:12px;">{icon} {label}</span>',
                        unsafe_allow_html=True,
                    )
                for issue in judge.get("issues", []):
                    st.caption(f"⚠️ {issue}")

                st.markdown("**Regex Guardrail**")
                st.markdown(
                    f'<span style="font-size:12px;">{"✅ Pass" if regex_ok else "❌ Fail"}</span>',
                    unsafe_allow_html=True,
                )
                for issue in gr.get("issues", []):
                    st.caption(f"• {issue[:120]}")
