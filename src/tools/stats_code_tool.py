"""The core ask: an LLM-authored, code-executed statistics tool.

Instead of (a) pre-computing PRR/ROR in Python before the LLM ever runs, or
(b) letting the LLM guess the arithmetic from its weights, this tool exposes
the cleaned FAERS DataFrames to a PythonAstREPLTool. The agent decides WHEN
a calculation is needed and WRITES the exact pandas code for it; the tool
EXECUTES that code for real, so the numbers are always correct even for
counts in the hundreds of thousands. The LLM authors the logic; it never
produces the final number from memory.
"""

from __future__ import annotations

import re

import pandas as pd
from langchain_core.tools import tool as tool_decorator
from langchain_experimental.tools import PythonAstREPLTool

from src.ingestion.faers_ingest import load_signal_summary

_PAIR_IN_FILTER = re.compile(
    r"drugname\s*==\s*['\"]([^'\"]+)['\"].*?\bpt\s*==\s*['\"]([^'\"]+)['\"]",
    re.I | re.S,
)


def _lookup_python(drug: str, event: str) -> str:
    drug, event = drug.strip().upper(), event.strip().upper()
    return (
        "row = signal_df[(signal_df['drugname']=="
        f"'{drug}') & (signal_df['pt']=='{event}')]\n"
        "print(row[['prr','ror','a_drug_and_event','serious_reports']])"
    )


def _coerce_python_code(python_code: str, query: str) -> str:
    """Accept pandas from Groq, or a filter string from small local models."""
    code = (python_code or "").strip()
    filt = (query or "").strip()
    if code and "signal_df" in code:
        return code
    blob = code or filt
    match = _PAIR_IN_FILTER.search(blob)
    if match:
        return _lookup_python(match.group(1), match.group(2))
    if code:
        return code
    return filt

_STATS_TOOL_DESCRIPTION = """\
Execute Python/pandas code against the FAERS signal summary DataFrame to \
retrieve pre-computed statistics: case counts, PRR, ROR, and serious-outcome \
counts for any drug-event pair.

Available in your namespace:
  signal_df  -- one row per (drug, event) pair, columns:
                drugname, pt (MedDRA event term),
                a_drug_and_event  (unique reports with BOTH drug and event),
                b_drug_no_event   (reports with drug but NOT event),
                c_event_no_drug   (reports with event but NOT drug),
                d_neither         (reports with neither),
                total_reports     (all reports in the database),
                prr               (Proportional Reporting Ratio),
                ror               (Reporting Odds Ratio),
                serious_reports   (subset of a with a serious outcome code).

Always WRITE PANDAS CODE to look up values -- never state numbers from memory.
Example for drug D and event E:
  row = signal_df[
      (signal_df["drugname"] == "D") & (signal_df["pt"] == "E")
  ]
  print(row[["prr","ror","a_drug_and_event","serious_reports"]].to_dict("records"))

Drug names and event terms are UPPERCASE in signal_df (e.g. "DEPO-PROVERA",
"MENINGIOMA"). Use resolve_drug_name / resolve_event_name first if unsure of
the exact spelling. Print all numbers explicitly so they appear in tool output.
"""


def compute_2x2_counts(flat_df: pd.DataFrame, drug: str, event: str) -> dict:
    """Deterministic helper mirroring the formula the LLM is instructed to
    write itself. Exposed for tests (tests/test_stats_code_tool.py) to check
    the LLM-authored code against a known-correct reference implementation,
    and importable inside the REPL namespace if the LLM prefers to call it
    directly rather than re-deriving the 2x2 table each time.
    """
    drug = drug.strip().upper()
    event = event.strip().upper()

    reports_with_drug = set(flat_df.loc[flat_df["drugname"] == drug, "primaryid"])
    reports_with_event = set(flat_df.loc[flat_df["pt"] == event, "primaryid"])
    all_reports = set(flat_df["primaryid"])

    a = len(reports_with_drug & reports_with_event)
    b = len(reports_with_drug - reports_with_event)
    c = len(reports_with_event - reports_with_drug)
    d = len(all_reports - reports_with_drug - reports_with_event)

    prr = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and c > 0 and (c + d) > 0 else float("nan")
    ror = (a * d) / (b * c) if b > 0 and c > 0 else float("nan")

    serious_mask = (
        (flat_df["drugname"] == drug) & (flat_df["pt"] == event) & (flat_df["is_serious"])
    )
    serious_count = int(flat_df.loc[serious_mask, "primaryid"].nunique())

    return {
        "drug": drug,
        "event": event,
        "a_drug_and_event": a,
        "b_drug_no_event": b,
        "c_event_no_drug": c,
        "d_neither": d,
        "total_reports": len(all_reports),
        "prr": prr,
        "ror": ror,
        "serious_reports": serious_count,
    }


def _ensure_clean_output(code: str) -> str:
    """Guarantee the tool output is unambiguous for any LLM size.

    If the last statement is a bare DataFrame expression (no print / to_dict /
    to_string / to_json already applied), we wrap it so the row index is
    stripped and the result is returned as a list-of-dicts.  This prevents
    small models from misreading the numeric FAERS row-ID as a column value.
    """
    import re as _re

    stripped = code.strip()
    lines = stripped.splitlines()
    if not lines:
        return code

    last = lines[-1].strip()
    already_formatted = any(
        kw in last
        for kw in ("print(", "to_dict", "to_string", "to_json", "to_csv", "values")
    )
    # Only wrap bare expressions (no assignment, no existing formatting)
    is_bare_expr = (
        last
        and not last.startswith("#")
        and not already_formatted
        and "=" not in last.split("[")[0]   # not an assignment
    )
    if is_bare_expr:
        # Replace last line with a clean print statement
        lines[-1] = (
            f"_r = ({last})\n"
            "if hasattr(_r, 'reset_index'):\n"
            "    print(_r.reset_index(drop=True).to_dict('records'))\n"
            "else:\n"
            "    print(_r)"
        )
        return "\n".join(lines)

    return code


def build_stats_code_tool(signal_df: pd.DataFrame | None = None):
    """Build the code-execution tool with the pre-aggregated FAERS signal
    summary already bound in its local namespace. If signal_df is not
    supplied, it is loaded from the parquet produced by faers_ingest.py.

    Wrapped as a @tool with a clearly named `python_code` parameter so
    Groq's strict schema validation accepts the tool call without errors.
    """
    if signal_df is None:
        signal_df = load_signal_summary()

    _repl = PythonAstREPLTool(
        locals={
            "signal_df": signal_df,
            "pd": pd,
            "compute_2x2_counts": compute_2x2_counts,
        },
    )

    @tool_decorator("calculate_pv_statistics")
    def stats_tool(python_code: str = "", query: str = "") -> str:
        """Execute Python/pandas code against the FAERS signal summary DataFrame \
to retrieve PRR, ROR, case counts, or serious-outcome stats for a drug-event pair. \
The DataFrame `signal_df` is pre-loaded with columns: drugname, pt, \
a_drug_and_event, b_drug_no_event, c_event_no_drug, d_neither, \
total_reports, prr, ror, serious_reports. \
Prefer argument python_code with pandas, e.g. \
signal_df[(signal_df['drugname']=='ASPIRIN') & (signal_df['pt']=='HAEMORRHAGE')][['prr','ror']]. \
Small models may instead pass query like \
drugname=='DEPO-PROVERA' AND pt=='MENINGIOMA'. \
Always print results explicitly. Drug names and event terms are UPPERCASE."""
        code = _coerce_python_code(python_code, query)
        if not code:
            return (
                "Pass python_code (pandas against signal_df) or query "
                "(drugname=='DRUG' AND pt=='EVENT')."
            )
        clean_code = _ensure_clean_output(code)
        return _repl.run(clean_code)

    return stats_tool
