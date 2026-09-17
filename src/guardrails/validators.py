"""Post-generation guardrail checks: numeric faithfulness and empty-evidence
handling. Run after the agent produces its final answer, before display.
"""

from __future__ import annotations

import re

# --- number extraction ---------------------------------------------------------
# Do NOT put \s in the character class — it makes the regex greedily consume
# table column-separator whitespace, turning "195309  4972.24041" into one token.
# Instead we collapse "N NNN"-style space-thousands separators in a pre-processing
# step before running the regex.
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def _collapse_space_thousands(text: str) -> str:
    """Collapse space-as-thousands-separator patterns so '82 745.25' → '82745.25'.

    Only collapses a single space followed by exactly three digits — this matches
    the thousands-grouping convention without touching table column separators
    (which use two or more spaces) or unrelated adjacent numbers.
    Applied iteratively to handle chained groups like '1 234 567' → '1234567'.
    """
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(\d)[\s\u00A0\u202F\u2009\u200A](\d{3})(?!\d)", r"\1\2", text)
    return text


def _normalize(n: str) -> str:
    """Strip comma thousands-separators and trailing dots."""
    return n.replace(",", "").strip(".")


def extract_numbers(text: str) -> set[str]:
    """Extract and normalise all numeric tokens from *text*.

    Handles:
    - comma thousands  '4,972.24'  → '4972.24'
    - space thousands  '82 745.25' → '82745.25'
    - plain integers   '819683'    → '819683'
    - decimals         '4972.24041'→ '4972.24041'
    """
    processed = _collapse_space_thousands(text)
    return {_normalize(n) for n in NUMBER_PATTERN.findall(processed) if _normalize(n)}


def _numbers_match(a: str, b: str, tol: float = 0.01) -> bool:
    """True if two normalised numeric strings are equal or within *tol* relative
    tolerance.  This allows LLM rounding (e.g. 82745.248 → 82745.25) without
    triggering false guardrail failures.
    """
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
        denom = max(abs(fb), 1e-9)
        return abs(fa - fb) / denom <= tol
    except ValueError:
        return False


def _is_split_thousands_fragment(n: str, context_numbers: set[str]) -> bool:
    """True if `n` is the tail of a tool number after a thousands group was split.

    Example: answer extraction got '972.24' from '4 972.24' while the tool
    printed 4972.24041 (rounds to 4972.24).
    """
    for c in context_numbers:
        try:
            rounded = f"{float(c):.2f}"
        except ValueError:
            continue
        if rounded.endswith(n) and len(rounded) > len(n):
            prefix = rounded[: len(rounded) - len(n)]
            if prefix.replace(".", "").isdigit():
                return True
    return False


# --- individual checks --------------------------------------------------------

def check_numeric_faithfulness(answer_text: str, tool_output_text: str) -> tuple[bool, list[str]]:
    """Every number in the answer should trace back to something in the
    tool/retrieval outputs (within 1% rounding tolerance).

    Common-number exclusions:
    - Single-digit numbers (0-9) → too generic, ignored.
    - Small numbers ≤ 10 → frequently appear in prose ("step 1", "≥ 3 cases")
      and are exempt from tracing.
    """
    answer_numbers  = extract_numbers(answer_text)
    context_numbers = extract_numbers(tool_output_text)

    unverified = sorted(
        n for n in answer_numbers
        # Skip single chars and small integers — they appear naturally in prose
        if len(n) > 1 and float(n) > 10
        and not any(_numbers_match(n, c) for c in context_numbers)
        and not _is_split_thousands_fragment(n, context_numbers)
    )
    return (len(unverified) == 0, unverified)


def check_empty_evidence(tool_outputs: list[str]) -> bool:
    """True if at least one tool call actually returned usable evidence."""
    return any(
        output and "no matching" not in output.lower() and "error" not in output.lower()
        for output in tool_outputs
    )


# --- main entry point ---------------------------------------------------------

def run_guardrails(answer_text: str, tool_output_text: str, tool_outputs: list[str]) -> dict:
    faithful, unverified_numbers = check_numeric_faithfulness(answer_text, tool_output_text)
    has_evidence = check_empty_evidence(tool_outputs)
    issues = []
    if not faithful:
        issues.append(
            f"Unverified numbers in answer (not in tool output, > 10, not small prose numbers): "
            f"{unverified_numbers}"
        )
    if not has_evidence:
        issues.append("No tool returned usable evidence; the answer should state this explicitly.")
    return {"passed": len(issues) == 0, "issues": issues}
