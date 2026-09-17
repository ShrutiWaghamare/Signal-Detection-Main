"""Confirms compute_2x2_counts (the reference PRR/ROR implementation that
the LLM is instructed to reproduce as executed pandas code) matches
manually-verified expected values on a small synthetic FAERS-shaped table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.tools.stats_code_tool import compute_2x2_counts


@pytest.fixture
def flat_df() -> pd.DataFrame:
    # 10 reports total.
    # DRUG_A + EVENT_X co-occur in reports 1, 2, 3        -> a = 3
    # DRUG_A without EVENT_X in reports 4, 5               -> b = 2
    # EVENT_X without DRUG_A in reports 6, 7                -> c = 2
    # neither in reports 8, 9, 10                           -> d = 3
    rows = [
        {"primaryid": "1", "drugname": "DRUG_A", "pt": "EVENT_X", "is_serious": True},
        {"primaryid": "2", "drugname": "DRUG_A", "pt": "EVENT_X", "is_serious": False},
        {"primaryid": "3", "drugname": "DRUG_A", "pt": "EVENT_X", "is_serious": True},
        {"primaryid": "4", "drugname": "DRUG_A", "pt": "EVENT_Y", "is_serious": False},
        {"primaryid": "5", "drugname": "DRUG_A", "pt": "EVENT_Y", "is_serious": False},
        {"primaryid": "6", "drugname": "DRUG_B", "pt": "EVENT_X", "is_serious": False},
        {"primaryid": "7", "drugname": "DRUG_B", "pt": "EVENT_X", "is_serious": False},
        {"primaryid": "8", "drugname": "DRUG_B", "pt": "EVENT_Y", "is_serious": False},
        {"primaryid": "9", "drugname": "DRUG_B", "pt": "EVENT_Y", "is_serious": False},
        {"primaryid": "10", "drugname": "DRUG_C", "pt": "EVENT_Z", "is_serious": False},
    ]
    return pd.DataFrame(rows)


def test_counts_2x2_table(flat_df: pd.DataFrame) -> None:
    result = compute_2x2_counts(flat_df, "DRUG_A", "EVENT_X")
    assert result["a_drug_and_event"] == 3
    assert result["b_drug_no_event"] == 2
    assert result["c_event_no_drug"] == 2
    assert result["d_neither"] == 3
    assert result["total_reports"] == 10


def test_prr_matches_manual_calculation(flat_df: pd.DataFrame) -> None:
    result = compute_2x2_counts(flat_df, "DRUG_A", "EVENT_X")
    # PRR = (a / (a+b)) / (c / (c+d)) = (3/5) / (2/5) = 1.5
    assert result["prr"] == pytest.approx(1.5, rel=1e-6)


def test_ror_matches_manual_calculation(flat_df: pd.DataFrame) -> None:
    result = compute_2x2_counts(flat_df, "DRUG_A", "EVENT_X")
    # ROR = (a*d) / (b*c) = (3*3) / (2*2) = 2.25
    assert result["ror"] == pytest.approx(2.25, rel=1e-6)


def test_serious_report_count(flat_df: pd.DataFrame) -> None:
    result = compute_2x2_counts(flat_df, "DRUG_A", "EVENT_X")
    assert result["serious_reports"] == 2


def test_case_insensitive_and_whitespace_tolerant(flat_df: pd.DataFrame) -> None:
    result = compute_2x2_counts(flat_df, "  drug_a ", " event_x  ")
    assert result["a_drug_and_event"] == 3


def test_no_cooccurrence_returns_zero(flat_df: pd.DataFrame) -> None:
    result = compute_2x2_counts(flat_df, "DRUG_C", "EVENT_X")
    assert result["a_drug_and_event"] == 0
