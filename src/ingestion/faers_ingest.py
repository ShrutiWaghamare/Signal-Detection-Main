"""Loads and cleans raw FAERS ASCII tables, and performs the DEMO/DRUG/REAC/
OUTC join exactly once here. Everything downstream (the stats code tool, the
signal retriever) reads the flattened parquet this module produces -- it
never re-joins the raw tables itself.

FAERS structure (FDA's schema, not a choice made by this project):
  DEMO  -- one row per case report: primaryid, case demographics
  DRUG  -- one row per drug per case report: primaryid, drugname, role_cod
  REAC  -- one row per reaction per case report: primaryid, pt (MedDRA
           preferred term for the adverse event)
  OUTC  -- one row per outcome per case report: primaryid, outc_cod
           (DE=death, HO=hospitalization, LT=life-threatening, etc.)

A single fact -- "Drug X was associated with Reaction Y in report Z" -- only
exists once these are joined on primaryid. This module does that join once
at ingestion time and persists the result, so the join cost is paid once,
not on every query.

Usage:
    python -m src.ingestion.faers_ingest
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    DEMO_PARQUET_PATH,
    DRUG_PARQUET_PATH,
    FLAT_PARQUET_PATH,
    MIN_SIGNAL_COOCCURRENCE,
    OUTC_PARQUET_PATH,
    RAW_DIR,
    REAC_PARQUET_PATH,
    SIGNAL_SUMMARY_PATH,
    SUPPORTED_QUARTERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# FAERS ASCII files are '$'-delimited; column names are lowercased on load
# so the same code works whether the source used upper/lower case headers.
_DELIM = "$"

_DEMO_COLS = ["primaryid", "caseid", "age", "age_cod", "sex", "event_dt", "fda_dt", "occr_country"]
_DRUG_COLS = ["primaryid", "caseid", "drug_seq", "drugname", "role_cod", "prod_ai"]
_REAC_COLS = ["primaryid", "caseid", "pt", "drug_rec_act"]
_OUTC_COLS = ["primaryid", "caseid", "outc_cod"]


def _read_ascii_table(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    """Read one FAERS ASCII file, keep only known columns (missing ones are
    filled with NA so schema stays stable across quarters that add/drop
    optional columns), and lowercase every column name."""
    df = pd.read_csv(path, sep=_DELIM, dtype=str, low_memory=False, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[expected_cols]


def _load_quarter(table: str, quarter: str, expected_cols: list[str]) -> pd.DataFrame:
    """Locate and read a single table for a single quarter.

    Quarter directory discovery: FDA zip extractions use varying folder names
    (e.g. the zip for "2025Q1" extracts to "faers_ascii_2025q1" rather than
    "2025Q1").  We glob RAW_DIR for any subdirectory whose name contains the
    quarter string (case-insensitive) so neither the files nor the config
    quarter labels need to be renamed when the FDA changes its naming scheme.

    File glob inside that directory: <TABLE>*.txt (upper then lower) so both
    DEMO25Q1.txt and demo25q1.txt are matched regardless of extraction casing.
    """
    # Find the quarter subdirectory -- name may have a prefix like faers_ascii_
    quarter_upper = quarter.upper()
    dir_candidates = [
        p for p in RAW_DIR.iterdir()
        if p.is_dir() and quarter_upper in p.name.upper()
    ]
    if not dir_candidates:
        raise FileNotFoundError(
            f"No directory found for quarter '{quarter}' under {RAW_DIR}. "
            f"Expected a folder whose name contains '{quarter}' "
            f"(e.g. faers_ascii_{quarter.lower()} or {quarter})."
        )
    quarter_dir = dir_candidates[0]

    matches = (
        list(quarter_dir.glob(f"{table.upper()}*.txt"))
        + list(quarter_dir.glob(f"{table.lower()}*.txt"))
        + list(quarter_dir.glob(f"{table.upper()}*.TXT"))
    )
    if not matches:
        raise FileNotFoundError(
            f"No {table} file found for quarter {quarter} under {quarter_dir}. "
            f"Expected e.g. {quarter_dir / (table.upper() + '25Q1.txt')}"
        )
    return _read_ascii_table(matches[0], expected_cols)


def load_raw_tables(quarters: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load and concatenate DEMO/DRUG/REAC/OUTC across the given quarters
    (defaults to config.SUPPORTED_QUARTERS)."""
    quarters = quarters or SUPPORTED_QUARTERS
    if not quarters:
        raise ValueError(
            "No quarters configured. Add quarter labels to SUPPORTED_QUARTERS "
            "in src/config.py (e.g. ['2023Q1', '2023Q2']) and drop the raw "
            "FAERS ASCII files under data/raw/<quarter>/."
        )

    tables: dict[str, list[pd.DataFrame]] = {"demo": [], "drug": [], "reac": [], "outc": []}
    for quarter in quarters:
        logger.info("Loading quarter %s", quarter)
        tables["demo"].append(_load_quarter("demo", quarter, _DEMO_COLS))
        tables["drug"].append(_load_quarter("drug", quarter, _DRUG_COLS))
        tables["reac"].append(_load_quarter("reac", quarter, _REAC_COLS))
        tables["outc"].append(_load_quarter("outc", quarter, _OUTC_COLS))

    return {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}


def clean_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize drug names and event terms so joins/lookups aren't broken
    by casing or stray whitespace. Deterministic, no LLM involved."""
    demo, drug, reac, outc = tables["demo"], tables["drug"], tables["reac"], tables["outc"]

    drug = drug.copy()
    drug["drugname"] = drug["drugname"].astype(str).str.strip().str.upper()
    drug = drug.dropna(subset=["primaryid", "drugname"])

    reac = reac.copy()
    reac["pt"] = reac["pt"].astype(str).str.strip().str.upper()
    reac = reac.dropna(subset=["primaryid", "pt"])

    outc = outc.copy()
    outc["outc_cod"] = outc["outc_cod"].astype(str).str.strip().str.upper()

    demo = demo.copy()
    demo = demo.drop_duplicates(subset=["primaryid"])

    return {"demo": demo, "drug": drug, "reac": reac, "outc": outc}


def build_flat_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join DRUG + REAC + OUTC on primaryid and return a slim flat table.

    Columns kept: primaryid, drugname, pt, outc_codes, is_serious.
    Demographic columns (age, sex, country, etc.) are intentionally omitted:
    the stats code tool (PRR/ROR/case counts) only uses the five columns above,
    and including demo in the join roughly quadruples memory consumption for no
    downstream benefit. demo.parquet is still written separately by run_ingestion
    so demographic analysis can be added later without re-ingesting raw files.

    This is the only place in the codebase that joins raw FAERS tables.
    """
    drug, reac, outc = tables["drug"], tables["reac"], tables["outc"]

    # Inner join: only keep (primaryid, drugname, pt) triples where both a
    # drug report and a reaction report exist for the same case.
    drug_reac = drug[["primaryid", "drugname"]].merge(
        reac[["primaryid", "pt"]], on="primaryid", how="inner"
    )

    # A report can have multiple outcomes; collapse to a single serious flag
    # plus a joined outcome-codes string rather than exploding rows further.
    SERIOUS_CODES = {"DE", "HO", "LT", "DS", "CA", "RI"}
    outc_agg = (
        outc.groupby("primaryid")["outc_cod"]
        .agg(lambda codes: ",".join(sorted(set(codes.dropna()))))
        .rename("outc_codes")
        .reset_index()
    )
    outc_agg["is_serious"] = outc_agg["outc_codes"].apply(
        lambda codes: any(code in SERIOUS_CODES for code in codes.split(","))
    )

    flat = drug_reac.merge(outc_agg, on="primaryid", how="left")
    flat["outc_codes"] = flat["outc_codes"].fillna("")
    flat["is_serious"] = flat["is_serious"].fillna(False)

    return flat


def load_signal_summary() -> pd.DataFrame:
    """Load the pre-aggregated signal summary produced by run_ingestion.
    Raises FileNotFoundError if ingestion has not been run yet."""
    if not SIGNAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SIGNAL_SUMMARY_PATH} does not exist yet. "
            "Run `python -m src.ingestion.faers_ingest` first."
        )
    return pd.read_parquet(SIGNAL_SUMMARY_PATH)


def run_ingestion(quarters: list[str] | None = None) -> pd.DataFrame:
    """Full pipeline: load raw -> clean -> join once -> persist all tables.

    Per-table parquets (demo/drug/reac/outc) are written from all quarters
    loaded together -- they are much smaller than the joined flat table.

    The flat table is built and written ONE QUARTER AT A TIME using a
    pyarrow ParquetWriter so the join never holds more than one quarter in
    RAM simultaneously. 6 quarters × ~1-2 GB per quarter is manageable;
    joining all 6 at once was causing a 7+ GB allocation failure.
    """
    quarters = quarters or SUPPORTED_QUARTERS

    # ---- per-table parquets (all quarters, no join, smaller footprint) ----
    raw_tables = load_raw_tables(quarters)
    cleaned_all = clean_tables(raw_tables)
    cleaned_all["demo"].to_parquet(DEMO_PARQUET_PATH, index=False)
    cleaned_all["drug"].to_parquet(DRUG_PARQUET_PATH, index=False)
    cleaned_all["reac"].to_parquet(REAC_PARQUET_PATH, index=False)
    cleaned_all["outc"].to_parquet(OUTC_PARQUET_PATH, index=False)
    logger.info("Wrote cleaned per-table parquet files to %s", DEMO_PARQUET_PATH.parent)
    del raw_tables, cleaned_all  # free before the join pass

    # ---- flat table + signal summary: one quarter at a time ----
    # The full flat table (143M+ rows) cannot be held in RAM.
    # We stream each quarter's flat data directly to parquet via pyarrow, and
    # simultaneously accumulate the per-(drug, event) counts needed for PRR/ROR
    # so we never need to reload the flat table later.
    from collections import defaultdict
    pair_a: dict[tuple, int] = defaultdict(int)        # (drug, pt) -> co-reports
    pair_serious: dict[tuple, int] = defaultdict(int)  # (drug, pt) -> serious
    drug_total: dict[str, int] = defaultdict(int)      # drug -> total unique reports
    event_total: dict[str, int] = defaultdict(int)     # event -> total unique reports
    grand_total = 0                                     # all unique primaryids

    flat_writer: pq.ParquetWriter | None = None
    total_rows = 0

    for quarter in quarters:
        logger.info("Building flat table for quarter %s", quarter)
        raw_q = load_raw_tables([quarter])
        cleaned_q = clean_tables(raw_q)
        flat_q = build_flat_table(cleaned_q)
        total_rows += len(flat_q)

        # ---- accumulate signal stats (one-pass groupby, stays small) ----
        q_pids = flat_q["primaryid"].nunique()
        grand_total += q_pids

        for (drug, pt), grp in flat_q.groupby(["drugname", "pt"]):
            key = (drug, pt)
            pair_a[key] += grp["primaryid"].nunique()
            pair_serious[key] += grp.loc[grp["is_serious"], "primaryid"].nunique()

        for drug, grp in flat_q.groupby("drugname"):
            drug_total[drug] += grp["primaryid"].nunique()

        for pt, grp in flat_q.groupby("pt"):
            event_total[pt] += grp["primaryid"].nunique()

        # ---- stream flat quarter to parquet ----
        arrow_table = pa.Table.from_pandas(flat_q, preserve_index=False)
        if flat_writer is None:
            flat_writer = pq.ParquetWriter(str(FLAT_PARQUET_PATH), arrow_table.schema)
        flat_writer.write_table(arrow_table)
        del raw_q, cleaned_q, flat_q, arrow_table

    if flat_writer is not None:
        flat_writer.close()
    logger.info("Wrote flattened table (%d rows) to %s", total_rows, FLAT_PARQUET_PATH)

    # ---- build and save signal summary (small, fits in RAM) ----
    summary_rows = []
    for (drug, pt), a in pair_a.items():
        if a < MIN_SIGNAL_COOCCURRENCE:
            continue
        ab = drug_total[drug]
        ac = event_total[pt]
        b = ab - a
        c = ac - a
        d = grand_total - ab - ac + a
        prr = (a / ab) / (c / (c + d)) if ab > 0 and c > 0 and (c + d) > 0 else float("nan")
        ror = (a * d) / (b * c) if b > 0 and c > 0 else float("nan")
        summary_rows.append({
            "drugname": drug,
            "pt": pt,
            "a_drug_and_event": a,
            "b_drug_no_event": b,
            "c_event_no_drug": c,
            "d_neither": d,
            "total_reports": grand_total,
            "prr": round(prr, 6),
            "ror": round(ror, 6),
            "serious_reports": pair_serious[(drug, pt)],
        })

    signal_summary = pd.DataFrame(summary_rows)
    signal_summary.to_parquet(SIGNAL_SUMMARY_PATH, index=False)
    logger.info(
        "Wrote signal summary (%d pairs, %d total reports) to %s",
        len(signal_summary), grand_total, SIGNAL_SUMMARY_PATH,
    )

    return signal_summary


def load_flat_table() -> pd.DataFrame:
    """Convenience loader used by tools/retrievers: read the already-built
    flattened parquet instead of re-running ingestion."""
    if not FLAT_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"{FLAT_PARQUET_PATH} does not exist yet. Run "
            "`python -m src.ingestion.faers_ingest` first."
        )
    return pd.read_parquet(FLAT_PARQUET_PATH)


if __name__ == "__main__":
    run_ingestion()
