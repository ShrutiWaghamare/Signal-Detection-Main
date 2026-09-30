"""Pharmacovigilance Agent — RAGAS Evaluation Harness.

What this measures (plain English):
  faithfulness        — Did the model only say things that came from the tools?
  answer_relevancy    — Did the model actually answer what was asked?
  context_recall      — Did the model use all the key facts from the tools?

Ground truth is now a real expected answer (not just drug/event names),
so context_recall has something meaningful to measure against.

Usage:
    # Full evaluation across all Groq models
    python -m src.eval.ragas_eval

    # Quick single-query test (saves tokens — use during development)
    python -m src.eval.ragas_eval --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from src.agent.build_agent import build_pv_agent
from src.config import MODEL_PROVIDERS, provider_for_model
from src.guardrails.llm_judge import run_llm_judge
from src.guardrails.validators import run_guardrails
from src.retrieval.signal_retriever import get_embeddings

logger = logging.getLogger(__name__)

EVAL_QUERIES_PATH = Path(__file__).parent / "eval_queries.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Groq 120B as RAGAS judge — same model pool, temperature 0 for determinism
_RAGAS_JUDGE_MODEL = "openai/gpt-oss-120b"

# Rate-limit buffer between API calls (free tier)
_SLEEP_S = 8.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_contexts(intermediate_steps: list) -> list[str]:
    """Pull raw tool outputs from the agent's intermediate steps."""
    return [str(obs) for _action, obs in intermediate_steps]


def _ragas_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return ChatGroq(
        model=_RAGAS_JUDGE_MODEL,
        temperature=0,
        api_key=api_key,
        max_retries=2,
        max_tokens=2048,
    )


def _score_label(score: float) -> str:
    """Convert 0–1 RAGAS score to a human-readable label."""
    if score >= 0.80:
        return "High ✅"
    if score >= 0.55:
        return "Medium ⚠️"
    return "Low ❌"


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def run_eval_for_model(
    model_name: str,
    queries: list[dict],
    provider: str | None = None,
) -> pd.DataFrame:
    """Run every query through the agent and collect answers + contexts."""
    prov = provider or provider_for_model(model_name)
    executor = build_pv_agent(model=model_name, provider=prov)
    rows = []

    for item in queries:
        logger.info("Running query [%s] on model %s", item["id"], model_name)
        try:
            result = executor.invoke({"input": item["query"]})
        except Exception as exc:
            logger.warning("Agent failed for %s / %s: %s", model_name, item["id"], exc)
            rows.append({
                "id": item["id"],
                "category": item.get("category", "unknown"),
                "question": item["query"],
                "answer": f"[ERROR: {exc}]",
                "contexts": [""],
                "ground_truth": item.get("ground_truth", ""),
                "guardrail_passed": False,
                "judge_verdict": "error",
                "judge_issues": [str(exc)],
            })
            time.sleep(_SLEEP_S)
            continue

        answer = result.get("output", "")
        contexts = _extract_contexts(result.get("intermediate_steps", []))
        tool_output_combined = "\n".join(contexts)

        # ── Guardrails ──────────────────────────────────────────────────────
        regex_result = run_guardrails(answer, tool_output_combined, contexts)
        judge_result = run_llm_judge(answer, contexts)
        guardrail_ok = regex_result["passed"] and judge_result.get("overall_verdict") == "pass"

        rows.append({
            "id": item["id"],
            "category": item.get("category", "unknown"),
            "question": item["query"],
            "answer": answer,
            "contexts": contexts if contexts else [""],
            "ground_truth": item.get("ground_truth", ""),
            "guardrail_passed": guardrail_ok,
            "judge_verdict": judge_result.get("overall_verdict", "error"),
            "judge_issues": judge_result.get("issues", []),
        })
        time.sleep(_SLEEP_S)

    return pd.DataFrame(rows)


def score_with_ragas(df: pd.DataFrame) -> dict:
    """Run RAGAS metrics and return per-metric averages."""
    dataset = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=_ragas_llm(),
        embeddings=get_embeddings(),
        batch_size=1,
    )
    df_scores = scores.to_pandas()
    return {
        "faithfulness":      round(float(df_scores["faithfulness"].mean()), 3),
        "answer_relevancy":  round(float(df_scores["answer_relevancy"].mean()), 3),
        "context_recall":    round(float(df_scores["context_recall"].mean()), 3),
    }


# ---------------------------------------------------------------------------
# Manager-friendly report
# ---------------------------------------------------------------------------

def print_manager_report(all_results: dict[str, dict]) -> None:
    """Print a plain-English comparison table a non-technical manager can read."""

    print("\n" + "=" * 70)
    print("  PHARMACOVIGILANCE AGENT — EVALUATION REPORT")
    print("=" * 70)
    print(
        "Scores are 0–1. Higher = better.\n"
        "  Faithfulness   = Did the model only use facts from the tools?\n"
        "  Ans. Relevancy = Did it actually answer the question asked?\n"
        "  Context Recall = Did it include all key facts from the tools?\n"
        "  Guardrails     = Did it pass all safety checks (numbers, language)?\n"
    )

    # Header
    col = "{:<30} {:>14} {:>16} {:>16} {:>12}"
    print(col.format("Model", "Faithfulness", "Ans.Relevancy", "CtxRecall", "Guardrails"))
    print("-" * 90)

    winner = None
    best_avg = -1.0

    for model, res in all_results.items():
        s = res.get("ragas_scores", {})
        faith  = s.get("faithfulness", 0.0)
        relev  = s.get("answer_relevancy", 0.0)
        recall = s.get("context_recall", 0.0)
        gr_pct = res.get("guardrail_pass_rate", 0.0)

        avg = (faith + relev + recall) / 3
        if avg > best_avg:
            best_avg = avg
            winner = model

        print(col.format(
            model[:30],
            f"{faith:.2f} ({_score_label(faith)[:4].strip()})",
            f"{relev:.2f} ({_score_label(relev)[:4].strip()})",
            f"{recall:.2f} ({_score_label(recall)[:4].strip()})",
            f"{gr_pct*100:.0f}%",
        ))

    print("-" * 90)
    if winner:
        print(f"\n🏆  Best overall model: {winner}  (avg RAGAS score: {best_avg:.2f})")

    print("\n  WHAT THIS MEANS FOR THE POC:")
    for model, res in all_results.items():
        s = res.get("ragas_scores", {})
        faith  = s.get("faithfulness", 0.0)
        relev  = s.get("answer_relevancy", 0.0)
        recall = s.get("context_recall", 0.0)
        gr_pct = res.get("guardrail_pass_rate", 0.0)
        issues = res.get("sample_issues", [])

        verdict = []
        if faith >= 0.8:
            verdict.append("stays within tool facts")
        else:
            verdict.append("occasionally adds claims not in tools")
        if relev >= 0.8:
            verdict.append("answers the question well")
        else:
            verdict.append("sometimes goes off-topic")
        if recall >= 0.8:
            verdict.append("recalls key numbers accurately")
        else:
            verdict.append("may miss key figures")
        if gr_pct == 1.0:
            verdict.append("passed all safety guardrails")
        else:
            verdict.append(f"failed guardrails on {int((1-gr_pct)*100)}% of queries")

        print(f"\n  {model}:")
        print(f"    → {'; '.join(verdict)}.")
        if issues:
            print(f"    ⚠ Sample issues: {issues[0]}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(quick: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    all_queries: list[dict] = json.loads(EVAL_QUERIES_PATH.read_text())

    if quick:
        # Single stats query only — saves ~80% of tokens during development
        queries = [q for q in all_queries if q.get("id") == "pv_001"]
        models_to_eval = {
            m: provider_for_model(m)
            for m in list(MODEL_PROVIDERS.keys())[:1]   # first Groq model only
        }
        logger.info("QUICK MODE: 1 query × 1 model to minimise token usage.")
    else:
        queries = all_queries
        models_to_eval = {m: provider_for_model(m) for m in MODEL_PROVIDERS}

    all_results: dict[str, dict] = {}

    for model_name, provider in models_to_eval.items():
        print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
        try:
            df = run_eval_for_model(model_name, queries, provider=provider)
            time.sleep(_SLEEP_S)

            ragas_scores = score_with_ragas(df)
            guardrail_pass_rate = float(df["guardrail_passed"].mean())
            sample_issues = [
                issue
                for issues in df["judge_issues"].tolist()
                for issue in (issues or [])
            ]

            all_results[model_name] = {
                "ragas_scores": ragas_scores,
                "guardrail_pass_rate": guardrail_pass_rate,
                "sample_issues": sample_issues[:3],
                "detail_df": df,
            }

            print(f"RAGAS scores: {ragas_scores}")
            print(f"Guardrail pass rate: {guardrail_pass_rate:.0%}")

        except Exception as exc:
            logger.warning("Skipping %s: %s", model_name, exc)
            all_results[model_name] = {
                "ragas_scores": {},
                "guardrail_pass_rate": 0.0,
                "sample_issues": [str(exc)],
            }
        time.sleep(_SLEEP_S)

    print_manager_report(all_results)

    # Save results to JSON for future reference
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}{'_quick' if quick else ''}.json"
    serialisable = {
        model: {k: v for k, v in res.items() if k != "detail_df"}
        for model, res in all_results.items()
    }
    out_path.write_text(json.dumps(serialisable, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 1 query × 1 model only (saves tokens during development)",
    )
    args = parser.parse_args()
    main(quick=args.quick)
