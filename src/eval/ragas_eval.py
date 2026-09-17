"""Runs the hand-labeled query set through each candidate model and scores
retrieval/generation quality with RAGAS, producing a side-by-side model
comparison table.

Usage: python -m src.eval.ragas_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before anything reads GROQ_API_KEY from the environment

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.agent.build_agent import build_pv_agent
from src.config import CANDIDATE_LLM_MODELS

EVAL_QUERIES_PATH = Path(__file__).parent / "eval_queries.json"


def _extract_contexts(intermediate_steps: list) -> list[str]:
    return [str(observation) for _action, observation in intermediate_steps]


def run_eval_for_model(model_name: str, queries: list[dict]) -> pd.DataFrame:
    executor = build_pv_agent(model=model_name)
    rows = []
    for item in queries:
        result = executor.invoke({"input": item["query"]})
        answer_text = result.get("output", "")
        contexts = _extract_contexts(result.get("intermediate_steps", []))
        rows.append(
            {
                "question": item["query"],
                "answer": answer_text,
                "contexts": contexts if contexts else [""],
                "ground_truth": f"{item.get('expected_drug', '')} {item.get('expected_event', '')}".strip(),
            }
        )
    return pd.DataFrame(rows)


def score_with_ragas(df: pd.DataFrame):
    dataset = Dataset.from_pandas(df)
    return evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])


def main() -> None:
    queries = json.loads(EVAL_QUERIES_PATH.read_text())
    comparison = {}
    for model_name in CANDIDATE_LLM_MODELS:
        print(f"\n=== Evaluating {model_name} ===")
        try:
            df = run_eval_for_model(model_name, queries)
            scores = score_with_ragas(df)
            comparison[model_name] = scores
            print(scores)
        except Exception as exc:  # noqa: BLE001 - eval harness should keep going on model failure
            print(f"Skipping {model_name}: {exc}")

    print("\n=== Comparison summary ===")
    for model_name, scores in comparison.items():
        print(model_name, scores)


if __name__ == "__main__":
    main()
