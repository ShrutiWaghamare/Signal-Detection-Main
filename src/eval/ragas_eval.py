"""Runs one labeled drug-event query through Groq candidate models and
scores them with RAGAS using Groq (not OpenAI) plus local FastEmbed.

Usage:
    python -m src.eval.ragas_eval
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.agent.build_agent import build_pv_agent
from src.config import MODEL_PROVIDERS, provider_for_model
from src.retrieval.signal_retriever import get_embeddings

EVAL_QUERIES_PATH = Path(__file__).parent / "eval_queries.json"
# Same Groq 120B as the main agent. Shares that model's TPM/TPD bucket.
_RAGAS_JUDGE_MODEL = "openai/gpt-oss-120b"
# Only Groq agents in this harness (fits free-tier; skip un-pulled Ollama).
GROQ_EVAL_MODELS = [m for m, p in MODEL_PROVIDERS.items() if p == "groq"]
_SLEEP_S = 8.0


def _extract_contexts(intermediate_steps: list) -> list[str]:
    return [str(observation) for _action, observation in intermediate_steps]


def _ragas_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env before running RAGAS.")
    return ChatGroq(
        model=_RAGAS_JUDGE_MODEL,
        temperature=0,
        api_key=api_key,
        max_retries=2,
        max_tokens=2048,
    )


def run_eval_for_model(model_name: str, queries: list[dict]) -> pd.DataFrame:
    executor = build_pv_agent(model=model_name, provider=provider_for_model(model_name))
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
        time.sleep(_SLEEP_S)
    return pd.DataFrame(rows)


def score_with_ragas(df: pd.DataFrame):
    dataset = Dataset.from_pandas(df)
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_ragas_llm(),
        embeddings=get_embeddings(),
        batch_size=1,
    )


def main() -> None:
    queries = json.loads(EVAL_QUERIES_PATH.read_text())
    comparison = {}
    for model_name in GROQ_EVAL_MODELS:
        print(f"\n=== Evaluating {model_name} ===")
        try:
            df = run_eval_for_model(model_name, queries)
            time.sleep(_SLEEP_S)
            scores = score_with_ragas(df)
            comparison[model_name] = scores
            print(scores)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {model_name}: {exc}")
        time.sleep(_SLEEP_S)

    print("\n=== Comparison summary ===")
    if not comparison:
        print("No models produced RAGAS scores.")
        return
    for model_name, scores in comparison.items():
        print(model_name, scores)


if __name__ == "__main__":
    main()
