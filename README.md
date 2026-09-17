# Pharmacovigilance Signal RAG Agent

A tool-calling LangChain agent for exploring FAERS adverse-event reporting
patterns: case counts, PRR/ROR disproportionality statistics, prior signal
evidence, literature, and drug label warnings — all decided and retrieved
per query by the agent itself, not a fixed pipeline. See
`PROJECT_DETAILS.txt` for the full architecture writeup and rationale.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

**Default LLM provider is Groq** (hosted, fast, free tier, no GPU needed —
recommended if your local GPU has limited VRAM):

1. Sign up at https://console.groq.com (no credit card required).
2. Generate an API key.
3. Put it in `.env`: `GROQ_API_KEY=your-key-here`.

That's it — no local model download needed. `src/config.py` already lists
tool-calling-capable Groq models in `CANDIDATE_LLM_MODELS`, with
`llama-3.3-70b-versatile` as the default.

**Optional: local serving via Ollama** (only worth it if you have a GPU
with enough VRAM — roughly 6GB+ for a quantized 7B/8B model):

```bash
ollama pull qwen2.5:7b-instruct
```

Then pass `provider="ollama"` wherever `build_pv_agent(...)` is called, or
select it from the model list built from `LOCAL_CANDIDATE_LLM_MODELS`.

## 2. Add data

Drop raw FAERS ASCII quarter files under:

```
data/raw/<quarter>/DEMO<quarter>.txt
data/raw/<quarter>/DRUG<quarter>.txt
data/raw/<quarter>/REAC<quarter>.txt
data/raw/<quarter>/OUTC<quarter>.txt
```

Add each quarter label (e.g. `"2023Q1"`) to `SUPPORTED_QUARTERS` in
`src/config.py`.

Optionally add PDFs for extra evidence sources:

```
data/external_docs/labels/*.pdf
data/external_docs/literature/*.pdf
```

## 3. Build the data + indexes

```bash
python -m src.ingestion.faers_ingest        # clean + join raw tables once
python -m src.ingestion.build_vector_stores  # build all FAISS indexes
```

## 4. Run

```bash
streamlit run app.py
```

Ask things like:

> What is the PRR and ROR for DEPO-PROVERA and MENINGIOMA?

The UI shows the final structured answer, a pass/fail guardrail check, and
the full tool-call trace (which tool fired, with what input/output).

## 5. Tests

```bash
pytest
```

## 6. Multi-model evaluation

Edit `CANDIDATE_LLM_MODELS` in `src/config.py`, pull each model in Ollama,
then:

```bash
python -m src.eval.ragas_eval
```

Prints a side-by-side RAGAS comparison (faithfulness, answer relevancy,
context precision/recall) across every candidate model on the hand-labeled
queries in `src/eval/eval_queries.json`.

## Project layout

See `PROJECT_DETAILS.txt` for the annotated tech stack and a description of
every file in `src/`.
