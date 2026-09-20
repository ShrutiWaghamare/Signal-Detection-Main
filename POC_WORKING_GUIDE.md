# PV Signal Console — Complete working guide (this POC)

Single reference: purpose, tech stack, data, RAG, LangChain, tools, stats, UI, Groq/Ollama, guardrails, evals, commands, demo, limits. Matches the **current repo**, not the older `PROJECT_DETAILS.txt` notes.

---

## 1. What this POC is

A **tool-calling LangChain agent** for **FAERS pharmacovigilance signal detection**.

Analyst asks about a **drug–event** pair (or free text). The agent **chooses tools**. **Pandas** looks up **precomputed PRR/ROR** on `signal_df`. **Hybrid RAG** (BM25 + FAISS + RRF) returns **top-3** literature/label snippets. The LLM **writes the narrative**. **Guardrails** check numbers came from tools and the text does not claim causation.

**Not:** dump FAERS into the LLM. **Not:** LLM computes PRR from weights. **Not:** a fixed retrieve-then-template pipeline with no agent.

**One line for a technical manager:** *The LLM orchestrates and explains; code owns the 2×2; RAG owns unstructured evidence; guardrails gate display.*

---

## 2. Problem this solves (old vs new)

**Old pattern:** Precompute a report → one FAISS index → LLM only rewrites retrieved rows. Retrieval searched **already-printed numbers**. The model had nothing real to do.

**This POC:** Per query the agent decides:

- statistics → **code tool** on `signal_df`
- literature / labels / signal summaries → **retrievers**
- enough evidence or another tool call

Numbers still come from **executed pandas**, not from the transformer.

---

## 3. Tech stack (what is actually used)

### Orchestration
| Package | Role |
|---|---|
| langchain / langchain-core 0.3 | Agent, tools, prompts, runnables |
| langchain-community | FAISS, BM25, PDF loader |
| langchain-experimental | `PythonAstREPLTool` |
| langgraph (in requirements) | **Not the live loop.** Live loop = `create_tool_calling_agent` + `AgentExecutor` |

### LLM serving
| Package | Role |
|---|---|
| langchain-groq | Default hosted chat + tool calling |
| langchain-ollama | Local comparison models |
| python-dotenv | `GROQ_API_KEY` |
| langchain-openai | Optional vLLM path (commented) |

**Default:** Groq `openai/gpt-oss-120b`, temperature **0**.  
**Also Groq:** `qwen/qwen3.8-27b`.  
**Judge / RAGAS judge:** `openai/gpt-oss-20b` (separate TPM bucket).  
**Ollama:** `qwen2.5:3b` (pulled on this machine), `llama3.1:8b` (dropdown; not pulled → 404).

### Retrieval / embeddings
| Piece | Setting |
|---|---|
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` (ONNX, no PyTorch) |
| Vector DB | FAISS (`faiss-cpu`), **three** indexes |
| Sparse | `rank_bm25` |
| Fusion | Reciprocal Rank Fusion, **k = 60** |
| Top-k | 10 dense, 10 sparse, 10 fused, **3 final** |
| Name match | RapidFuzz (drug/event vocabulary) |
| PDF | pypdf + RecursiveCharacterTextSplitter **800 / 120** |

Reranker file still names a cross-encoder; **runtime rerank is cosine similarity** on the same FastEmbed vectors (no torch).

### Data / UI / eval
| Piece | Role |
|---|---|
| pandas, pyarrow | FAERS join, parquet, `signal_df` |
| Streamlit 1.41 | PV Signal Console |
| pydantic | `SignalAnswer` schema (causal filter); live path is regex + judge |
| ragas 0.2.10, datasets | Offline RAGAS (Groq judge + FastEmbed) |
| pytest | Stats formula + retrieval tests |

### Hardware note (this laptop)
- RAM ~16 GB  
- RTX 3050 **4 GB VRAM** → cannot host 120B; Groq is required for the primary model  
- Ollama 3B fits; 8B will CPU/RAM offload and is slow  

---

## 4. Folder map

```
app.py                          Streamlit entry
requirements.txt
.env / .env.example             GROQ_API_KEY
INTERVIEW_QA.md                 Interview Q&A
POC_WORKING_GUIDE.md            This file

src/config.py                   Paths, quarters, models, chunk/top-k, disclaimer

src/ingestion/
  faers_ingest.py               ASCII → parquet + signal_summary (PRR/ROR)
  pdf_ingest.py                 PDF load + chunk
  build_vector_stores.py        Build/save 3 FAISS indexes

src/tools/
  stats_code_tool.py            calculate_pv_statistics (pandas REPL)
  signal_lookup_tool.py         resolve_drug_name / resolve_event_name

src/retrieval/
  signal_retriever.py           HybridRetriever + RRF + embeddings
  literature_retriever.py       Same hybrid + filename/event prefer
  label_retriever.py            Labels corpus
  reranker.py                   Cosine rerank

src/agent/
  llm_provider.py               groq | ollama | vllm
  prompts.py                    Tool policy + answer format
  build_agent.py                Register tools, AgentExecutor

src/guardrails/
  validators.py                 Numeric faithfulness, empty evidence
  llm_judge.py                  LLM-as-judge (20B)
  schemas.py                    Pydantic SignalAnswer

src/eval/
  eval_queries.json             RAGAS query set (currently 1 pair)
  ragas_eval.py                 Groq RAGAS harness

tests/
  test_stats_code_tool.py
  test_retrieval.py

data/
  raw/                          FAERS ASCII quarters
  processed/                    parquet including signal_summary.parquet
  external_docs/literature/     PDFs (Depo, Amiodarone, Warfarin ICH, …)
  vector_stores/                signals/, literature/, labels/
```

Quarters in config: **2026Q1, 2026Q2**.

Literature PDFs (examples):  
`Depo-Provera.pdf`, `Amiodarone-Pulmonary toxicity.pdf`, `Warfarin-Intracranial Haemorrhage.pdf`, `FDA Limitations.pdf`, `Good Pharmacovigilance.pdf`.

---

## 5. Offline pipeline (once / when data changes)

```
FAERS ASCII (DEMO, DRUG, REAC, OUTC)
        │  pandas join on primaryid
        ▼
  faers_flat.parquet
        │  unique cases per (drugname, pt)
        ▼
  2×2: a, b, c, d
  PRR = (a/(a+b)) / (c/(c+d))
  ROR = (a*d) / (b*c)
        ▼
  signal_summary.parquet  →  loaded as signal_df
```

**How pairs stay correct when drug and event are on different rows:** same **`primaryid`**. Count **distinct cases**, not exploded join rows.

`MIN_SIGNAL_COOCCURRENCE` in config is **500**; if a pair with *n*=4 appears in `signal_df`, that parquet was built under a different/lower cutoff. Always read the **row in parquet**, not the constant, in a demo.

PDFs:

```
PDF → chunks 800/120 → embed bge-small → FAISS save
```

After adding a PDF: rebuild **literature** index, **restart Streamlit** (`get_agent` is cached).

```powershell
python -m src.ingestion.faers_ingest
python -m src.ingestion.build_vector_stores
.\.venv\Scripts\python.exe -c "from src.ingestion.build_vector_stores import build_literature_index; build_literature_index()"
```

---

## 6. Online query path

1. Streamlit: Signal lookup (drug + event) or Query (free text).  
2. Model dropdown → `provider_for_model` → Groq or Ollama.  
3. Signal mode builds a query that skips resolve (UI already fuzzy-matched) and asks for stats + literature.  
4. `AgentExecutor.invoke`  
5. Tools run; observations go back to the LLM.  
6. Final prose.  
7. `run_guardrails` (regex) + `run_llm_judge` (20B).  
8. UI: badge/table from **`signal_df` row** (signal mode), supporting evidence (collapsed), standard interpretation block, limitations, audit trail.

**Strength rule (UI + prompt):** PRR ≥ **10** Strong, ≥ **5** Moderate, ≥ **2** Weak, else no signal.

**Disclaimer:** FAERS disproportionality ≠ confirmed causal relationship; qualified PV review required.

---

## 7. Tools (registered)

| Tool | What it does |
|---|---|
| `resolve_drug_name` | Fuzzy/exact FAERS drug string |
| `resolve_event_name` | Fuzzy/exact MedDRA PT |
| `calculate_pv_statistics` | Execute pandas vs `signal_df`. Prefers `python_code`. Also accepts `query` like `drugname=='X' AND pt=='Y'` (small local models). |
| `search_signal_evidence` | Hybrid over pair summaries. **May be a different pair.** Never the PRR source for a named pair. |
| `search_literature` | Hybrid over literature PDFs; skip cosine rerank; filename/event boost |
| `search_drug_label` | Label PDFs if index exists |

Stats tool namespace: `signal_df`, `pd`, `compute_2x2_counts` (reference formula for tests; live path is **lookup** of precomputed columns).

---

## 8. RAG in this POC

**Agentic RAG:** retrieve is a **tool**, not a hidden always-on prepend of the whole corpus.

**Hybrid:** BM25 (keywords / MedDRA / drug strings) + FAISS (semantics) → RRF → top 3 scored `[relevance x.xxx]` snippets to the LLM.

**Why three indexes:** structured signal text ≠ long PDF prose.

**Chunking:** RecursiveCharacterTextSplitter, **800 characters**, **120 overlap** (not LLM tokens). bge-small typically embeds ≤ ~512 tokens; 800 chars fit.

**FAISS has no token cap.** We index this corpus only.

**Why the UI does not show “Top 1/2/3” as a list:** the tool **returns three** snippets. **Supporting evidence** keeps **1–2 event-related sentences**. Full three: **audit trail** and **terminal log**.

**PRR is never taken from FAISS** for a named pair.

---

## 9. Why pandas / why not the LLM for PRR

PRR/ROR are **safety numbers**. LLMs mis-count, mix columns (e.g. *b*=619 reported as *a*), invent thresholds (e.g. “500”).

pandas = exact filter/count on a table. Ingestion does the 2×2 **once**; query time is **O(lookup)**.

**Non-technical:** Excel COUNTIFS vs asking someone to remember the warehouse.  
**Technical:** LLM samples tokens; REPL executes AST on `signal_df`.

The LLM still **uses** the stats: it **must** copy tool figures and apply ≥10/≥5/≥2. Guardrails check membership of numbers in tool text.

---

## 10. LangChain wiring

- `build_llm(model, provider)` → ChatGroq / ChatOllama / ChatOpenAI(vLLM)  
- `AGENT_PROMPT` = system policy + `{input}` + `agent_scratchpad`  
- `create_tool_calling_agent` + `AgentExecutor(verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)`  
- UI `get_agent(model)` is `@st.cache_resource` — restart after prompt/index/tool changes  

---

## 11. Guardrails vs evals

**Not the same.**

| | Guardrail | Eval |
|---|---|---|
| When | Every answer, before/on display | Later / offline |
| Job | Gate this response | Score models/pipeline |
| Here | Prompt policy; regex faithfulness; empty evidence; traceability %; LLM judge; UI table from parquet | pytest; RAGAS harness |

They **overlap** on faithfulness because PV numbers must be grounded.

**No “confidence score” in the codebase.** Closest: retrieval **relevance**, **traceability %**, fuzzy **match %**, PRR **strength**. Do not present LLM softmax as calibrated confidence.

**LLM judge** sees only the first ~200 chars of each tool output → can **false-fail** literature % (e.g. 17%/10% in a later chunk) while regex on **full** text **passes**.

---

## 12. RAGAS (how to run, what it means)

**Full form:** Retrieval Augmented Generation Assessment.

```powershell
cd "E:\AIML Projects\Pharmacovigilance Signal Detection Main"
.venv\Scripts\python.exe -m src.eval.ragas_eval
```

- Queries: currently **one** pair — DEPO-PROVERA / MENINGIOMA  
- Agents: Groq 120B and Qwen 3.8 only  
- RAGAS LLM: Groq **20B** (no OpenAI key)  
- Embeddings: local FastEmbed  
- Metrics: faithfulness, answer_relevancy, context_precision, context_recall  
- `batch_size=1`, sleeps to reduce 429  

**Observed:** relevancy ~0.88 useful. Faithfulness **nan** when 20B 429 / `max_tokens` cut off. Context precision/recall **misread** pandas tables vs a ground-truth string of drug+event names. **RAGAS does not replace the stats tool.**

Wait ~15 minutes after heavy 120B use; do not run Streamlit 120B at the same time.

---

## 13. Groq vs Ollama

**Groq:** hosted inference API. You send chat + tool schemas. You **do not** upload Llama or FAERS. Free org: ~30 RPM, ~8k **TPM**, ~200k **TPD**, **per model id**. 429 = quota, not a bill. 120B full ≠ Qwen 3.8 empty.

**Ollama:** `localhost:11434`, weights on disk (`ollama list` / `ollama pull`).

Same FAERS tools. Only the chat backend changes.

**Which performed better here:** Groq **120B** >> Groq **Qwen 3.8** >> Ollama **qwen2.5:3b** (calls tools; wrong strength / invented 500). Llama 70B / Qwen 3.6 Groq IDs **404** on this key.

---

## 14. UI (Streamlit)

- Left: mode, drug/event or question, model (label includes provider), history  
- Centre: title, match %, signal badge + bar, statistical evidence table, supporting evidence, AI interpretation (standardized headings), static limitations/review, guardrail banner  
- Right: audit trail + evaluation strip  

Interpretation is **reassembled** (stats from row/tools, literature from chunks, a/b/c from model) so Qwen and 120B **look** the same even if prose differs.

---

## 15. Commands

```powershell
cd "E:\AIML Projects\Pharmacovigilance Signal Detection Main"
.venv\Scripts\activate
.venv\Scripts\python.exe -m streamlit run app.py
```

http://localhost:8501

```powershell
ollama --version
ollama list
nvidia-smi
pytest
python -m src.eval.ragas_eval
```

---

## 16. Demo script (manager)

1. Model **openai/gpt-oss-120b (groq)**.  
2. **DEPO-PROVERA × MENINGIOMA** — Strong, *n*=9684, PRR ~4972.24, ROR ~82745.25, serious 324.  
3. Optional Query: amiodarone lung warnings → Cordarone PDF.  
4. Show audit trail + disclaimer.  
5. If 429 → **qwen/qwen3.8-27b**.  
6. Do not hero **qwen2.5:3b** or unpulled **llama3.1:8b**.  
7. Do not ask for full observed/expected 2×2 (long code → truncate / 429).

**Known wrong answers to watch:** mixing **b=619** into **a**; Amiodarone *n*=8 still “Strong” by ≥10 but **fragile**.

---

## 17. Limitations (say them)

- Groq 429 uncaught → Streamlit traceback  
- Judge truncation false fails  
- Small local models break PRR policy  
- Literature = **our PDFs** (often PI), not live PubMed  
- RAGAS set tiny; metrics mismatch tool-calling agents  
- Two FAERS quarters only  
- `search_signal_evidence` can retrieve the wrong pair  

---

## 18. Architecture prompt (technical manager → ChatGPT image)

Use the technical swimlane prompt (offline 2×2 + parquet + 3 FAISS vs online AgentExecutor + stats REPL + hybrid top-3 + guardrails + Groq TPM). Colors: blue deterministic, orange LLM, green guardrails. Never draw the LLM computing PRR.

---

## 19. Interview closer

*Code counts (pandas on signal_df). RAG retrieves unstructured evidence (hybrid top-3). LangChain decides tools. Groq/Ollama only supply tokens. Guardrails check this answer; RAGAS scores the pipeline later. Disproportionality is not causation.*
