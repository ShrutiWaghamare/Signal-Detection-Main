# PV Signal Console — Interview Q&A (this POC)

Use this as the single prep sheet. Answers are about **this repo**, not generic RAG theory.

---

## 1. One-minute pitch

This is a **tool-calling RAG agent** for **FAERS pharmacovigilance signal detection**. An analyst asks about a drug–event pair. A LangChain agent **chooses tools**. **Pandas** looks up **precomputed PRR/ROR** on `signal_df`. **Hybrid retrieval** (BM25 + FAISS + RRF) pulls label/literature chunks. The LLM **writes the narrative**. **Guardrails** check that every number came from a tool and that the text does not claim causation.

**Not** “dump FAERS into the LLM.” **Not** “LLM calculates PRR.”

---

## 2. How RAG is used here

Classic RAG = retrieve text → stuff into prompt → generate.

This POC is **agentic RAG**:

1. User query (drug + event, or free text).
2. Agent may **resolve names** (fuzzy FAERS spelling).
3. Agent **must call** `calculate_pv_statistics` for any number (PRR, ROR, counts).
4. Agent **calls** `search_literature` / `search_signal_evidence` / `search_drug_label` as needed.
5. Retrieved **top-3** snippets go back to the model as **tool observations**, not as a hidden vector dump.
6. Model writes the answer. Guardrails run **after** generation.

Three **separate** corpora (not one mixed index):

| Index | What is in it |
|---|---|
| `data/vector_stores/signals` | High-count FAERS pair summaries |
| `data/vector_stores/literature` | PDF chunks (Depo-Provera, Amiodarone, Warfarin ICH, GVP, FAERS limits) |
| `data/vector_stores/labels` | FDA label PDFs (if present) |

**Why separate indexes?** Structured signal rows and long PDF prose retrieve differently. Merging them hurts both.

---

## 3. How LangChain is used here

| Piece | Role in this POC |
|---|---|
| `create_tool_calling_agent` + `AgentExecutor` | Agent loop: think → tool → observe → answer |
| `return_intermediate_steps=True` | UI **audit trail** (which tool, in, out) |
| `@tool` | `resolve_drug_name`, `resolve_event_name`, retrievers, stats |
| `PythonAstREPLTool` | Executes **pandas** the model writes against `signal_df` |
| `ChatGroq` / `ChatOllama` | Same LangChain chat interface; swap provider, keep tools |
| `ChatPromptTemplate` + scratchpad | System policy: no guessed numbers, no “causes” |
| FAISS + BM25 retrievers | LangChain retriever objects inside tools |

**LangGraph** is in the design notes as a future inspectable graph. The **running POC uses LangChain AgentExecutor**, not a custom LangGraph state machine.

---

## 4. Retrieval, chunking, index (exact settings)

**Retrieval method:** **Hybrid**  
- **Dense:** FAISS + `BAAI/bge-small-en-v1.5` (FastEmbed / ONNX)  
- **Sparse:** BM25 (`rank_bm25`)  
- **Fusion:** Reciprocal Rank Fusion (RRF, k=60)  
- **Then:** keep **top 3** (`DEFAULT_TOP_K_FINAL = 3`)  
- Signals: extra cosine rerank; literature: **filename/event match** so Amiodarone does not return Warfarin  

Pull **10** from FAISS and **10** from BM25, fuse, show **3**.

**Chunking:** `RecursiveCharacterTextSplitter`  
- **800 characters** per chunk, **120 overlap**  
- Split order: paragraph → newline → sentence → space  
- This is **character** size, not LLM tokens  

**Index:** **FAISS** (`faiss-cpu`), local, no data leaves the machine for vectors.  
FAISS has **no token limit**. We only index this corpus. The embedding model (bge-small) typically embeds up to **~512 tokens** per chunk; our 800-char chunks fit that.

**Why you do not see “Top 1/2/3” as a list:** the tool **does** return three `[relevance x.xxx]` snippets to the LLM. The UI **Supporting evidence** card **collapses** them to 1–2 event-related sentences. The **full three** are in the **terminal log** and **audit trail**.

**Stats are not from those 3 chunks.** PRR/ROR come from the **stats tool**.

---

## 5. Why calculations are not given to the LLM

**Interview answer:** PRR/ROR are **regulatory-facing numbers**. Transformers are bad at exact arithmetic on large counts (9,684 cases). A wrong PRR is a **false safety signal**. So:

- The LLM **decides when** stats are needed and may **write pandas**.
- **Python executes** the lookup. The number is **not sampled from weights**.

Tools used for statistics: **`calculate_pv_statistics`** (`PythonAstREPLTool` + `signal_df`). Name tools only **resolve spelling** (`DEPO-PROVERA`), they do not compute PRR.

---

## 6. Why PRR/ROR are computed beforehand

Raw FAERS = DEMO + DRUG + REAC + OUTC, millions of rows. Joining and building a 2×2 **on every query** is slow and heavy.

**Once** at ingestion (`python -m src.ingestion.faers_ingest`):

- Clean + join → parquet  
- One row per (drug, event) with `a,b,c,d`, **prr**, **ror**, serious counts → `signal_summary.parquet`  
- That table is loaded as **`signal_df`**

At query time: `signal_df[(drugname==…) & (pt==…)]` — **lookup**, not a full recount.

**Why the LLM still “uses” those stats:** after the tool prints PRR/ROR, the model **must copy those figures** into the answer and apply **our** strength rule (PRR ≥ 10 strong, ≥ 5 moderate, ≥ 2 weak). Guardrails check the figures appear in **tool output**.

---

## 7. Guardrails (how they relate to Q&A)

**Before generation (prompt):** no estimated numbers; no “causes / proves / confirms”; cite tools; FAERS disclaimer.

**After generation (this is what the UI banner uses):**

| Guardrail | What it asks | Relates to the answer how |
|---|---|---|
| **Regex / numeric faithfulness** | Every number **> 10** in the answer must appear in **tool text** (1% rounding OK) | Stops invented PRR, fake incidence, fake “threshold 500” |
| **Empty evidence** | At least one tool returned usable data | Stops answering when tools found nothing |
| **Traceability %** | Share of countable numbers that matched tools | Shown on the banner (e.g. 100% or 80%) |
| **LLM-as-judge** (`gpt-oss-20b`) | Numbers in context? Causal words? Grounded? | Second Groq call; **separate token bucket** from 120B |
| **UI strength badge** | PRR ≥ 10 / 5 / 2 from **`signal_df` row**, not from model prose | Even if 3B says “Moderate”, the badge can still say STRONG |

**Known POC limit:** the judge only sees the **first ~200 characters** of each tool output, so it can **false-fail** literature answers (17% / 10% were in a later snippet). Regex using the **full** tool text still **passed**. Say this if asked.

**Pydantic `SignalAnswer`** exists (`schemas.py`) with a causal-language validator; the **live path** is regex + LLM judge + UI format, not `with_structured_output` on every call.

---

## 8. Evals used

**A. Online / per query (in the app)**  
- Regex guardrail + traceability  
- LLM-as-judge (pass/fail, issues)  
- Human: audit trail of tools  

**B. Offline RAGAS** (`python -m src.eval.ragas_eval`)  
Metrics: **faithfulness**, **answer relevancy**, **context precision**, **context recall**.  
Judge is Groq `gpt-oss-20b` (no OpenAI). One labeled pair in `src/eval/eval_queries.json`:

- PRR/ROR for DEPO-PROVERA and MENINGIOMA  

Ground truth in the harness is currently **expected drug + event names**, not the numeric PRR. Be honest: RAGAS compares answer vs retrieved **tool contexts**; it does not replace the pandas check.

**C. Unit tests**  
- `tests/test_stats_code_tool.py` — PRR/ROR formula vs hand 2×2  
- `tests/test_retrieval.py` — hybrid / exact-match behaviour  

---

## 9. What is Groq? Why Groq? Why Ollama?

**Groq** = hosted **inference API** (fast open-weight models). You send messages + tool schemas; they return tokens. You **do not upload** Llama or FAERS into Groq. **No GPU needed** on the laptop.

**Why Groq for this POC:** RTX **3050 4 GB** cannot run 120B. We need **strong tool calling**. Groq **free tier**: ~30 RPM, ~1000 RPD, **~8000 TPM**, **~200k TPD — per model ID**. 429 = quota, not a bill.

**Limits are per model**, not one shared “all Groq” pot. 120B full ≠ Qwen 3.8 empty. The **judge uses 20B** so it does not sit in the 120B TPM bucket.

**Ollama** = **local** server (`localhost:11434`). `ollama pull` stores weights **on disk**. Dropdown `qwen2.5:3b (ollama)` never spends Groq tokens (except the judge, which still calls Groq).

**We do not load our local model into Groq.** Two backends, same tools.

---

## 10. Model comparison (what we actually saw)

| Model | Where | Tool calling | Strength rule | Demo? |
|---|---|---|---|---|
| **openai/gpt-oss-120b** | Groq | Best: pandas + literature, parallel-ish tool use | Correct Strong on PRR 4972 | **Primary** |
| **qwen/qwen3.8-27b** | Groq | Works; own TPD bucket | Use if 120B 429 | Backup |
| **openai/gpt-oss-20b** | Groq | Faster, weaker than 120B; used as **judge** | — | Not in dropdown |
| **qwen2.5:3b** | Ollama (already pulled, 1.9 GB) | Can call stats + literature | **Failed:** said Moderate, invented threshold **500** → regex fail 80% | Comparison only |
| **llama3.1:8b** | Ollama | In dropdown | **Not pulled**; 4 GB VRAM is tight | Do not click in demo |

**Which performs better?** For this POC: **Groq 120B >> Groq Qwen 3.8 >> local Qwen 2.5 3B.**  
120B: grounded numbers, correct ≥10 rule, better literature wording.  
3B: proves **local OSS works**, but is **not** the PV brain.

Llama 70B / Qwen 3.6 on Groq **404** on this free org. Listed ≠ enabled.

---

## 11. Expected interview Q&A

### RAG / LangChain

**Q: Is this RAG or an agent?**  
**A:** Both. Retrieval is RAG (hybrid FAISS+BM25). Orchestration is a **LangChain tool-calling agent**. Stats are **not** retrieved from embeddings; they are a **code tool**.

**Q: Why not put the whole FAERS table in the prompt?**  
**A:** Too big for context, costs tokens, LLM cannot count 800k rows. We retrieve **one pair** + **3 chunks**.

**Q: Why hybrid, not FAISS only?**  
**A:** Drug names and MedDRA terms need **exact tokens** (BM25). Prose needs **semantics** (FAISS). RRF merges ranks.

**Q: Why three indexes?**  
**A:** Signal summaries vs PDF prose. Mixing drops quality. Literature post-filter uses **filename**.

**Q: Chunk size 800 — tokens or characters?**  
**A:** **Characters**, RecursiveCharacterTextSplitter, overlap 120.

**Q: What is RRF?**  
**A:** Score = sum 1/(k + rank) across BM25 and FAISS lists. k=60.

**Q: Where is LangGraph?**  
**A:** Optional next step for retries/trace. POC = AgentExecutor.

### Stats / tools

**Q: Does the LLM calculate PRR?**  
**A:** No. Ingestion computed PRR/ROR into `signal_df`. The tool **looks up** the row. LLM authors pandas; **REPL executes** it.

**Q: What is signal_df?**  
**A:** One row per drug–event: `a,b,c,d`, prr, ror, serious_reports. From `signal_summary.parquet`.

**Q: Why parquet?**  
**A:** Fast columnar lookup vs re-parsing `$`-delimited ASCII. SQLite/Postgres would also work; parquet fits local pandas.

**Q: What is PRR / ROR in one line?**  
**A:** Disproportionality in **spontaneous reports**, not incidence, **not causation**.

**Q: Strength rule?**  
**A:** PRR ≥ 10 Strong, ≥ 5 Moderate, ≥ 2 Weak. UI badge uses the **table**, not the model’s adjective.

### Guardrails / eval

**Q: What if the model invents 500 as a threshold?**  
**A:** Regex fails (number not in tools). Happened on Qwen 3B. Traceability 80%.

**Q: What is RAGAS faithfulness?**  
**A:** Claims supported by retrieved **contexts** (here: tool outputs). We also have a **harder** numeric regex check.

**Q: Why two guardrails?**  
**A:** Regex = cheap, exact numbers. Judge = causal language and semantics. Judge can false-fail if tool text is truncated.

### Groq / Ollama / hardware

**Q: Why Groq?**  
**A:** Hosted tool-calling, no local 120B GPU. Laptop is **4 GB VRAM**.

**Q: Do we upload Llama to Groq?**  
**A:** No.

**Q: Token limits?**  
**A:** Per **model**. ~200k TPD, ~8k TPM on free 120B. 429 = wait or switch model.

**Q: Why Ollama?**  
**A:** Run **open-source instruct** models on-prem for comparison (`qwen2.5:3b` already local). Same agent, different `provider`.

### Demo / PV domain

**Q: Can this replace a PV scientist?**  
**A:** No. Disclaimer: qualified review required. Signal ≠ causality.

**Q: Warfarin ICH PRR ~9 with n=4?**  
**A:** Moderate by our cut-off; **small n** → unstable. Formula OK; do not over-claim.

**Q: Literature said “no matching” but PDF exists?**  
**A:** Retriever may return off-section chunks (dyes, pediatrics). Agent should say **not relevant to the event**. Rebuild index after adding PDFs; restart Streamlit (agent is cached).

---

## 12. Demo script (manager)

1. Model: **`openai/gpt-oss-120b (groq)`**.  
2. Signal lookup: **DEPO-PROVERA** × **MENINGIOMA**.  
3. Show: badge Strong, stats table, supporting evidence, AI interpretation, **audit trail**, disclaimer.  
4. Optional: Query *“What lung warnings are described for amiodarone?”* → literature PDF.  
5. If 429: **`qwen/qwen3.8-27b`**.  
6. Do **not** use Llama 8B (not pulled) or 3B as the “correct” answer.  
7. Do **not** ask for “observed and expected counts” (long code → 429).

**One closer:** *“The LLM chooses tools and writes wording. PRR/ROR come from FAERS code. We never dump the database into the model.”*

---

## 13. Honest limitations (say them)

- Groq free **TPD/TPM**; app can **traceback on 429**.  
- Judge truncation → false fail on literature %.  
- Small local models **mis-apply** PRR thresholds.  
- Literature corpus is **our PDFs** (often PI), not live PubMed.  
- Quarters in config: **2026Q1–Q2**.  
- RAGAS set is **small** (3 queries).  
- `search_signal_evidence` can hit a **different pair**; prompt says ignore for PRR.

---

## 14. Commands

```powershell
cd "E:\AIML Projects\Pharmacovigilance Signal Detection Main"
.venv\Scripts\python.exe -m streamlit run app.py
```

```text
python -m src.ingestion.faers_ingest
python -m src.ingestion.build_vector_stores
.\.venv\Scripts\python.exe -c "from src.ingestion.build_vector_stores import build_literature_index; build_literature_index()"
pytest
python -m src.eval.ragas_eval
ollama list
ollama pull llama3.1:8b
```

App: **http://localhost:8501**
