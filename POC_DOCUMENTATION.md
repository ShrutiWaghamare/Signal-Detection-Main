# PV Signal Console — Complete POC Documentation

> **Scope:** This is a Proof-of-Concept for a pharmacovigilance team.  
> It is NOT a production medical system. It does NOT provide diagnosis, treatment recommendations, or regulatory decisions.  
> All outputs require review by a qualified pharmacovigilance professional.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Business Value](#2-business-value)
3. [Dataset](#3-dataset)
4. [Data Flow](#4-data-flow)
5. [Data Cleaning](#5-data-cleaning)
6. [Why Pandas](#6-why-pandas)
7. [Statistical Signal Detection](#7-statistical-signal-detection)
8. [Retrieval](#8-retrieval)
9. [Chunking Strategy](#9-chunking-strategy)
10. [Embeddings](#10-embeddings)
11. [Vector Database](#11-vector-database)
12. [RAG Pipeline](#12-rag-pipeline)
13. [LangChain Components](#13-langchain-components)
14. [Agent Flow](#14-agent-flow)
15. [Tools](#15-tools)
16. [LLM Models](#16-llm-models)
17. [Token Budgets and Rate Limits](#17-token-budgets-and-rate-limits)
18. [Prompting Strategy](#18-prompting-strategy)
19. [Guardrails](#19-guardrails)
20. [Evaluation Framework](#20-evaluation-framework)
21. [Golden Test Set](#21-golden-test-set)
22. [UI](#22-ui)
23. [Audit Trail](#23-audit-trail)
24. [Error Handling](#24-error-handling)
25. [Current Limitations](#25-current-limitations)
26. [Future Enhancements](#26-future-enhancements)
27. [End-to-End Example](#27-end-to-end-example)
28. [Architecture Diagram](#28-architecture-diagram)
29. [Component Table](#29-component-table)
30. [Interview Questions](#30-interview-questions)
31. [Manager Summary (2 minutes)](#31-manager-summary)
32. [Technical Summary (5 minutes)](#32-technical-summary)

---

## 1. Problem Statement

A pharmacovigilance team monitors drug safety by reviewing adverse event reports submitted to regulatory databases. Manually reviewing thousands of reports to identify patterns is slow, error-prone, and difficult to scale. This POC builds an AI-assisted signal detection assistant that automatically retrieves relevant evidence from public FAERS data, computes disproportionality statistics deterministically, retrieves supporting text from drug labels and literature, and generates a structured, grounded explanation using an LLM — with guardrails to prevent hallucination, causal overclaiming, or unsupported conclusions.

**Example:** A PV analyst types `DEPO-PROVERA` + `MENINGIOMA`. The system retrieves FAERS co-occurrence data, computes PRR = 4972.24, finds the label warning about meningioma, and presents a structured analysis — clearly stating this is a reporting signal requiring expert review, not proof of causation.

---

## 2. Business Value

| Stakeholder | Value |
|---|---|
| PV Analyst | Reduces manual search from hours to seconds |
| Medical Affairs | Consistent, grounded signal summaries with source traceability |
| Regulatory team | Auditable tool calls, structured outputs, no hallucinated statistics |
| Manager / POC reviewer | Clear model comparison, guardrail pass/fail, latency metrics |

---

## 3. Dataset

### What is FAERS?

FAERS (FDA Adverse Event Reporting System) is a public database of voluntary adverse event reports submitted to the FDA by patients, healthcare professionals, and manufacturers. It is released quarterly as ASCII text files.

### Files used

Quarters in this POC: **2026Q1 + 2026Q2** (`SUPPORTED_QUARTERS` in `src/config.py`). Counts below are from the **processed parquet** after clean + concat (measured with PyArrow metadata).

| Table | Parquet | Rows | What it is |
|---|---|---|---|
| DEMO | `demo.parquet` | **819,683** | One row per report (`primaryid`). Same number as unique cases in the UI (“of 819,683 reports”). |
| DRUG | `drug.parquet` | **3,330,435** | One row per drug mentioned on a report (~4 drugs per report on average). |
| REAC | `reac.parquet` | **2,725,426** | One row per MedDRA PT on a report. |
| OUTC | `outc.parquet` | **595,285** | One row per coded outcome. Not every report has an OUTC row. |
| Flat join | `faers_flat.parquet` | **39,173,759** | DRUG × REAC on `primaryid` (plus outcome flags). This is why one report becomes many rows. |
| Signal summary | `signal_summary.parquet` | **811,095** | One row per (drugname, pt) pair with **a, b, c, d, PRR, ROR, serious**. 8,357 drugs × 7,770 events (pairs that exist). Pair **a** ranges from 3 to 12,719 (median 6). |

The FAISS **signals** index only embeds pairs with **a ≥ 500** (`MIN_SIGNAL_COOCCURRENCE`) — **1,857** pairs. The stats tool still looks up the full 811k-row summary (e.g. WARFARIN/HAEMORRHAGE has a=16).

**Worked example (already in the UI):** DEPO-PROVERA × MENINGIOMA → a=9,684, serious=324, PRR=4,972.24, ROR=82,745.25, among 819,683 reports.

### Key identifiers

- **`primaryid`** — unique identifier for a single report submission. Used to join all 4 tables.
- **`caseid`** — groups follow-up reports about the same patient case. Multiple `primaryid` can share one `caseid`.

### Relationships

```
DEMO (1 row per report)
  └── primaryid
        ├── DRUG (1+ rows — multiple drugs per report)
        ├── REAC (1+ rows — multiple reactions per report)
        └── OUTC (1+ rows — multiple outcomes per report)
```

### Why one case has multiple drugs, reactions, and outcomes

A patient may be taking multiple drugs simultaneously. They may experience multiple adverse events. The report may record multiple outcomes (e.g. both hospitalisation and recovery). This is real-world complexity — the database reflects polypharmacy.

### Why duplicates exist

FAERS accepts follow-up reports (same case, updated information). The same event may be reported by the patient, the doctor, and the manufacturer separately. This inflates raw counts and is a known FAERS limitation.

### How Q1 and Q2 are combined

The ingestion pipeline loads each quarter separately using `load_raw_tables()`, then concatenates all quarters using `pd.concat(ignore_index=True)`. The flat join is done one quarter at a time using a `pyarrow.ParquetWriter` streaming approach so the exploded DRUG×REAC table (~39 million rows for these two quarters) is never held all at once in RAM.

### Data format in the project

```
Raw TXT ($ delimited)
    ↓  pd.read_csv(sep="$", dtype=str, encoding="latin-1")
DataFrame (in memory)
    ↓  clean, normalise, join
Flat parquet (data/processed/flat_faers.parquet)
    ↓  pre-aggregated per (drug, event) pair
Signal summary parquet (data/processed/signal_summary.parquet)
```

### Why Parquet?

| Reason | Explanation |
|---|---|
| Column-oriented | Reads only the columns needed (drugname, pt, prr) — not the whole row |
| Compressed | Flat join is ~39 million rows (~42 MB parquet); much smaller than the same table as CSV |
| Fast I/O | `pd.read_parquet()` is faster than re-parsing CSVs every startup |
| Schema-stable | Column types are preserved; no re-casting on every load |
| Persistent | Ingestion runs once; the app reads the pre-built parquet at startup |

### Why not send raw TXT (or the 39M-row join) to the LLM?

The raw ASCII plus the exploded join are tens of millions of rows. An LLM context window is ~128K tokens. You cannot paste 819,683 reports or 39 million join rows into a prompt. The model cannot join DEMO/DRUG/REAC/OUTC, count distinct `primaryid`s, or compute a 2×2 reliably from text. It would invent numbers. Cost would be prohibitive. The same query would not reproduce.

---

## 4. Data Flow

```
Raw FAERS ASCII (.txt, $-delimited)
    ↓  faers_ingest.py: _read_ascii_table()
DataFrames (DEMO, DRUG, REAC, OUTC)
    ↓  clean_tables(): .strip().upper(), dropna()
Cleaned DataFrames
    ↓  build_flat_table(): inner join on primaryid
Flat DataFrame (primaryid, drugname, pt, outc_codes, is_serious)
    ↓  written one quarter at a time via pyarrow ParquetWriter
flat_faers.parquet
    ↓  run_ingestion(): compute PRR/ROR aggregations
signal_summary.parquet (one row per drug-event pair)
    ↓  loaded at app startup into signal_df
signal_df (in-memory pandas DataFrame)
    ↓  calculate_pv_statistics tool executes LLM-authored pandas code against this
PRR, ROR, counts returned to agent
```

---

## 5. Data Cleaning

**Implemented in** `src/ingestion/faers_ingest.py :: clean_tables()`

| Stage | What happens | Why |
|---|---|---|
| Column lowercasing | `df.columns = [c.strip().lower() for c in df.columns]` | FDA files inconsistently use upper/lower case headers across quarters |
| Drug name normalisation | `.astype(str).str.strip().str.upper()` | "depo-provera", "DEPO-PROVERA", " Depo Provera " all become "DEPO-PROVERA" |
| Event term normalisation | `.astype(str).str.strip().str.upper()` | "meningioma", "MENINGIOMA" become identical |
| Null removal | `dropna(subset=["primaryid", "drugname"])` | Rows without a drug name or report ID are unusable |
| Outcome code normalisation | `.str.strip().str.upper()` | Consistent comparison against SERIOUS_CODES set |
| DEMO deduplication | `drop_duplicates(subset=["primaryid"])` | Removes duplicate demographic rows for the same report |
| Missing column handling | Fill with `pd.NA` if a column absent in a quarter | Schema stability across quarters that add/drop optional columns |

### What is NOT yet implemented (future enhancements)

| Gap | Current state | Future |
|---|---|---|
| Brand → generic mapping | Not implemented | Map "DEPO-PROVERA" → "medroxyprogesterone acetate" |
| MedDRA hierarchy | Not implemented | Map event terms to MedDRA SOC/HLGT for broader grouping |
| Duplicate case detection | Partial (`drop_duplicates` on primaryid only) | Full caseid-based duplicate suppression |
| Fuzzy drug name matching at ingestion | Not done at ingest | Done at query time via RapidFuzz in resolve_drug_name tool |
| Synonym mapping | Not implemented | Map trade names, generics, abbreviations |

---

## 6. Why Pandas

**Principle:**
```
DATA → PANDAS/PYTHON → STATISTICS → LLM EXPLANATION
```
NOT:
```
DATA → LLM → STATISTICS
```

**Where Pandas is used:**

| Location | Purpose |
|---|---|
| `faers_ingest.py` | Load, clean, join, aggregate all FAERS tables |
| `stats_code_tool.py` | Runtime execution of LLM-authored pandas code against `signal_df` |
| `signal_retriever.py` | Load signal_df for fuzzy drug/event name resolution |
| `app.py` | `lookup_pair()` for fuzzy UI lookups, `get_signal_df()` for display |

**Why we do not give the LLM the complex calculation**

PRR/ROR look like “just arithmetic,” but they are **not** a single number the model can guess. For every pair the pipeline must:

1. Join four tables on `primaryid` (DEMO, DRUG, REAC, OUTC).
2. Count **distinct reports**, not exploded join rows (one case can have many drugs and many events).
3. Build the 2×2: a, b, c, d against **all other drugs and events** in 819,683 reports.
4. Apply `PRR = (a/(a+b)) / (c/(c+d))` and `ROR = (a×d)/(b×c)` with divide-by-zero guards.
5. Count serious as reports whose OUTC code is in `{DE, HO, LT, DS, CA, RI}`.

An LLM is the wrong engine for that:

| If we asked the LLM to calculate… | What goes wrong |
|---|---|
| Hallucination | It invents a plausible PRR (e.g. 12.4) with no row behind it. Guardrails exist because this already happens when models write prose. |
| Scale | Context cannot hold 39 million join rows or 811k pairs. Even 819k report IDs would blow the token budget and Groq ~8k TPM. |
| Distinct counts | Models mix “rows in the join” with “unique `primaryid`.” One report with 8 drugs and 6 PTs is 48 join rows, still **one** case. |
| Formula drift | Off-by-one on b/c/d, using (a+b+c+d) as the denominator, rounding mid-way, mixing ROR and PRR. |
| Non-determinism | Same pair, different day, different number. A PV audit cannot accept that. |
| Chi-square / CI | Easy to get the formula wrong; this POC does not even ask the model to invent those. |
| Tokens | Streaming millions of cells into the prompt is the opposite of the token strategy (tools return a **4-number** table). |

What we do instead: **ingestion computes PRR/ROR once** into `signal_summary.parquet`. At query time the LLM only **authors** a pandas filter (`drugname==…`, `pt==…`). `PythonAstREPLTool` **executes** it. The number that comes back is from the parquet, not from the model’s weights. Regex + judge then check that the prose did not invent extra digits.

---

---

## 7. Statistical Signal Detection

### Implemented statistics

| Statistic | Implemented | Location |
|---|---|---|
| Report count (a_drug_and_event) | ✅ | signal_summary.parquet |
| PRR | ✅ | faers_ingest.py |
| ROR | ✅ | faers_ingest.py |
| Serious reports count | ✅ | signal_summary.parquet |
| 2×2 contingency table | ✅ | compute_2x2_counts() in stats_code_tool.py |
| Confidence intervals | ❌ | Future enhancement |
| p-value | ❌ | Future enhancement |
| EBGM / EBGM05 | ❌ | Future enhancement |

### The 2×2 Contingency Table

```
                    Event E (MENINGIOMA)    All other events
Drug D (DEPO-PROVERA)        a                    b
All other drugs              c                    d
```

| Cell | Meaning |
|---|---|
| **a** | Reports with BOTH Drug D AND Event E (co-occurrences) |
| **b** | Reports with Drug D but NOT Event E |
| **c** | Reports with Event E but NOT Drug D |
| **d** | Reports with neither Drug D nor Event E |

### Formulas

```
PRR = (a / (a + b)) / (c / (c + d))

ROR = (a × d) / (b × c)
```

**PRR interpretation:**
- PRR = 1.0 → no disproportionality
- PRR = 2.0 → event reported twice as often with this drug vs. other drugs
- PRR ≥ 2 → weak signal
- PRR ≥ 5 → moderate-high signal
- PRR ≥ 10 → strong signal (Evans criteria)
- PRR = 4972 (DEPO-PROVERA/MENINGIOMA) → extreme disproportionality

**What PRR/ROR do NOT mean:**
- They do NOT prove the drug causes the event
- They reflect reporting patterns in a voluntary, biased database
- High PRR may result from: reporting bias, channelling, indication confounding, publicity effect, or true pharmacological risk
- Disproportionality analysis is a signal-generation tool, not signal confirmation

### Cases vs serious vs “High” strength (DEPO-PROVERA × MENINGIOMA)

A common UI question: **9,684 cases but only 324 serious (3.3%) — why is signal strength still High?**

These three numbers measure **different things**. High is **not** “most cases are serious.”

| UI field | Value | What it is |
|---|---|---|
| **Cases** | 9,684 | Unique FAERS reports in 2026Q1–2026Q2 that mention **both** DEPO-PROVERA and MENINGIOMA. This is cell **a** in the 2×2. |
| **Serious** | 324 (3.3%) | The **subset of those 9,684** with a FAERS serious-outcome code: death (DE), hospitalisation (HO), life-threatening (LT), disability (DS), congenital anomaly (CA), or required intervention (RI). Coded in `src/ingestion/faers_ingest.py` as `SERIOUS_CODES`. |
| **Signal strength: High** | PRR = 4,972.24 | UI band uses **PRR only** (not the serious %). High = PRR ≥ 10. |

**Why 324 ≪ 9,684 is expected:** FAERS “serious” is a reporter-coded **outcome** (death, hospitalisation, etc.), not “clinically important disease.” Many meningioma reports have a blank OUTC row, “other,” or no DE/HO/LT/DS/CA/RI code. Meningioma can still be clinically important and not coded as serious.

**Why strength is still High:** 324 ÷ 9,684 ≈ 3.3% is **not** the strength rule. The red banner uses Evans-style PRR bands:

| Band | Rule |
|---|---|
| High | PRR ≥ 10 |
| Moderate | PRR 5–10 |
| Low | PRR 2–5 |
| Not elevated | PRR < 2 |

PRR 4,972 means this pair is reported thousands of times more often than expected versus other drugs/events. That is **reporting disproportionality**, not “324 serious cases prove a strong clinical signal.” Screening still also checks **Cases ≥ 3** and **PRR ≥ 2**. A safety team would review seriousness separately; this POC does not mix it into the strength band.

**Short version:** 9,684 = how often the pair was reported. 324 = how many of those had a serious-outcome flag. High = PRR is huge.

---

## 8. Retrieval

**Implemented in:** `src/retrieval/signal_retriever.py`, `src/retrieval/literature_retriever.py`, `src/retrieval/label_retriever.py`, `src/retrieval/reranker.py`

### Architecture: Hybrid Retrieval

```
Query text
    ↓
FAISS dense search (semantic)    +    BM25 sparse search (keyword)
    ↓                                         ↓
Top-10 by cosine similarity           Top-10 by BM25 score
    ↓                                         ↓
         Reciprocal Rank Fusion (RRF, k=60)
                    ↓
         Merged + deduplicated ranked list
                    ↓
       Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
                    ↓
              Top-3 chunks → sent to LLM
```

### Why hybrid retrieval?

| Retrieval type | Good at | Bad at |
|---|---|---|
| FAISS (dense/semantic) | Paraphrased content, semantic similarity | Exact drug name matches, rare terms |
| BM25 (sparse/keyword) | Exact term matching, rare proper nouns | Semantic variation, synonyms |
| Hybrid (both) | Both exact and semantic matching | Nothing significant — this is the right approach |

### Step-by-step for "DEPO-PROVERA + MENINGIOMA"

1. User enters drug + event in UI
2. `resolve_drug_name` fuzzy-matches "DEPO-PROVERA" → exact match in signal_df (score 100)
3. `resolve_event_name` fuzzy-matches "MENINGIOMA" → exact match (score 100)
4. `search_literature` fires: query = "DEPO-PROVERA MENINGIOMA"
5. FAISS returns top-10 PDF chunks by cosine similarity to query embedding
6. BM25 returns top-10 PDF chunks by keyword match
7. RRF fuses both lists: `score(doc) = sum(1/(k + rank_in_list))`
8. Cross-encoder reranks merged list by semantic relevance
9. Top-3 chunks returned with relevance scores
10. UI `_summarize_evidence_items()` deduplicates and caps at 2 displayed items
11. All 3 chunks passed to LLM for reasoning

### Exact drug/event lookup (structured path)

For PRR/ROR, there is NO vector retrieval. The agent writes pandas code:
```python
signal_df[(signal_df['drugname']=='DEPO-PROVERA') & (signal_df['pt']=='MENINGIOMA')]
  [['prr','ror','a_drug_and_event','serious_reports']]
```
This is an exact O(n) lookup on a pre-built in-memory DataFrame. No embeddings. No approximation. Exact answer.

---

## 9. Chunking Strategy

**Implemented in:** `src/ingestion/pdf_ingest.py`, `src/ingestion/build_vector_stores.py`

### What is chunked?

Only **PDF documents** — drug labels (prescribing information) and clinical literature. FAERS tabular data is NOT chunked — it is stored as a pre-aggregated parquet and queried directly with pandas.

### Parameters

| Parameter | Value | Why |
|---|---|---|
| Chunk size | 800 characters | Fits within context budget while preserving a full label section |
| Overlap | 120 characters | Prevents important sentences being split across chunk boundaries |
| Splitter | `RecursiveCharacterTextSplitter` | Tries to split at paragraph → newline → sentence → space, preserving semantic units |

### Why NOT chunk FAERS tabular data?

FAERS data is structured (each row = one drug-event association). It is better handled with:
- Exact pandas lookup for known drug-event pairs
- Pre-aggregated signal summaries for the FAISS signal index

Chunking tabular rows into text fragments would lose structure and introduce retrieval noise.

### Chunk size trade-offs

| Too large | Too small |
|---|---|
| Exceeds LLM context budget | Splits sentences mid-thought |
| Returns redundant content | Loses surrounding clinical context |
| Higher token cost per retrieval | Lower relevance per chunk |

---

## 10. Embeddings

**Model:** `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX Runtime)

**Implemented in:** `src/retrieval/signal_retriever.py :: get_embeddings()`

| Property | Detail |
|---|---|
| Framework | FastEmbedEmbeddings (ONNX Runtime backend) |
| Model size | ~50MB, downloaded once, cached in `~/.cache/fastembed/` |
| Dimensionality | 384 |
| Why this model | No PyTorch required, no GPU needed, no DLL issues on Windows, strong retrieval performance for English biomedical text |
| Where used | Building all 3 FAISS indexes + RAGAS evaluation embeddings |
| Weights location | Local — weights are NOT sent to Groq or any external API |

### Why we do not embed the FAERS dataset (no vector index for PRR)

**Question:** Why don’t we embed our FAERS tables? Why not use an embedding / FAISS index for that data?

**Answer:** Embeddings find related *wording*. FAERS needs the *right pair and the right count*. Those are different jobs.

An embedding index (FAISS) answers: “which **text chunks** are closest in meaning to this question?” That is correct for a PDF warning (“Discontinue if meningioma is diagnosed”). A named FAERS pair needs the **exact row** `(DEPO-PROVERA, MENINGIOMA)` and a **formula** on a, b, c, d. That is an exact parquet lookup, not “nearest paragraph.”

| If we embedded FAERS / put PRR on a vector index… | What goes wrong |
|---|---|
| Scale | ~39 million exploded join rows, or 811k pair rows. Embedding all of that is slow, large on disk, and almost none of it is needed for one query. |
| Wrong match | FAISS returns *similar* text. You could get another progestogen or another CNS tumour and quote **its** PRR. |
| Numbers are not meaning | `4972.24` is not a sentence. Cosine similarity does not compute `(a/(a+b)) / (c/(c+d))`. The model would invent or copy a neighbour’s number. |
| Join is on `primaryid` | Drug and event live on **different rows**. Chunking DEMO/DRUG/REAC separately breaks the key that makes the 2×2 correct. |
| Tokens | Pulling 10 “similar” FAERS chunks every query burns Groq TPM. We already only keep **top-3 PDF** snippets. |
| We already have an index | `(drugname, pt)` in `signal_summary.parquet` **is** the index. Pandas filter is exact and cheap. |

**What we *do* embed:** PDF chunks only (prescribing information / literature) → FAISS + BM25 → top-3. There is also a small FAISS **signals** text card for pairs with **a ≥ 500** (1,857 docs) so the agent can browse high-count pairs. The prompt still says: **do not take PRR from those hits**; call `calculate_pv_statistics`.

---

## 11. Vector Database

**Technology:** FAISS (Facebook AI Similarity Search) — local, in-process

**Three indexes:**

| Index | Content | Built from |
|---|---|---|
| `signals_index` | FAERS signal summary text per drug-event pair | Pre-aggregated signal_summary parquet |
| `literature_index` | Clinical literature PDF chunks | PDF files in `data/external_docs/literature/` |
| `labels_index` | Drug prescribing information PDF chunks | PDF files in `data/external_docs/labels/` |

**Why FAISS over cloud vector DBs (Pinecone, Weaviate)?**

- POC runs fully local — no cloud account needed
- No data leaves the machine
- Free — no API costs
- Sufficient for POC-scale corpus (~hundreds of documents)
- Pinecone is an optional future upgrade (config.py has PINECONE_API_KEY slot)

---

## 12. RAG Pipeline

**What is RAG?**
Retrieval-Augmented Generation — instead of relying on the LLM's training memory, the system retrieves relevant evidence from a knowledge base and includes it in the prompt. The LLM then generates an answer grounded in that evidence rather than from memory.

**Why RAG here?**
- Drug-event statistics change with each FAERS quarter — LLM training data is stale
- PRR/ROR numbers are specific to this dataset — not in any LLM's training
- Prescribing information warnings are specific and must be cited exactly
- RAG makes the system grounded, traceable, and updatable

**Complete RAG flow:**

```
User query: "DEPO-PROVERA / MENINGIOMA"
         ↓
Agent decides to call tools in parallel:
    ├── calculate_pv_statistics → pandas lookup → PRR=4972.24, ROR=82745.25
    └── search_literature → hybrid retrieval → top-3 label chunks
         ↓
Both results available in agent scratchpad
         ↓
LLM constructs answer using ONLY the retrieved numbers and text
         ↓
Guardrails verify numbers trace to tool output
         ↓
UI displays structured answer + guardrail verdict
```

**Important:** the pandas stats step in the diagram is **not RAG**. Only `search_literature` / `search_signal_evidence` / `search_drug_label` are RAG.

### Where RAG is used (and where it is not)

| Used? | What | Files |
|---|---|---|
| ✅ RAG | Ingest PDF prescribing information and literature: load, chunk 800/120, embed, save FAISS | `src/ingestion/pdf_ingest.py`, `src/ingestion/build_vector_stores.py` |
| ✅ RAG | Hybrid retrieve at query time: FAISS 10 + BM25 10 → RRF → cross-encoder rerank → top-3 | `src/retrieval/literature_retriever.py`, `label_retriever.py`, `signal_retriever.py`, `reranker.py` |
| ✅ RAG | Those retrievers are wrapped as agent tools; snippets go into the LLM context | `src/agent/build_agent.py` (`_make_retriever_tool`) |
| ✅ RAG | UI cards show the same retrieved passages (not the model’s memory) | `mainapp.py` evidence helpers |
| ✅ RAG eval | RAGAS faithfulness / relevancy / context_recall on retrieved contexts | `src/eval/ragas_eval.py` |
| ❌ Not RAG | FAERS PRR / ROR / case counts | `src/tools/stats_code_tool.py` + `signal_summary.parquet` (pandas) |
| ❌ Not RAG | Drug / event name resolve | `src/tools/signal_lookup_tool.py` (RapidFuzz) |
| ❌ Not RAG | Regex + LLM judge | `src/guardrails/` (post-generation checks) |

**Why this split:** you cannot RAG a 2×2. PRR is a formula on counts. Embedding FAERS rows and hoping the model “finds” 4972.24 is how numbers get invented. RAG is only for **prose in PDFs**.

### Why we did not use RAG on the FAERS dataset

RAG (chunk → embed → FAISS → “nearest text”) is for **unstructured prose**. FAERS is **relational tables**.

| If we RAG’d FAERS… | What happens |
|---|---|
| Size | 39 million flat rows, or even 811k pair summaries, cannot all sit in a prompt. Embedding every join row is slow, huge on disk, and most rows are useless for one query. |
| Wrong object | A FAISS hit is “text that looks similar,” not “the exact (DEPO-PROVERA, MENINGIOMA) 2×2.” Similar pairs (other progestogens, other CNS tumours) would pollute the answer. |
| Numbers are not semantics | `4972.24041` is not a sentence. Cosine similarity does not compute a/(a+b). The model would quote a neighbour’s PRR or invent one. |
| Join semantics lost | Drug and event sit on **different source rows**, tied only by `primaryid`. Chunking DEMO/DRUG/REAC separately breaks that key. |
| Token cost | Retrieving 10+10 FAERS “chunks” every query wastes the Groq budget we already cap (top-3 PDFs only). |
| We already have an index key | `(drugname, pt)` is an exact lookup. Pandas / parquet is O(filter), not “maybe the right paragraph.” |

What we **do** embed from FAERS: only a **small text card** per high-count pair (a ≥ 500 → 1,857 docs) for `search_signal_evidence`. That tool is **not** allowed to supply PRR for a named pair — the prompt says call `calculate_pv_statistics` instead. PDFs stay on the RAG path because warnings are sentences (“Discontinue if meningioma is diagnosed”).

---

---

## 13. LangChain Components

**LangChain version:** 0.3.x (not 1.x — upgrade blocked by AgentExecutor removal)

| Component | Used | Purpose |
|---|---|---|
| `ChatGroq` | ✅ | Main LLM chat model (Groq API) |
| `ChatOpenAI` | ✅ | Mistral + vLLM endpoint (OpenAI-compatible) |
| `ChatOllama` | ✅ | Local Ollama models |
| `AgentExecutor` | ✅ | Orchestrates tool-calling agent loop |
| `create_tool_calling_agent` | ✅ | Builds agent with structured tool schema |
| `ChatPromptTemplate` | ✅ | System prompt + scratchpad template |
| `MessagesPlaceholder` | ✅ | Injects tool call history into prompt |
| `@tool` decorator | ✅ | Wraps Python functions as LangChain tools |
| `PythonAstREPLTool` | ✅ | Executes LLM-authored pandas code safely |
| `FAISS` (vectorstore) | ✅ | Vector similarity search |
| `BM25Retriever` | ✅ | Sparse keyword retrieval |
| `BaseRetriever` | ✅ | Abstract base for custom HybridRetriever |
| `FastEmbedEmbeddings` | ✅ | Local embedding model |
| LangGraph | ❌ | Not used — future migration target |
| Output parser | ❌ | Not used — model follows prompt format instructions |
| LangSmith | Optional | Tracing (LANGCHAIN_TRACING_V2 in .env) |

**Why LangChain instead of raw API calls?**
- Tool schema validation is handled automatically
- `return_intermediate_steps=True` gives full tool call audit trail for free
- Model swapping (Groq → Mistral → Ollama) requires changing one argument
- `AgentExecutor` handles the tool-call → observe → continue loop

### What `create_tool_calling_agent` is

LangChain helper in `src/agent/build_agent.py`:

```python
agent = create_tool_calling_agent(llm, tools, AGENT_PROMPT)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)
```

It is **native JSON function-calling**, not ReAct text (`Thought:` / `Action:`).

| Piece | Role |
|---|---|
| `llm` | Chat model that already supports tool calls (`ChatGroq` / `ChatOpenAI`) |
| `tools` | Python functions with a JSON schema (`@tool`, `PythonAstREPLTool`) |
| `AGENT_PROMPT` | System prompt + `{input}` + `agent_scratchpad` |
| `AgentExecutor` | Loop: LLM may emit `tool_calls` → Python runs the tools → results return as tool messages → LLM continues until it emits a **final message with no tool calls** |

**Why not ReAct?** Groq and Mistral models already emit structured `tool_calls`. Parsing free-text “Action: search_literature” is more brittle.

**Why not LangGraph?** This POC only needs “call tools until done.” LangGraph is a future rewrite (LangChain 1.x), not the live loop.

**Signal lookup vs Query:** the UI can tell the agent to call stats + literature in **one** step (2 tools). Query mode lets the model loop — that is why `search_literature` can fire 3 times on a vague question.

### Where LangChain is used in this project

| Area | LangChain pieces | File(s) |
|---|---|---|
| Chat models | `ChatGroq`, `ChatOpenAI`, `ChatOllama` | `src/agent/llm_provider.py` |
| Agent loop | `create_tool_calling_agent`, `AgentExecutor` | `src/agent/build_agent.py` |
| Prompt | `ChatPromptTemplate`, `MessagesPlaceholder` | `src/agent/prompts.py` |
| Tools | `@tool`, `PythonAstREPLTool` | `src/tools/stats_code_tool.py`, `signal_lookup_tool.py` |
| PDF ingest | `PyPDFLoader`, `RecursiveCharacterTextSplitter` | `src/ingestion/pdf_ingest.py` |
| Indexes | `FAISS.from_documents` / `FAISS.load_local` | `src/ingestion/build_vector_stores.py` |
| Embeddings | `FastEmbedEmbeddings` | `src/retrieval/signal_retriever.py` |
| Hybrid RAG | `FAISS`, `BM25Retriever`, custom `HybridRetriever(BaseRetriever)` | `src/retrieval/*` |
| Judge | `ChatGroq` + `SystemMessage` / `HumanMessage` | `src/guardrails/llm_judge.py` |
| RAGAS | `ChatGroq` as the eval LLM | `src/eval/ragas_eval.py` |
| Not used live | LangGraph (listed in requirements only) | — |

---

## 14. Agent Flow

**Architecture:** LangChain `create_tool_calling_agent` + `AgentExecutor`

```
User query
    ↓
AgentExecutor.invoke({"input": query})
    ↓
Agent (LLM) decides which tools to call
    ↓  (parallel where possible — one LLM call for 2 tool calls)
Tool execution (Python runtime)
    ↓
Results injected into agent scratchpad
    ↓
Agent (LLM) synthesises final answer
    ↓
AgentExecutor returns {"output": answer, "intermediate_steps": [...]}
```

**For DEPO-PROVERA / MENINGIOMA (signal-lookup mode):**

| Step | Tool | Input | Output |
|---|---|---|---|
| 1 (skipped) | resolve_drug_name | Skipped — UI pre-resolves | — |
| 2 (parallel) | calculate_pv_statistics | pandas code for DEPO-PROVERA/MENINGIOMA | PRR=4972.24, ROR=82745.25, a=9684, serious=324 |
| 2 (parallel) | search_literature | "DEPO-PROVERA MENINGIOMA" | Top-3 label chunks (meningioma warning) |
| 3 | LLM synthesis | Both tool results | Structured Format A answer |

**Why parallel tool calls?**
Statistics and literature are independent — one does not depend on the other. Calling both in one LLM step halves the number of API round-trips and cuts latency in half (~3s vs ~6s).

---

## 15. Tools

| Tool | File | What it does | Source of truth |
|---|---|---|---|
| `resolve_drug_name` | `signal_lookup_tool.py` | RapidFuzz fuzzy match against signal_df drugname vocabulary | signal_df |
| `resolve_event_name` | `signal_lookup_tool.py` | RapidFuzz fuzzy match against signal_df pt vocabulary | signal_df |
| `calculate_pv_statistics` | `stats_code_tool.py` | LLM authors pandas code → PythonAstREPLTool executes it | signal_df (parquet) |
| `search_literature` | `literature_retriever.py` | Hybrid FAISS+BM25 over literature PDFs → reranked top-3 | Literature FAISS index |
| `search_signal_evidence` | `signal_retriever.py` | Hybrid FAISS+BM25 over FAERS signal text summaries | Signals FAISS index |

**Critical design: `calculate_pv_statistics`**

The LLM does NOT calculate PRR/ROR. It writes the pandas filter logic. Python executes it. This means:
- Numbers are always from real data
- LLM cannot hallucinate a statistic
- The `_ensure_clean_output()` wrapper strips DataFrame row indices to prevent models misreading the FAERS record ID as a data value

---

## 16. LLM Models

### Main agent models (selectable in UI)

| Model | Provider API | Type | Role | Context Window | Max Output (set) |
|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | Groq | Proprietary (served via Groq) | Main agent (default) | 131,072 tokens | 1,024 tokens |
| `qwen/qwen3.8-27b` | Groq | Open-weight (Qwen) | Main agent | 131,072 tokens | 1,024 tokens |
| `openai/gpt-oss-20b` | Groq | Proprietary (served via Groq) | Main agent | 131,072 tokens | 1,024 tokens |
| `ministral-8b-latest` | Mistral AI | Proprietary (Mistral) | Main agent | 128,000 tokens | 1,024 tokens |
| `ministral-14b-latest` | Mistral AI | Proprietary (Mistral) | Main agent | 128,000 tokens | 1,024 tokens |

### Judge model (fixed, not selectable)

| Model | Provider API | Role | Temp | Max Output |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | Groq | LLM Judge (post-generation safety check) | 0 | 1,024 tokens |

The UI dropdown (mainapp) is only three answer models: `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `ministral-14b-latest`. The judge is **always** Groq `gpt-oss-120b` and is not in that dropdown.

### Why one model answers and another judges (tokens)

Yes — **when the answer model is not Groq 120b**, this split avoids wasting Groq tokens.

| Role | Who | What it spends |
|---|---|---|
| Answer LLM | User pick: 120b, Qwen, or Mistral 14B | Tool-calling + the long Format A/B write-up (the expensive call) |
| Judge | Always Groq `gpt-oss-120b`, temp=0 | One short JSON: pass/fail, causal language, grounding |
| Regex | No LLM | Free numeric traceability |

**Why a fixed judge:** every demo is scored by the **same** model, so “Verified” means the same thing whether Qwen or Mistral wrote the answer.

**When tokens are actually saved**

| Answer model | Groq TPM used | Effect |
|---|---|---|
| `ministral-14b-latest` | Judge only (small JSON) | Best split — long answer is on Mistral’s quota |
| `qwen/qwen3.8-27b` | Qwen + judge (two Groq models) | Answer is cheaper/faster than 120b; still Groq |
| `openai/gpt-oss-120b` | Agent **and** judge share the same ~8k TPM bucket | **Not** a save — this is the 429 you see after a 120b run (`7049 + 1475 > 8000`) |

The judge is kept small on purpose (JSON only, literature truncated). Regex already checks numbers, so the judge is not asked to redo the 2×2. For demos that must stay under Groq free TPM, use **Mistral 14B** as the answer model and leave the judge on 120b.

### Important concept: Model ≠ Provider

- **Model** = the neural network weights (e.g. GPT-OSS-120B)
- **Inference Provider** = the hardware/API serving the model (e.g. Groq)
- **Groq's advantage** = extremely fast inference on custom LPU hardware — same model, ~10× faster than standard GPU inference
- Weights are NOT uploaded to Groq — Groq licences and hosts the weights

### Context window vs. rate limits

```
Context window = how much text the model CAN process per call (131,072 tokens)
Max output     = how much text the model CAN write per call (our cap: 1,024)
RPM            = how many API calls per minute (Groq free: ~30)
TPM            = tokens processed per minute (Groq free: ~8,000)
RPD            = requests per day (Groq free: ~200,000 per model)
```

These are four completely independent limits. A single call can hit TPM without exceeding RPM.

---

## 17. Token Budgets and Rate Limits

### DEV_MODE token budget

Controlled by `DEV_MODE` in `.env`:

| Mode | max_tokens | When to use |
|---|---|---|
| `DEV_MODE=true` | 300 | Testing pipeline / debugging tool calls |
| `DEV_MODE=false` | 1,024 | Real use / demos / evaluation |
| RAGAS eval | 2,048 | Always hardcoded for evaluation scoring |

### The 429 Rate Limit Error

**Why it happens:**
```
TPM limit  = 8,000  tokens/minute
Used so far = 7,049  tokens (earlier calls this minute)
New request = 1,737  tokens
Total = 8,786 > 8,000 → REJECTED with HTTP 429
```

**Why model comparison scripts hit it faster:**

Each model evaluation involves:
- Agent call (prompt + tool outputs + answer) ≈ 1,500–3,000 tokens
- LLM Judge call ≈ 500–1,000 tokens
- Total per model ≈ 2,000–4,000 tokens
- 5 models × 4,000 = up to 20,000 tokens → guaranteed 429 within 1 minute

**Mitigation strategies implemented:**
- `SLEEP_S = 6` between model evaluations
- `DEV_MODE=true` reduces output tokens to 300 during testing
- Rate-limit errors are surfaced in the evaluation output (not hidden)
- `max_retries=1` on agent, `max_retries=0` on judge (prevents retry storms)

---

## 18. Prompting Strategy

**Implemented in:** `src/agent/prompts.py`

### System prompt structure

1. **Role definition** — "pharmacovigilance research assistant... never answer from memory alone"
2. **Tool-use policy** (6 rules) — when to resolve, when to calculate, when to call literature, how many times to retry
3. **Language policy** — banned words (causes/proves/confirms), required language (reporting signal, disproportionality)
4. **Format A** — for FAERS stats queries: Statistics → Literature → Bullets (a/b/c)
5. **Format B** — for label/literature queries: 2–4 plain English paragraphs, no headers

### Why two formats?

Format A is designed for signal detection queries (PRR/ROR). Format B is designed for label questions ("what does the label say about X?"). The previous single template caused models to try filling empty stats sections for label-only questions, producing garbled output.

### Prompt injection defence

Retrieved PDF chunks are passed as tool observation strings (not as user messages). The system prompt instructs the agent to use chunks as evidence only. This reduces (but does not eliminate) injection risk.

---

## 19. Guardrails

**Implemented in:** `src/guardrails/validators.py` + `src/guardrails/llm_judge.py`

### Layer 1: Regex Guardrail (instant, free)

**Check 1 — Numeric faithfulness:**
- Extract all numbers >10 from the answer
- Verify each traces to tool output within 1% tolerance
- Handles rounding (4972.24041 → 4972.24 is OK)
- Handles space-thousands separators ("82 745.25" → "82745.25")

**Check 2 — Empty evidence acknowledgement:**
- If ALL tools returned empty/no-match, the answer must say "no data" or "not found"
- Catches models that answer from memory when tools fail

**Check 3 — Drug name identity:**
- Expected drug name must appear in the answer
- Catches encoding bugs (DEPO?PROVERA) and answers about wrong drugs

### Layer 2: LLM Judge (one extra Groq call, temp=0)

**Scores three dimensions:**

| Dimension | What it checks |
|---|---|
| `numeric_faithfulness` | Do numbers in the answer match tool output? |
| `causal_language_used` | Does the answer say "causes", "proves", "confirms"? |
| `evidence_grounded` | Are literature claims backed by retrieved text? |

**Returns:** `overall_verdict: pass | fail` + `issues` list + `reasoning` string

**Judge sees:**
- Stats tool output: first 300 chars, labelled "row index = FAERS record ID, not a data value"
- Literature tool output: up to 800 chars (increased from 200 to prevent truncating relevant snippets at lower relevance ranks)

### Guardrails NOT yet implemented (future)

| Guardrail | Status |
|---|---|
| Medical advice detection | Partial (language policy in prompt) |
| PII detection/masking | Not implemented |
| Prompt injection detection | Partial (tool-as-observation pattern) |
| Output schema validation | Not implemented (structured JSON output) |
| Scope filter (off-topic) | Not implemented |

---

## 20. Evaluation Framework

**Implemented in:** `src/eval/ragas_eval.py`

### What is evaluated

| Metric | What it measures | Ground truth needed? |
|---|---|---|
| `faithfulness` | Did the model only use facts from tool output? | No |
| `answer_relevancy` | Did it answer what was actually asked? | No |
| `context_recall` | Did it include all key facts from the expected answer? | **Yes** |

### Guardrail evaluation (per query, per model)

| Check | Free? | What it catches |
|---|---|---|
| Key numbers present | Free (string search) | PRR/ROR/counts missing or wrong |
| Regex guardrail | Free | Untraced numbers, drug mismatch |
| LLM Judge | 1 API call | Causal language, ungrounded claims |

### RAGAS vs. guardrails

| | RAGAS | Guardrails |
|---|---|---|
| When runs | Manually, offline | Every query, real-time |
| Purpose | Benchmark model quality | Protect analyst from bad answers |
| Requires API? | Yes (judge LLM + embeddings) | Judge only |
| Output | Score 0–1 | Pass/Fail |

---

## 21. Golden Test Set

**File:** `src/eval/eval_queries.json`

**Why hand-written, not LLM-generated:**
The correct PRR for DEPO-PROVERA/MENINGIOMA is 4972.24 — a mathematical fact from real data. No LLM can reliably generate this. Using an LLM to write ground truth would be circular validation (one model verifying another of the same family). Ground truth comes from the data, not from the evaluator.

**Current test cases (3):**

| ID | Query type | Drug | Event | Ground truth contains |
|---|---|---|---|---|
| pv_001 | FAERS stats | DEPO-PROVERA | MENINGIOMA | PRR=4972.24, ROR=82745.25, a=9684, serious=324 |
| pv_002 | Label/literature | DEPO-PROVERA | BONE DENSITY | 2-year limit, may not fully reverse |
| pv_003 | Label/literature | DEPO-PROVERA | CONTRAINDICATION | Thrombophlebitis, breast malignancy, hypersensitivity |

**Recommended additional test cases (future):**

| # | Scenario | Expected behavior |
|---|---|---|
| 4 | Drug not in FAERS | "No data found for this drug-event pair" |
| 5 | Event not in FAERS | "No data found" |
| 6 | Drug synonym ("medroxyprogesterone") | Fuzzy match → DEPO-PROVERA |
| 7 | Case variation ("depo provera") | Normalise → DEPO-PROVERA |
| 8 | Ambiguous drug name | Ask for clarification or list matches |
| 9 | Out-of-scope question | "This system only covers PV signal detection" |
| 10 | Prompt injection attempt | Ignore injected instruction, answer normally |
| 11 | Missing data scenario | Acknowledge insufficient evidence |
| 12 | No causal claim | Verify answer uses "reporting signal" not "causes" |

---

## 22. UI

**Technology:** Streamlit  
**Layout:** 3 columns

### Left column — Query panel

- Mode toggle: "Signal lookup (drug + event)" / "Query" (free text)
- Drug name input + adverse event input
- **Run analysis** button
- Model selector dropdown (all 5 models)
- Data quarters label
- Session history (last 8 queries, clickable)

### Centre column — Results

| Section | What shows | Source |
|---|---|---|
| Result title | Drug × Event, match scores | UI fuzzy lookup |
| Signal badge | STRONG/MODERATE/WEAK/NONE + Evans criteria | signal_df PRR value |
| Signal strength bar | Visual progress bar | PRR value |
| Statistical evidence table | Co-reports, PRR, ROR, serious%, total reports | signal_df (direct, not model prose) |
| Supporting evidence | 1–2 deduped PDF chunks with relevance scores | search_literature tool |
| AI Interpretation | Format A or B structured answer | LLM output |
| Limitations box | Standard FAERS limitations | Static (always shown) |
| Recommended review box | Escalation guidance | Static (always shown) |
| Guardrail banner | Green ✅ Verified / Red ❌ Review needed + traceability % | validators.py + llm_judge.py |

### Right column — Audit trail + Evaluation

**Audit trail:**
- One card per tool call: tool name, input, output (first 220 chars)
- Icons: 💊 resolve, 📊 stats, 📚 literature, 🔍 signal evidence

**Evaluation panel:**
- Overall verdict (Verified / Review needed) with colour
- Tool calls count, elapsed time, traceability %, evidence found
- Expandable audit details: LLM Judge sub-checks (3 booleans), Regex result, specific issues

---

## 23. Audit Trail

**What is recorded per query** (in `run_and_collect()`, logged to terminal):

| Field | Logged | Notes |
|---|---|---|
| Timestamp | ✅ | Python logging timestamp |
| Model name | ✅ | Shown in UI top bar |
| Query | ✅ | Full query string |
| Tool calls (name + input) | ✅ | `intermediate_steps` |
| Tool outputs | ✅ | First 200 chars per tool |
| Final answer | ✅ | Full text |
| Elapsed time | ✅ | Shown in evaluation panel |
| Traceability % | ✅ | Computed from number matching |
| Regex guardrail result | ✅ | Pass/fail + issues |
| LLM Judge result | ✅ | Verdict + reasoning + issues |

**Not yet implemented (future):**
- Session/user ID
- Persistent log file
- Token count per call
- Dataset version tag
- Formal audit log format (e.g. NDJSON)

**Why auditability matters in pharma:**
Any AI-assisted regulatory decision must be explainable and reproducible. If a signal leads to a label change, the PV team must be able to show exactly what data was used, which model produced the answer, what guardrails passed, and what a qualified reviewer decided. Without an audit trail, the system cannot be used in any semi-regulated context.

> **Note:** This POC audit trail is informational only. It does NOT constitute a regulatory-grade audit log.

---

## 24. Error Handling

| Error | Handling |
|---|---|
| HTTP 429 (rate limit) | Logged with full message, surfaced in evaluation output, retried with sleep |
| Tool execution error | Returned as error string to agent, agent tries to recover |
| Empty tool result | `check_empty_tool_response()` guardrail flags if model answers anyway |
| Drug/event not found | `resolve_drug_name` returns "no match" message |
| Missing API key | RuntimeError with signup URL at build time |
| FAISS index missing | FileNotFoundError with instructions to run ingestion |
| Unicode encoding (Windows terminal) | `.encode("ascii", errors="replace").decode("ascii")` in eval scripts |

---

## 25. Current Limitations

| Limitation | Impact |
|---|---|
| Groq free tier: ~8K TPM | Model comparison hits 429 with 5 models in quick succession |
| No MedDRA hierarchy | Cannot aggregate signals across related event terms |
| No duplicate case detection | Over-counting in some drug-event pairs |
| No confidence intervals | Cannot assess statistical significance of PRR |
| `openai/gpt-oss-20b` encoding bug | Writes DEPO?PROVERA — fails regex guardrail consistently |
| `ministral-8b` causal language | Uses "causes/confirms" — fails LLM Judge |
| LangChain 0.3.x (not 1.x) | Cannot use Gemini 3.x or migrate to LangGraph without refactor |
| FAISS is local only | Cannot be shared across multiple users |
| No user authentication | Any user can run any query |
| Evaluation only 3 golden queries | Not enough for statistically meaningful RAGAS scores |

---

## 26. Future Enhancements

| Enhancement | Why deferred |
|---|---|
| MedDRA hierarchy integration | Requires licensed MedDRA database |
| Confidence intervals on PRR/ROR | POC scope — adds complexity |
| Temporal trend analysis | Requires multi-year FAERS data pipeline |
| Bayesian signal detection (EBGM) | Complex — exceeds POC scope |
| Migrate to LangGraph | Requires full agent rewrite from AgentExecutor |
| Gemini 3.x support | Needs LangGraph migration first |
| Production audit logging (NDJSON) | Not needed for POC review |
| RBAC and user auth | Infrastructure concern, not POC scope |
| Pinecone cloud vector DB | Local FAISS sufficient for POC scale |
| Human-in-the-loop review workflow | Requires web framework beyond Streamlit |
| Model drift detection | Production concern |
| Enterprise deployment (Docker/K8s) | Not needed for POC |

---

## 27. End-to-End Example

**Query:** DEPO-PROVERA / MENINGIOMA

```
1. User enters "DEPO-PROVERA" + "MENINGIOMA" in UI → clicks Run analysis

2. app.py builds optimised query:
   "Analyze FAERS pair DEPO-PROVERA / MENINGIOMA. Spellings are exact —
    skip resolve tools. In one step call both: calculate_pv_statistics
    and search_literature..."

3. AgentExecutor runs agent (openai/gpt-oss-120b on Groq)
   → LLM decides: call both tools in parallel

4. PARALLEL TOOL CALLS:
   Tool 1: calculate_pv_statistics
     Input: pandas code filtering signal_df for DEPO-PROVERA + MENINGIOMA
     Execute: _ensure_clean_output() wraps bare expr with reset_index().to_dict()
     Output: [{'prr': 4972.24041, 'ror': 82745.248693,
               'a_drug_and_event': 9684, 'serious_reports': 324}]

   Tool 2: search_literature
     Input: "DEPO-PROVERA MENINGIOMA"
     FAISS: top-10 chunks by cosine similarity
     BM25:  top-10 chunks by keyword match
     RRF:   fuse → ranked merged list
     Reranker: cross-encoder scores → top-3 kept
     Output: [relevance 1.000] HIGHLIGHTS OF PRESCRIBING INFORMATION...
             [relevance 0.960] WARNING: LOSS OF BONE MINERAL DENSITY...
             [relevance 0.920] Meningioma: Discontinue Depo-Provera CI...

5. LLM synthesises Format A answer:
   "Here is the FAERS analysis for DEPO-PROVERA / MENINGIOMA.
    Statistics: PRR 4972.24, ROR 82745.25, co-reports 9684, serious 324
    Literature: Prescribing information states discontinue if meningioma diagnosed.
    Bullets: (a) Strong — PRR far exceeds ≥10... (b) Limitation... (c) Next step..."

6. Guardrail Layer:
   Regex check:
     Extract numbers from answer: {4972, 82745, 9684, 324}
     All found in tool output → PASS
     "DEPO-PROVERA" in answer → PASS
     Tool output non-empty → PASS

   LLM Judge (separate Groq call):
     Checks numeric faithfulness → TRUE
     Checks causal language → FALSE (no "causes")
     Checks evidence grounded → TRUE
     Verdict: PASS

7. UI displays:
   Signal badge: 🔴 STRONG SIGNAL (PRR ≥ 10 — Marked disproportionality)
   Stats table: 9,684 co-reports | PRR 4,972.24 | ROR 82,745.25 | 324 serious
   Evidence: "Meningioma: Discontinue Depo-Provera CI if meningioma is diagnosed."
   Guardrail banner: ✅ All figures traced to tool output · traceability 100%
   Audit trail: 2 tool calls, 3.2s elapsed
```

---

## 28. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER (PV Analyst)                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                    UI (Streamlit — 3 pane)                              │
│  Left: Drug+Event input / Model selector                                │
│  Centre: Signal badge / Stats table / AI interpretation / Guardrail    │
│  Right: Audit trail / Evaluation panel                                  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│           LangChain AgentExecutor (LangChain 0.3.x)                    │
│           create_tool_calling_agent + return_intermediate_steps=True    │
│           Token budget: DEV_MODE → 300 / prod → 1024                   │
└──────┬──────────────┬──────────────┬──────────────┬─────────────────────┘
       │              │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│resolve_drug │ │resolve_    │ │calculate_│ │search_      │
│_name        │ │event_name  │ │pv_stats  │ │literature   │
│RapidFuzz vs │ │RapidFuzz vs│ │PythonAst │ │Hybrid FAISS │
│signal_df    │ │signal_df   │ │REPLTool  │ │+BM25+RRF    │
│vocabulary   │ │vocabulary  │ │→ pandas  │ │+reranker    │
└──────┬──────┘ └─────┬──────┘ └────┬─────┘ └──────┬──────┘
       │              │              │              │
┌──────▼──────────────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│      signal_df             │ │signal_   │ │FAISS indexes│
│  (signal_summary.parquet)  │ │summary.  │ │literature + │
│  PRR, ROR, counts          │ │parquet   │ │labels +     │
│  in-memory DataFrame       │ │          │ │signals      │
└────────────────────────────┘ └──────────┘ └──────┬──────┘
                                                    │
                                         FastEmbed BAAI/bge-small-en-v1.5
                                         (ONNX, local, ~50MB)
       ┌──────────────────────────────────────────────────────────┐
       │              LLM Generation (temp=0)                     │
       │  Groq: gpt-oss-120b / qwen3.8-27b / gpt-oss-20b         │
       │  Mistral: ministral-8b / ministral-14b                   │
       │  Format A (stats) or Format B (prose)                    │
       └──────────────────────────┬───────────────────────────────┘
                                  │
       ┌──────────────────────────▼───────────────────────────────┐
       │                POST-GENERATION GUARDRAILS                │
       │  Regex: numeric faithfulness + drug match + empty check  │
       │  LLM Judge (Groq gpt-oss-120b, temp=0):                 │
       │    numeric_faithfulness + causal_language + grounded     │
       └──────────────────────────┬───────────────────────────────┘
                                  │
       ┌──────────────────────────▼───────────────────────────────┐
       │              EVALUATION (offline, manual)                │
       │  Golden test set (3 queries + ground truth)              │
       │  Per-model: key numbers + regex + judge                  │
       │  RAGAS: faithfulness + relevancy + context_recall        │
       └──────────────────────────────────────────────────────────┘

DATA PIPELINE (runs once at ingestion time):
FAERS ASCII (.txt, $-delimited)
  → faers_ingest.py: load, clean, join on primaryid
  → signal_summary.parquet (one row per drug-event pair, PRR+ROR pre-computed)
  → pdf_ingest.py: PDF → chunks (size=800, overlap=120)
  → build_vector_stores.py: chunks → embeddings → 3 FAISS indexes
```

---

## 29. Component Table

| Component | Technology | Purpose | Input | Output | Why used | Status |
|---|---|---|---|---|---|---|
| UI | Streamlit | User interface | User clicks | Structured display | Rapid POC UI | ✅ Implemented |
| Ingestion | pandas + pyarrow | Load/clean/join FAERS | Raw .txt files | signal_summary.parquet | Deterministic, scalable | ✅ Implemented |
| PRR/ROR calc | pandas (runtime) | Compute statistics | signal_df | PRR, ROR, counts | Reproducible, no hallucination | ✅ Implemented |
| Drug resolution | RapidFuzz | Fuzzy name matching | User input | Exact FAERS spelling | Handles typos, casing | ✅ Implemented |
| PDF chunking | LangChain RecursiveCharacterTextSplitter | Split PDF text | PDF text | Text chunks | Fits LLM context window | ✅ Implemented |
| Embeddings | FastEmbed BAAI/bge-small-en-v1.5 | Semantic vectors | Text chunks | 384-dim vectors | Local, no GPU, fast | ✅ Implemented |
| Vector DB | FAISS | Similarity search | Query vector | Top-K chunks | Local, free, sufficient for POC | ✅ Implemented |
| Sparse retrieval | BM25 | Keyword search | Query text | Top-K chunks | Exact term matching | ✅ Implemented |
| Fusion | RRF (k=60) | Merge ranked lists | 2 ranked lists | 1 ranked list | Combines semantic+keyword | ✅ Implemented |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Rerank retrieved chunks | Fused list | Reranked top-3 | Better precision | ✅ Implemented |
| Agent orchestration | LangChain AgentExecutor | Tool calling loop | Query | Answer + steps | Handles tool-call lifecycle | ✅ Implemented |
| Main LLM | Groq / Mistral | Generate answer | Prompt + tool results | Structured text | Fast, free tier | ✅ Implemented |
| LLM Judge | Groq gpt-oss-120b | Safety evaluation | Answer + tool outputs | pass/fail JSON | LLM-level semantic check | ✅ Implemented |
| Regex guardrail | Python re + validators | Numeric traceability | Answer + tool outputs | pass/fail | Fast, free, deterministic | ✅ Implemented |
| RAGAS eval | RAGAS + Groq | Quality scoring | Answer + contexts + ground_truth | 0–1 scores | Standard RAG eval framework | ✅ Implemented |
| Token budget | DEV_MODE env flag | Control API spend | .env setting | max_tokens 300/1024 | Free tier management | ✅ Implemented |
| Audit trail | Python logging + UI | Traceability | All intermediate steps | Logged output | Pharma accountability | Partial |
| MedDRA | Not implemented | Event hierarchy | — | — | Requires licence | ❌ Future |
| LangGraph | Not implemented | Modern agent framework | — | — | Requires LangChain 1.x migration | ❌ Future |

---

## 30. Interview Questions

### Data

| Question | Expected answer direction |
|---|---|
| Why FAERS? | Public, de-identified, quarterly, FDA-standard, largest spontaneous reporting DB |
| How large is this POC dataset? | 2026Q1+Q2: 819,683 reports (DEMO). 3.33M drug rows, 2.73M reaction rows, 595k outcome rows. Exploded DRUG×REAC join = 39.17M rows. Signal summary = 811,095 pairs (8,357 drugs, 7,770 events). FAISS signals index keeps only pairs with a≥500 (1,857). |
| What is a primaryid? | Unique report submission ID — joins DEMO/DRUG/REAC/OUTC |
| Why can one case have multiple drugs? | Polypharmacy — patient may take many drugs simultaneously |
| Why can one case have multiple reactions? | A patient can experience multiple adverse events |
| Why combine Q1 and Q2? | More data = more stable PRR/ROR estimates, broader coverage |
| Why Parquet? | Column-oriented, compressed, fast reads, schema-stable |
| Why not send raw TXT to LLM? | Too large, LLM can't join tables, LLM would hallucinate statistics |
| How are drug names normalised? | `.strip().upper()` at ingestion, RapidFuzz fuzzy match at query time |
| How are duplicates handled? | `drop_duplicates(primaryid)` in DEMO — full caseid-based dedup is a future enhancement |

### Statistics

| Question | Expected answer |
|---|---|
| What is PRR? | Proportional Reporting Ratio = (a/(a+b)) / (c/(c+d)) — measures disproportionality |
| What is the 2×2 table? | a=drug+event, b=drug+no event, c=event+no drug, d=neither |
| Why use Python not LLM for PRR? | Deterministic, reproducible, LLM hallucinates numbers |
| Why not let the LLM do the complex calculation? | PRR needs a 2×2 over 819k reports and distinct primaryid counts. The model cannot hold 39M join rows, mixes join-rows with cases, invents formulas, and is non-reproducible. Ingestion precomputes parquet; the LLM only writes a pandas filter; Python executes it. |
| What does PRR=4972 mean? | MENINGIOMA is reported 4972× more often with DEPO-PROVERA than with other drugs |
| Does PRR=4972 prove causation? | No — it is a reporting signal, not causal evidence |
| How are Cases and Serious related? | Cases = reports with both drug and event (cell a). Serious = the subset of those reports with a FAERS outcome code DE/HO/LT/DS/CA/RI. They are not the same count. |
| Why 9,684 cases / 324 serious and still High? | High is PRR ≥ 10, not serious %. PRR is 4,972 so the pair is High. 324/9,684 = 3.3% only means most of those reports were not coded with a serious-outcome flag. Meningioma can be clinically important and still not coded as serious in FAERS. |
| What are FAERS limitations? | Voluntary reporting bias, no denominator, duplicate reports, indication confounding |
| What is ROR? | Reporting Odds Ratio = (a×d)/(b×c) — similar to PRR, used as cross-check |

### RAG

| Question | Expected answer |
|---|---|
| What is RAG? | Retrieval-Augmented Generation — ground LLM in retrieved evidence |
| Why RAG not just LLM? | LLM training data is stale, can't know specific FAERS numbers |
| Where is RAG used? | PDFs only: ingest → FAISS/BM25 → tools `search_literature` / `search_signal_evidence` / `search_drug_label` → UI cards + RAGAS. Not used for PRR/ROR. |
| Why not RAG on the FAERS dataset? | FAERS is tables, not prose. 39M join rows / 811k pairs cannot be prompted. FAISS returns “similar text,” not the exact 2×2. Drug and event are on different rows joined by primaryid. Pandas already has an exact (drug, pt) key. |
| Why don’t we embed the FAERS dataset / use a vector index for it? | Embeddings find related wording. FAERS needs the exact (drug, event) row and a 2×2 formula. Cosine similarity does not compute PRR. `(drugname, pt)` in parquet is already the index. We only embed PDFs (plus 1,857 high-count pair text cards that must not supply PRR). |
| What retrieval method? | Hybrid: FAISS (dense) + BM25 (sparse) fused with RRF |
| Why hybrid? | FAISS handles paraphrased content; BM25 handles exact drug names |
| What is RRF? | Reciprocal Rank Fusion — merges ranked lists: score = sum(1/(k+rank)) |
| Why top-3 not top-10? | Token budget — free tier limits; reranker selects best 3 from fused 20 |
| Why 10+10 before reranking? | Cast wide net with both retrieval methods before precision-selecting |

### Chunking

| Question | Expected answer |
|---|---|
| What is chunked? | Only PDFs (labels + literature). FAERS tabular data is NOT chunked. |
| Why chunk? | LLM context window cannot hold full PDFs |
| Chunk size 800, overlap 120 — why? | Fits context budget, overlap prevents sentence splitting at boundaries |
| Why not chunk FAERS rows? | They are structured data — better handled by pandas exact lookup |

### LangChain

| Question | Expected answer |
|---|---|
| Why LangChain? | Tool schema, agent loop, model swapping, intermediate steps |
| Where is LangChain used? | Agent (`create_tool_calling_agent` + `AgentExecutor`), tools (`@tool`, `PythonAstREPLTool`), prompts, FAISS/BM25/embeddings/PDF loaders, judge `ChatGroq`, RAGAS `ChatGroq`. LangGraph is not the live loop. |
| What is create_tool_calling_agent? | Binds tools to the chat model as native JSON function calls (not ReAct text). AgentExecutor loops until the model returns a final answer with no tool_calls. |
| What is AgentExecutor? | Runs LLM → tool calls → observe → LLM → repeat until done |
| What is a tool? | Python function decorated with @tool, exposed to the agent with a schema |
| What is return_intermediate_steps? | Returns all tool calls + outputs for audit trail |
| Why not LangGraph? | Would require full agent rewrite — deferred as future enhancement |

### LLM Models

| Question | Expected answer |
|---|---|
| Which models? | Groq: gpt-oss-120b, qwen3.8-27b, gpt-oss-20b. Mistral: ministral-8b, ministral-14b |
| Model vs. provider? | Model = neural weights. Provider = inference hardware. Groq serves model on LPU. |
| What is context window vs. TPM? | Context = what fits in one call. TPM = tokens per minute rate limit. Independent. |
| Why did HTTP 429 occur? | 7049 + 1737 > 8000 TPM limit — judge call consumed remaining quota |
| How to handle 429? | Sleep, reduce max_tokens, use DEV_MODE, sequential evaluation |
| Which model performed best? | qwen3.8-27b (fastest, 1.9s), gpt-oss-120b (cleanest output) |
| Which model failed and why? | gpt-oss-20b: encoding bug. ministral-8b: causal language violation. |

### Evaluation

| Question | Expected answer |
|---|---|
| What is faithfulness? | Did the model only say things that came from retrieved tool output? |
| What is context recall? | Did the model include all key facts from the ground truth? |
| What is a golden test set? | Hand-written questions with correct expected answers verified from real data |
| Why hand-written, not LLM-generated? | LLM cannot generate correct PRR/ROR — circular validation. Numbers come from data. |
| Why use LLM as judge? | LLMs are good at detecting causal language and grounding — not at verifying math |
| Why a different model for the judge? | Judge is always Groq gpt-oss-120b so scoring is consistent. If the answer model is Mistral 14B, the long write-up does not consume Groq TPM — only the short JSON judge does. If both are Groq 120b they share ~8k TPM and can 429. |
| What RAGAS metrics are used? | faithfulness, answer_relevancy, context_recall |

### Guardrails

| Question | Expected answer |
|---|---|
| Why regex guardrail? | Fast, free, catches numeric hallucinations deterministically |
| Why LLM judge? | Semantic checks regex can't do — causal language, unsupported literature claims |
| What is traceability %? | Percentage of answer numbers that trace back to tool output |
| What is the empty-evidence check? | If tools returned nothing, model must say so — not answer from memory |
| Why ban "causes/confirms"? | FAERS signals are correlational — causal language is scientifically wrong and dangerous |

### Architecture

| Question | Expected answer |
|---|---|
| Why Pandas + RAG + LLM? | Pandas for deterministic stats, RAG for grounded evidence, LLM for explanation |
| What happens if LLM calculates PRR? | Hallucination risk — LLM invents plausible-sounding but wrong numbers |
| How would you scale? | Cloud vector DB (Pinecone), async agents, enterprise LLM endpoints, RBAC |
| What would change for production? | Full audit logging, MedDRA, duplicate detection, LangGraph, enterprise auth |

### Pharmacovigilance

| Question | Expected answer |
|---|---|
| What is pharmacovigilance? | Science of monitoring drug safety after market approval |
| What is a safety signal? | Pattern suggesting a possible drug-adverse event relationship requiring investigation |
| What is disproportionality analysis? | Statistical method comparing reporting rate of a drug-event pair vs background |
| Does a signal mean causation? | No — it is a hypothesis generator requiring expert clinical review |
| What is human-in-the-loop? | Qualified PV professional must review and decide — AI assists, not decides |
| Why is traceability important in pharma? | Regulatory accountability — every AI-assisted finding must be explainable and auditable |

---

## 31. Manager Summary

**2-minute explanation:**

"We built a POC that helps pharmacovigilance analysts investigate safety signals for drugs. Instead of manually reviewing thousands of FAERS reports, an analyst types a drug name and adverse event — for example, Depo-Provera and meningioma — and the system instantly retrieves the statistical signal strength, supporting text from the drug label, and a plain-language explanation.

The key design principle is: statistics are computed from real data using Python, not guessed by the AI. The AI only explains what the data shows. Every answer goes through two safety checks — a numeric verification that all numbers trace back to the actual database, and an AI safety judge that checks the answer doesn't make unsupported causal claims.

We tested five different AI models. Three passed all checks. Two failed — one for a text encoding bug, one for making causal claims the data doesn't support. The system surfaces these failures so the analyst knows when to be cautious. It's a tool to help PV teams work faster and more consistently, not to replace expert review."

---

## 32. Technical Summary

**5-minute technical explanation:**

The system has three layers:

**Layer 1 — Data (deterministic):** Raw FAERS ASCII files from FDA (Q1+Q2 2026) are loaded with pandas, cleaned (uppercase normalisation, null removal), joined once on `primaryid`, and persisted as parquet. A pre-aggregated signal summary computes PRR and ROR for every drug-event pair offline. At query time, the agent authors a pandas filter expression; Python executes it against the in-memory DataFrame. Numbers are mathematically certain.

**Layer 2 — Retrieval (hybrid):** Drug labels and clinical literature PDFs are chunked (800 chars, 120 overlap) and embedded with FastEmbed BAAI/bge-small-en-v1.5 into three FAISS indexes. At query time, FAISS dense retrieval and BM25 sparse retrieval each return 10 candidates. Reciprocal Rank Fusion merges them; a cross-encoder reranker picks the top 3. The stats lookup is a direct O(n) DataFrame filter — no vectors involved.

**Layer 3 — Agent + Guardrails:** A LangChain AgentExecutor with five tools orchestrates the flow. For a signal lookup query, the agent issues two parallel tool calls in one LLM step (stats + literature), then synthesises a structured answer. Two guardrail layers run post-generation: a regex check verifying every number >10 traces to tool output (within 1% tolerance), and an LLM Judge (separate Groq call, temp=0) scoring numeric faithfulness, causal language, and evidence grounding. Results and the full tool call audit trail are shown in the Streamlit UI.

RAGAS evaluation (offline) scores faithfulness, answer relevancy, and context recall against a 3-query golden test set with real ground-truth answers. A model comparison script runs all five models on the same UI-equivalent query and reports pass/fail per check, elapsed time, and tool call count.
