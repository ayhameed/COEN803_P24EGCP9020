# COEN803 Solo Assignment — Abdulhameed Yunusa (P24EGCP9020)

## Overview

UEIR-Gated RAG Pipeline — an NLP/RAG application that uses UEIR v3.5 forbidden-band gating as a real document filtering decision. Chunks whose UEIR conductance (phi) falls in the forbidden band `(0.5, 0.6]` are excluded from the LLM context window before answering a query.

---

## Files

| File | Description |
|---|---|
| `COEN803_Solo_entry.py` | Main submission file — scorecard, application, break report |
| `ueir_v3_5.py` | UEIR v3.5 reference library (do not modify) |
| `requirements.txt` | Python dependencies |
| `.env` | API key |

---

## Setup
### 1. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv env
source env/bin/activate
```

Windows Command Prompt:

```bat
py -m venv env
env\Scripts\activate
```

Windows PowerShell:

```powershell
py -m venv env
.\env\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

Or install manually:

```bash
pip3 install anthropic numpy scipy
```

### 3. Add your Anthropic API key

Create a `.env` file in the project folder:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the program

macOS/Linux:

```bash
export $(cat .env | xargs) && python3 COEN803_Solo_entry.py
```

Windows Command Prompt:

```bat
set ANTHROPIC_API_KEY=your_api_key_here
python COEN803_Solo_entry.py
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "your_api_key_here"
python COEN803_Solo_entry.py
```

---

## What it does

1. Runs the mandatory UEIR v3.5 six-point scorecard — all 6 checks must pass before the application starts.
2. Runs the UEIR orchestrator (`max_depth=24`) to produce `phi_eff`, `stability_score`, and `forbidden_events`.
3. Assigns each of 8 NLP/RAG document chunks to a UEIR trace step by index.
4. Applies the forbidden-band gate: chunks with phi in `(0.5, 0.6]` are **rejected**; the rest are **approved**.
5. Passes only approved chunks as context to Claude (Haiku) to answer the query.
6. Runs the Break Report: two genuine attempts to break v3.5, with documented outcomes and proposed fixes.

---

## UEIR Metrics (verified output)

| Metric | Value |
|---|---|
| phi_eff | 0.48899 |
| stability_score | 0.991537 |
| forbidden_events | 10 |
| approved_chunks | 5 / 8 |
| rejected_chunks | 3 / 8 |

---

## Terminal Output

```
============================================================
COEN803 Solo Assignment — Abdulhameed Yunusa (P24EGCP9020)
UEIR v3.5 Scorecard
============================================================
  OK 1. Phi in (0,1) at every step                 [min=0.4805, max=0.5203]
  OK 2. cat in [0,4] — no fallback                 [cats: [0, 1, 2, 3, 4]]
  OK 3. Forbidden band triggered                   10 events
  OK 4. |final - 0.5| < 0.05                       [distance=0.004232]
  OK 5. Cat 4: 64 primes via sieve                 [293..691]
  OK 6. k=181 structurally reached                 at depth(s) [17]

  ALL CHECKS PASS — proceed to your application.
============================================================

Solo Results Summary
----------------------------------------
  student          : Abdulhameed Yunusa
  reg_no           : P24EGCP9020
  application      : UEIR-Gated RAG Pipeline
  domain           : NLP/RAG
  phi_eff          : 0.48899
  stability        : 0.991537
  forbidden_events : 10
  query            : Explain how retrieval quality is measured and improved in RAG pipelines.
  corpus_size      : 8
  approved_chunks  : 5
  rejected_chunks  : 3
```

---

## LLM Response (Claude Haiku)

> **Query:** Explain how retrieval quality is measured and improved in RAG pipelines.
>
> **Approved context:** Chunks 0–4 (Chunks 5–7 rejected by UEIR forbidden-band gate)

```
# Retrieval Quality Measurement and Improvement in RAG Pipelines

Based on the provided excerpts, I can address how retrieval quality is **improved**
in RAG systems, though the documents do not contain explicit information about
measurement metrics.

## How Retrieval Quality is Improved:

**1. Dual Retrieval Approaches:**
- Dense retrieval using vector embeddings that represent semantic meaning in
  high-dimensional space, enabling similarity search via cosine or dot-product distance
- Sparse retrieval using BM25, a probabilistic method based on term frequency (TF)
  and inverse document frequency (IDF), effective for keyword-heavy queries

**2. Strategic Chunking:**
Different chunking strategies balance recall and context coherence:
- Fixed-size windows
- Sentence-level splits
- Semantic boundary detection

**3. Re-ranking with Cross-Encoders:**
Re-ranking improves retrieval precision by scoring query-chunk pairs jointly rather
than independently, allowing for more accurate assessment of relevance.

## Limitation:
The provided excerpts do not contain information about how retrieval quality is
measured (e.g., precision, recall, NDCG, MRR). Additional documentation would
be needed to fully answer that aspect.
```