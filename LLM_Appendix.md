# LLM Appendix — COEN803 Solo Assignment

**Student:** Abdulhameed Yunusa
**Reg. No:** P24EGCP9020
**Application:** UEIR-Gated RAG Pipeline (NLP/RAG Domain)
**Date:** 2026-07-04

This appendix documents the single LLM call made by `run_solo()` in
`COEN803_Solo_entry.py`, for the same run reported in the IEEE paper
(Φ_eff = 0.48899, stability = 0.991537, forbidden_events = 10). The prompts
below are reproduced verbatim as constructed by the code; the response is the
captured output of that call.

---

## Call Metadata

| Field | Value |
|---|---|
| Provider | Anthropic |
| Model | `claude-haiku-4-5-20251001` |
| `max_tokens` | 1024 |
| Temperature | default (1.0 — not set in code) |
| API method | `client.messages.create()` |
| Number of LLM calls in application | 1 |

---

## UEIR Gate Context (pre-call)

Before the LLM call, the UEIR forbidden-band gate was applied to all 8 document
chunks. Each chunk `i` is bound to trace step `i mod len(trace)`, inheriting that
step's conductance Φ and forbidden flag. Chunks whose Φ ∈ (0.5, 0.6] were
**excluded** from the context window. Values are the deterministic output of the
recursion (`max_depth = 24`, `initial_state = 0.5`).

| Chunk | Depth | Φ | Forbidden | Decision |
|---|---|---|---|---|
| 0 | 0 | 0.5000 | No | APPROVED |
| 1 | 1 | 0.4815 | No | APPROVED |
| 2 | 2 | 0.4857 | No | APPROVED |
| 3 | 3 | 0.4831 | No | APPROVED |
| 4 | 4 | 0.4947 | No | APPROVED |
| 5 | 5 | 0.5067 | **Yes** | **REJECTED** |
| 6 | 6 | 0.5184 | **Yes** | **REJECTED** |
| 7 | 7 | 0.5198 | **Yes** | **REJECTED** |

Only chunks 0–4 (5 of 8) were passed to the model. Chunks 5–7 — covering hybrid
retrieval (RRF), hallucination, and UEIR conductance — were structurally
excluded by the gate. Chunk 0 sits exactly on the attractor value Φ = 0.5000
(k = 181, n = 2k = 362), which is not in the forbidden band `(0.5, 0.6]` and is
therefore approved.

---

## System Prompt (verbatim)

The system prompt is a single concatenated string as passed to the API:

```
You are an expert in Natural Language Processing and Retrieval-Augmented Generation (RAG). Answer the user's question using ONLY the provided document excerpts. If the excerpts do not contain sufficient information to answer fully, say so explicitly. Do not hallucinate or add information not present in the excerpts.
```

---

## User Prompt (verbatim)

Reproduced exactly as assembled by the code — the header, a blank line, one
approved chunk per line, a blank line, then the question:

```
Document excerpts approved by UEIR forbidden-band gate (phi NOT in forbidden band (0.5, 0.6]):

[Chunk 0 | UEIR-APPROVED] Retrieval-Augmented Generation (RAG) combines dense retrieval with generative models to produce grounded, citation-backed responses.
[Chunk 1 | UEIR-APPROVED] Vector embeddings represent the semantic meaning of text in high-dimensional space, enabling similarity search via cosine or dot-product distance.
[Chunk 2 | UEIR-APPROVED] BM25 is a probabilistic sparse retrieval method based on term frequency (TF) and inverse document frequency (IDF), effective for keyword-heavy queries.
[Chunk 3 | UEIR-APPROVED] Chunking strategies for RAG include fixed-size windows, sentence-level splits, and semantic boundary detection to balance recall and context coherence.
[Chunk 4 | UEIR-APPROVED] Re-ranking with cross-encoders improves retrieval precision by scoring query-chunk pairs jointly rather than independently.

Question: Explain how retrieval quality is measured and improved in RAG pipelines.
```

---

## Model Response (verbatim)

```
# Retrieval Quality in RAG Pipelines

Based on the provided excerpts, I can explain some methods for improving retrieval quality,
though the documents do not provide comprehensive coverage of how quality is *measured*.

## Methods for Improving Retrieval Quality

**Re-ranking with Cross-Encoders:**
The excerpts indicate that retrieval precision can be improved through re-ranking using
cross-encoders, which score query-chunk pairs *jointly* rather than independently. This
represents a refinement step after initial retrieval.

**Chunking Strategies:**
The documents mention that chunking strategies—including fixed-size windows, sentence-level
splits, and semantic boundary detection—help "balance recall and context coherence," which
suggests these design choices impact retrieval effectiveness.

**Retrieval Method Selection:**
Two complementary approaches are described:
- Dense retrieval (vector embeddings): Uses semantic similarity via cosine or dot-product distance
- Sparse retrieval (BM25): A probabilistic method based on term frequency and inverse document
  frequency, noted as "effective for keyword-heavy queries"

## Limitations

The provided excerpts do not contain explicit information about:
- How retrieval quality is *measured* (e.g., metrics like precision, recall, MRR, NDCG, etc.)
- Quantitative benchmarking approaches
- Evaluation frameworks or datasets

To fully answer your question about measurement methodologies, additional source material
would be needed.
```

---

## Output Validation

The model response was validated on two criteria before being accepted:

1. **Grounding check:** The model answered only from the 5 approved chunks and did not introduce any information from the 3 rejected chunks (hybrid retrieval / RRF, hallucination, and UEIR conductance). No content unique to those rejected chunks appears in the response.

2. **Hallucination check:** The model correctly identified that the approved context did not contain retrieval *measurement* metrics (e.g., precision, recall, MRR, NDCG) and explicitly flagged the gap rather than fabricating an answer. This is the expected behaviour for a well-constrained RAG system operating on gated context.

Both checks passed. The response was accepted as the final output of the pipeline.

---

*Note on reproducibility:* the gate decisions and Φ values above are deterministic
and will reproduce exactly on any run of `COEN803_Solo_entry.py`. The model
response is a single captured generation at temperature 1.0; re-running the LLM
call may yield differently-worded output, so this exact text is frozen here and
matches Appendix B of the IEEE report.
