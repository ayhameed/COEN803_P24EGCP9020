"""
COEN803: Advanced Programming — Individual Solo Assignment Entry File
=====================================================================
Course  : COEN803 Advanced Programming, ABU Zaria
Issued  : Prof. T. H. Sikiru — RANA Technologies & UEIR Quantum Lab
Session : 2025/2026

Instructions
────────────
1. Place this file in the SAME FOLDER as ueir_v3_5.py.
   Do NOT modify ueir_v3_5.py.

2. Fill in your name and registration number below.

3. Run this file. The UEIR Scorecard must show ALL 6 checks ✓
   before you write any application code.

4. Implement your individual application in run_solo().
   The application must be ORIGINAL — not a copy of any group task.

5. Wire your LLM calls in call_llm(). Document every prompt.
"""

import json
import math
import numpy as np
import anthropic
from typing import Dict, Any

# ── Your details ──────────────────────────────────────────────────────────────
YOUR_NAME    = "Abdulhameed Yunusa"
YOUR_REG_NO  = "P24EGCP9020"

# ── UEIR v3.5 import ──────────────────────────────────────────────────────────
from ueir_v3_5 import (
    recursive_ueir_step, UEIR_Orchestrator,
    UEIR_BlackHole, UEIR_Cosmology,
    CATEGORIES, K_SEED, N_SCREEN,
    assign_cat, get_prime, compute_n,
    is_forbidden, compute_conductance,
)


# ── Mandatory scorecard ───────────────────────────────────────────────────────
def run_scorecard() -> bool:
    print("=" * 60)
    print(f"COEN803 Solo Assignment — {YOUR_NAME} ({YOUR_REG_NO})")
    print("UEIR v3.5 Scorecard")
    print("=" * 60)

    orch   = UEIR_Orchestrator(max_depth=24)
    result = orch.run(initial_state=0.5)
    trace  = orch.trace

    checks = {
        1: ("Phi in (0,1) at every step",
            all(0 < t["phi"] < 1 for t in trace),
            f"[min={min(t['phi'] for t in trace):.4f}, max={max(t['phi'] for t in trace):.4f}]"),
        2: ("cat in [0,4] — no fallback",
            all(0 <= t["cat"] <= 4 for t in trace),
            f"[cats: {sorted(set(t['cat'] for t in trace))}]"),
        3: ("Forbidden band triggered",
            result["forbidden_events"] > 0,
            f"{result['forbidden_events']} events"),
        4: ("|final - 0.5| < 0.05",
            abs(orch.final_state - 0.5) < 0.05,
            f"[distance={abs(orch.final_state-0.5):.6f}]"),
        5: ("Cat 4: 64 primes via sieve",
            len(CATEGORIES[4]) == 64,
            f"[{CATEGORIES[4][0]}..{CATEGORIES[4][-1]}]"),
        6: ("k=181 structurally reached",
            len(result['k181_at_depths']) > 0,
            f"at depth(s) {result['k181_at_depths']}"),
    }

    all_pass = True
    for i, (label, ok, detail) in checks.items():
        print(f"  {'OK' if ok else 'FAIL'} {i}. {label:<42s} {detail}")
        if not ok: all_pass = False

    print()
    if all_pass:
        print("  ALL CHECKS PASS — proceed to your application.")
    else:
        print("  ONE OR MORE CHECKS FAILED. Fix before proceeding.")
    print("=" * 60)
    print()
    return all_pass


# ── LLM stub ─────────────────────────────────────────────────────────────────
def call_llm(system: str, user: str) -> str:
    """Calls Anthropic Claude. Reads ANTHROPIC_API_KEY from environment."""
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


# ── Break Report ──────────────────────────────────────────────────────────────
def _run_break_report() -> Dict[str, Any]:
    """
    Two genuine attempts to break UEIR v3.5, with honest documentation
    of what happened and a proposed fix for each.
    """
    # Attempt 1: truncate max_depth to 4
    # Forbidden events first occur at depth 5; k=181 emerges at depth 17.
    # Cutting at depth 4 should kill both checks 3 and 6.
    orch1   = UEIR_Orchestrator(max_depth=4)
    result1 = orch1.run(initial_state=0.5)
    c3_pass = result1["forbidden_events"] > 0
    c6_pass = len(result1["k181_at_depths"]) > 0

    attempt1 = {
        "method":           "Set max_depth=4 (truncate recursion early)",
        "rationale":        "Forbidden band first fires at depth 5; k=181 emerges at depth 17. "
                            "Stopping at depth 4 should prevent both from occurring.",
        "forbidden_events": result1["forbidden_events"],
        "k181_at_depths":   result1["k181_at_depths"],
        "check_3_passed":   c3_pass,
        "check_6_passed":   c6_pass,
        "what_happened":    (
            f"Check 3 (forbidden band triggered): {'PASS' if c3_pass else 'FAIL'} "
            f"— {result1['forbidden_events']} events. "
            f"Check 6 (k=181 structurally reached): {'PASS' if c6_pass else 'FAIL'} "
            f"— k=181 at depths {result1['k181_at_depths']}."
        ),
        "proposed_fix":     "Add assertion in run_scorecard(): "
                            "assert orch.max_depth >= 24, "
                            "'max_depth must be >= 24 to reach all structural features.'",
    }

    # Attempt 2: unphysical initial_state = 10.0
    # phi is computed as k/n (independent of initial_state), so phi bounds
    # are preserved by construction. The question is whether the state itself
    # still converges to the 0.5 attractor (check 4).
    orch2      = UEIR_Orchestrator(max_depth=24)
    result2    = orch2.run(initial_state=10.0)
    final2     = orch2.final_state
    dist2      = abs(final2 - 0.5)
    c4_pass    = dist2 < 0.05
    phi_ok     = all(0 < t["phi"] < 1 for t in orch2.trace)

    attempt2 = {
        "method":              "Set initial_state=10.0 (far outside physical range)",
        "rationale":           "Test whether the attractor pull is strong enough to converge "
                               "from an unphysical starting state, and whether phi bounds break.",
        "final_state":         round(final2, 6),
        "attractor_distance":  round(dist2, 6),
        "phi_bounds_preserved": phi_ok,
        "check_4_passed":      c4_pass,
        "what_happened":       (
            f"Phi bounds: {'preserved' if phi_ok else 'VIOLATED'} — phi is k/n, "
            "independent of initial_state, so it stays in (0,1) regardless. "
            f"Attractor convergence (check 4): {'PASS' if c4_pass else 'FAIL'} — "
            f"|final − 0.5| = {dist2:.6f}."
        ),
        "proposed_fix":        "Add input validation in UEIR_Orchestrator.run(): "
                               "assert 0 < initial_state < 1, "
                               "'initial_state must be in (0, 1).'",
    }

    return {"attempt_1": attempt1, "attempt_2": attempt2}


# ── Your application ──────────────────────────────────────────────────────────
def run_solo() -> Dict[str, Any]:
    """
    Application: UEIR-Gated RAG Pipeline (NLP/RAG domain).

    Each document chunk is assigned to a UEIR trace step by index.
    If that step's phi falls in the forbidden band (0.5, 0.6], the chunk
    is EXCLUDED from the LLM context window — a real functional decision,
    not a visual label.

    UEIR metrics reported: phi_eff, stability_score, forbidden_events.
    """
    # 1. Run UEIR orchestrator
    orch   = UEIR_Orchestrator(max_depth=24)
    result = orch.run(initial_state=0.5, document_context=f"rag_nlp_{YOUR_REG_NO}")

    phi_eff   = result["effective_conductance"]
    stability = result["stability_score"]
    n_forb    = result["forbidden_events"]
    trace     = result["trace"]

    # 2. NLP/RAG document corpus (8 chunks)
    corpus = [
        {"id": 0, "text": "Retrieval-Augmented Generation (RAG) combines dense retrieval with generative models to produce grounded, citation-backed responses."},
        {"id": 1, "text": "Vector embeddings represent the semantic meaning of text in high-dimensional space, enabling similarity search via cosine or dot-product distance."},
        {"id": 2, "text": "BM25 is a probabilistic sparse retrieval method based on term frequency (TF) and inverse document frequency (IDF), effective for keyword-heavy queries."},
        {"id": 3, "text": "Chunking strategies for RAG include fixed-size windows, sentence-level splits, and semantic boundary detection to balance recall and context coherence."},
        {"id": 4, "text": "Re-ranking with cross-encoders improves retrieval precision by scoring query-chunk pairs jointly rather than independently."},
        {"id": 5, "text": "Hybrid retrieval fuses dense and sparse scores via Reciprocal Rank Fusion (RRF) to capture both semantic and lexical relevance signals."},
        {"id": 6, "text": "Hallucination in LLMs occurs when the model generates plausible but factually incorrect content not grounded in the retrieved context."},
        {"id": 7, "text": "UEIR conductance Phi = k/n measures the stability of information flow across recursive depth layers; the attractor at Phi=0.5 signals balanced retrieval confidence."},
    ]

    # 3 & 4. Apply UEIR forbidden-band gate per chunk
    approved     = []
    rejected     = []
    chunk_report = []

    for chunk in corpus:
        step      = trace[chunk["id"] % len(trace)]
        phi_chunk = step["phi"]
        forbidden = step["forbidden"]   # True if phi in (0.5, 0.6]

        chunk_report.append({
            "chunk_id":      chunk["id"],
            "ueir_depth":    step["depth"],
            "phi":           phi_chunk,
            "forbidden_gate": forbidden,
            "decision":      "REJECT" if forbidden else "APPROVE",
            "text_excerpt":  chunk["text"][:70] + "...",
        })

        if forbidden:
            rejected.append(chunk)
        else:
            approved.append(chunk)

    # 5. Build LLM context from approved chunks only
    query = "Explain how retrieval quality is measured and improved in RAG pipelines."

    # 6. Verbatim prompts
    system_prompt = (
        "You are an expert in Natural Language Processing and Retrieval-Augmented Generation (RAG). "
        "Answer the user's question using ONLY the provided document excerpts. "
        "If the excerpts do not contain sufficient information to answer fully, say so explicitly. "
        "Do not hallucinate or add information not present in the excerpts."
    )

    if approved:
        context_str = "\n".join(
            f"[Chunk {c['id']} | UEIR-APPROVED] {c['text']}" for c in approved
        )
        user_prompt = (
            f"Document excerpts approved by UEIR forbidden-band gate "
            f"(phi NOT in forbidden band (0.5, 0.6]):\n\n"
            f"{context_str}\n\n"
            f"Question: {query}"
        )
    else:
        user_prompt = (
            "WARNING: All document chunks were rejected by the UEIR forbidden-band gate. "
            "No approved context is available. "
            "Please acknowledge that you cannot answer due to insufficient approved context."
        )

    # 7. Call LLM
    llm_response = call_llm(system_prompt, user_prompt)

    # 8. Break Report
    break_report = _run_break_report()

    # 9. Return full result dict
    return {
        "student":          YOUR_NAME,
        "reg_no":           YOUR_REG_NO,
        "application":      "UEIR-Gated RAG Pipeline",
        "domain":           "NLP/RAG",
        "phi_eff":          phi_eff,
        "stability":        stability,
        "forbidden_events": n_forb,
        "query":            query,
        "system_prompt":    system_prompt,
        "user_prompt":      user_prompt,
        "llm_response":     llm_response,
        "corpus_size":      len(corpus),
        "approved_chunks":  len(approved),
        "rejected_chunks":  len(rejected),
        "chunk_gate_report": chunk_report,
        "break_report":     break_report,
        "ueir_result":      result,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not run_scorecard():
        raise SystemExit("Fix the scorecard before proceeding.")

    output = run_solo()

    print("Solo Results Summary")
    print("-" * 40)
    for key, val in output.items():
        if key == "ueir_result":
            print(f"  ueir_result    : dict({len(val)} keys)")
        elif isinstance(val, (list, dict)):
            print(f"  {key:<14s}   : {type(val).__name__}({len(val)} items)")
        else:
            print(f"  {key:<14s}   : {val}")
    print()
    print("Done. Submit as COEN803_Solo_[YourName].zip")
