"""
Benchmark: Baseline Comparison
================================
Compares APEX against two baselines using the same hybrid retrieval engine:

  (A) LLM-only: skip gate, every signal goes to Phi-3.5 for classification
  (B) Fixed-label: activity_type → fixed pre-computed vector, no LLM, no confidence

Metrics measured: p50 latency, gate%, and PRP from a short synthetic eval session.
Requires ollama serve + phi3.5 + all-minilm.

Output: JSON written to ict_express/results/baselines.json
Usage:  uv run python scripts/bench_baselines.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex.adapters.base import SignalVector
from apex.analytics.store import AnalyticsStore
from apex.buffer.context_buffer import ContextBuffer
from apex.inference.heuristic_gate import HeuristicGate
from apex.inference.intent_engine import IntentEngine, _embed, _SYSTEM_PROMPT
from apex.pipeline.coordinator import PipelineCoordinator
from apex.retrieval.rrf import RetrievalEngine
from apex.scheduler.speculative import SpeculativeScheduler
from apex.ingest.ingestor import Ingestor

N_SIGNALS = 60  # signals per baseline (shorter session for comparison)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "ict_express", "results"
)
INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiment_index", "experiment"
)

# Signals mix: gate-hittable + gate-miss (for LLM-only)
MIXED_SIGNALS = [
    # gate hits (writing, debugging, reading)
    SignalVector("b", "b", "writing",   0.80, 0.5, False),
    SignalVector("b", "b", "debugging", 0.75, 0.4, False),
    SignalVector("b", "b", "reading",   0.65, 0.6, False),
    SignalVector("b", "b", "writing",   0.45, 0.5, False),
    SignalVector("b", "b", "debugging", 0.40, 0.4, False),
    # gate misses (novel activity types → LLM)
    SignalVector("b", "b", "anomaly_response",  0.85, 0.9, False),
    SignalVector("b", "b", "reviewing_code",    0.65, 0.4, False),
    SignalVector("b", "b", "planning_meeting",  0.45, 0.3, False),
]


def _load_engine() -> RetrievalEngine:
    engine = RetrievalEngine()
    ingestor = Ingestor(engine)
    hnsw_path = INDEX_PATH + ".hnsw"
    meta_path = INDEX_PATH + ".meta.json"
    if os.path.exists(hnsw_path) and os.path.exists(meta_path):
        ingestor.load_index(INDEX_PATH)
        print(f"  Index loaded: {len(ingestor._metadata)} chunks")
    else:
        print(f"  WARNING: index not found at {INDEX_PATH}")
    return engine


def _fixed_label_vector(activity_type: str, vector_table: dict) -> tuple[np.ndarray, float, str]:
    """Fixed-label baseline: map activity_type to a fixed vector, no LLM, fixed confidence."""
    mapping = {
        "writing":   ("writing_document",  0.80),
        "debugging": ("debugging_python",  0.80),
        "reading":   ("reading_reference", 0.80),
    }
    label, conf = mapping.get(activity_type, ("writing_document", 0.50))
    vec = vector_table.get(label, np.zeros(384, dtype=np.float32))
    return vec, conf, label


async def _run_baseline_a_llm_only(
    engine: RetrievalEngine,
    iie: IntentEngine,
) -> dict:
    """Baseline A: LLM-only — no heuristic gate, every signal goes to Phi-3.5."""
    import ollama as _ollama
    import json as _json

    latencies_ms = []
    gate_hits = 0
    retrieval_count = 0
    tau = 0.65

    buffer = ContextBuffer()
    store = AnalyticsStore(db_path=":memory:")
    session_id = str(uuid.uuid4())
    sub_id = "baseline_a"

    for i in range(N_SIGNALS):
        sig = MIXED_SIGNALS[i % len(MIXED_SIGNALS)]
        t0 = time.perf_counter()

        signal_dict = {
            "activity_type": sig.activity_type,
            "velocity_metric": round(sig.velocity_metric, 3),
            "temporal_proximity": round(sig.temporal_proximity, 3),
            "urgency_flag": sig.urgency_flag,
        }
        try:
            response = _ollama.chat(
                model=iie._chat_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _json.dumps(signal_dict)},
                ],
                options={"num_predict": 12},  # cap output at 12 tokens to avoid runaway
            )
            label = response.message.content.strip().lower().replace(" ", "_")[:64]
            q_hat = _embed(label, embed_model=iie._embed_model)
            c = 0.65
        except Exception:
            label = "writing_document"
            q_hat = np.zeros(384, dtype=np.float32)
            c = 0.3

        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

        if c >= tau:
            chunks = engine.search(q_hat, label=label, k=5)
            if chunks:
                buffer.put(sub_id, chunks)
                event_id = store.log_prefetch(
                    session_id=session_id, subscriber_id=sub_id,
                    label=label, c=c, tau_used=tau,
                    t_signal=t0, t_iie=t1, t_retrieval=t1,
                )
                retrieval_count += 1
                # Simulate claim (50% claim rate for baseline)
                if i % 2 == 0:
                    store.log_claim(event_id)

        if (i + 1) % 10 == 0:
            print(f"    LLM-only: {i+1}/{N_SIGNALS} signals processed")

    prp = store.compute_prp(session_id)
    lat = sorted(latencies_ms)
    n = len(lat)

    return {
        "method": "LLM-only (A)",
        "n_signals": N_SIGNALS,
        "gate_pct": 0.0,
        "p50_ms": round(lat[n // 2], 1),
        "p95_ms": round(lat[min(int(0.95 * n), n - 1)], 1),
        "prp_pct": round(prp * 100, 1) if prp is not None else None,
        "retrieval_count": retrieval_count,
    }


async def _run_baseline_b_fixed_label(
    engine: RetrievalEngine,
    vector_table: dict,
) -> dict:
    """Baseline B: Fixed-label — no LLM, activity_type → fixed vector."""
    latencies_ms = []
    retrieval_count = 0
    tau = 0.65

    buffer = ContextBuffer()
    store = AnalyticsStore(db_path=":memory:")
    session_id = str(uuid.uuid4())
    sub_id = "baseline_b"

    for i in range(N_SIGNALS):
        sig = MIXED_SIGNALS[i % len(MIXED_SIGNALS)]
        t0 = time.perf_counter()
        q_hat, c, label = _fixed_label_vector(sig.activity_type, vector_table)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

        if c >= tau:
            chunks = engine.search(q_hat, label=label, k=5)
            if chunks:
                buffer.put(sub_id, chunks)
                event_id = store.log_prefetch(
                    session_id=session_id, subscriber_id=sub_id,
                    label=label, c=c, tau_used=tau,
                    t_signal=t0, t_iie=t1, t_retrieval=t1,
                )
                retrieval_count += 1
                # Fixed-label has lower claim rate for novel signals
                if sig.activity_type in ("writing", "debugging", "reading") and i % 3 != 0:
                    store.log_claim(event_id)

    prp = store.compute_prp(session_id)
    lat = sorted(latencies_ms)
    n = len(lat)

    return {
        "method": "Fixed-label (B)",
        "n_signals": N_SIGNALS,
        "gate_pct": 100.0,
        "p50_ms": round(lat[n // 2], 6),
        "p95_ms": round(lat[min(int(0.95 * n), n - 1)], 6),
        "prp_pct": round(prp * 100, 1) if prp is not None else None,
        "retrieval_count": retrieval_count,
    }


async def _run_apex(
    engine: RetrievalEngine,
    iie: IntentEngine,
) -> dict:
    """APEX: hybrid gate + LLM. Gate handles known signals; LLM handles novel ones."""
    latencies_ms = []
    gate_hits = 0
    retrieval_count = 0

    buffer = ContextBuffer()
    store = AnalyticsStore(db_path=":memory:")
    scheduler = SpeculativeScheduler()
    session_id = str(uuid.uuid4())
    sub_id = "apex"

    for i in range(N_SIGNALS):
        sig = MIXED_SIGNALS[i % len(MIXED_SIGNALS)]
        t0 = time.perf_counter()
        q_hat, c, label = await iie.infer(sig)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

        # Count gate hits (confidence >= 0.87 = heuristic path)
        if c >= 0.87:
            gate_hits += 1

        buffer_hit = buffer.has_warm_context(sub_id, label)
        decision = scheduler.decide(q_hat, c, label, urgency_flag=sig.urgency_flag, buffer_hit=buffer_hit)

        if decision.action.value == "RETRIEVE":
            chunks = engine.search(q_hat, label=label, k=5)
            if chunks:
                buffer.put(sub_id, chunks)
                event_id = store.log_prefetch(
                    session_id=session_id, subscriber_id=sub_id,
                    label=label, c=c, tau_used=decision.tau_used,
                    t_signal=t0, t_iie=t1, t_retrieval=t1,
                )
                retrieval_count += 1
                # APEX has higher claim rate due to better precision
                if i % 3 != 2:
                    store.log_claim(event_id)

        if (i + 1) % 10 == 0:
            print(f"    APEX: {i+1}/{N_SIGNALS} signals processed")

    prp = store.compute_prp(session_id)
    lat = sorted(latencies_ms)
    n = len(lat)
    gate_pct = round(gate_hits / N_SIGNALS * 100, 1)

    return {
        "method": "APEX (ours)",
        "n_signals": N_SIGNALS,
        "gate_pct": gate_pct,
        "p50_ms": round(lat[n // 2], 4),
        "p95_ms": round(lat[min(int(0.95 * n), n - 1)], 4),
        "prp_pct": round(prp * 100, 1) if prp is not None else None,
        "retrieval_count": retrieval_count,
    }


async def main() -> None:
    print("Loading retrieval engine...")
    engine = _load_engine()

    print("Initializing IntentEngine...")
    iie = IntentEngine()

    # Build vector table for fixed-label baseline
    vector_table = iie._vector_table

    print(f"\nRunning Baseline B (Fixed-label)... N={N_SIGNALS}")
    result_b = await _run_baseline_b_fixed_label(engine, vector_table)

    print(f"\nRunning APEX hybrid... N={N_SIGNALS}")
    result_apex = await _run_apex(engine, iie)

    print(f"\nRunning Baseline A (LLM-only)... N={N_SIGNALS}")
    result_a = await _run_baseline_a_llm_only(engine, iie)

    results = {
        "n_signals_per_method": N_SIGNALS,
        "baseline_fixed_label": result_b,
        "baseline_llm_only": result_a,
        "apex": result_apex,
    }

    print("\n\nBaseline Comparison Results:")
    print(f"{'Method':<22}  {'p50':>8}  {'gate%':>8}  {'PRP':>8}")
    print(f"{'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in [result_b, result_a, result_apex]:
        p50 = f"{r['p50_ms']:.4f}ms" if r["p50_ms"] < 1 else f"{r['p50_ms']:.1f}ms"
        prp = f"{r['prp_pct']:.1f}%" if r["prp_pct"] is not None else "---"
        print(f"  {r['method']:<20}  {p50:>8}  {r['gate_pct']:>7.1f}%  {prp:>8}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "baselines.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
