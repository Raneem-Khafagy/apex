"""
Benchmark: LLM Path Latency (phi3.5 via Ollama)
=================================================
Measures p50/p95 latency of the full LLM inference path (Ollama chat + embed)
over N=20 warm trials. Requires ollama serve + phi3.5 + all-minilm pulled.

Output: JSON written to ict_express/results/llm_latency.json
Usage:  uv run python scripts/bench_llm_latency.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex.adapters.base import SignalVector
from apex.inference.intent_engine import IntentEngine

N_WARMUP = 2
N_TRIALS = 20
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "ict_express", "results"
)

# Signals that will always miss the heuristic gate → go to LLM
_LLM_SIGNALS = [
    SignalVector(
        source_id="bench", content_hash="bench",
        activity_type="anomaly_response", velocity_metric=0.85,
        temporal_proximity=0.9, urgency_flag=False,
    ),
    SignalVector(
        source_id="bench", content_hash="bench",
        activity_type="maintenance_check", velocity_metric=0.55,
        temporal_proximity=0.6, urgency_flag=False,
    ),
    SignalVector(
        source_id="bench", content_hash="bench",
        activity_type="reviewing_code", velocity_metric=0.62,
        temporal_proximity=0.4, urgency_flag=False,
    ),
    SignalVector(
        source_id="bench", content_hash="bench",
        activity_type="planning_meeting", velocity_metric=0.45,
        temporal_proximity=0.3, urgency_flag=False,
    ),
]


async def _bench_llm_path(engine: IntentEngine) -> tuple[list[float], list[float]]:
    """
    Returns (chat_times, full_times) in seconds.
    chat_times: only the Ollama chat call duration.
    full_times: full _llm_infer() including embed.
    """
    chat_times: list[float] = []
    full_times: list[float] = []

    import ollama as _ollama
    import json as _json
    from apex.inference.intent_engine import _embed, _SYSTEM_PROMPT

    for i in range(N_WARMUP + N_TRIALS):
        sig = _LLM_SIGNALS[i % len(_LLM_SIGNALS)]
        signal_dict = {
            "activity_type": sig.activity_type,
            "velocity_metric": round(sig.velocity_metric, 3),
            "temporal_proximity": round(sig.temporal_proximity, 3),
            "urgency_flag": sig.urgency_flag,
        }

        t_full_start = time.perf_counter()

        # -- chat call --
        t_chat_start = time.perf_counter()
        try:
            response = _ollama.chat(
                model=engine._chat_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _json.dumps(signal_dict)},
                ],
            )
            label = response.message.content.strip().lower().replace(" ", "_")[:64]
        except Exception as e:
            print(f"Chat call failed: {e}")
            continue
        t_chat_end = time.perf_counter()

        # -- embed call --
        try:
            _embed(label, embed_model=engine._embed_model)
        except Exception as e:
            print(f"Embed call failed: {e}")
            continue

        t_full_end = time.perf_counter()

        if i >= N_WARMUP:
            chat_times.append(t_chat_end - t_chat_start)
            full_times.append(t_full_end - t_full_start)

        print(f"  trial {i-N_WARMUP+1:2d}/{N_TRIALS}: chat={1000*(t_chat_end-t_chat_start):.1f}ms  "
              f"full={1000*(t_full_end-t_full_start):.1f}ms  label='{label}'")

    return chat_times, full_times


async def main() -> None:
    print("Initializing IntentEngine (building vector table, warming up Ollama)...")
    engine = IntentEngine()

    print(f"\nRunning {N_WARMUP} warm-up + {N_TRIALS} timed LLM path trials...")
    chat_times, full_times = await _bench_llm_path(engine)

    if not full_times:
        print("ERROR: No successful trials — is ollama running with phi3.5 + all-minilm?")
        sys.exit(1)

    def stats(times: list[float], unit_ms: float = 1000) -> dict:
        ts = sorted(t * unit_ms for t in times)
        n = len(ts)
        return {
            "n": n,
            "p50_ms": round(ts[n // 2], 1),
            "p95_ms": round(ts[min(int(0.95 * n), n - 1)], 1),
            "p99_ms": round(ts[min(int(0.99 * n), n - 1)], 1),
            "mean_ms": round(sum(ts) / n, 1),
            "min_ms": round(ts[0], 1),
            "max_ms": round(ts[-1], 1),
        }

    chat_stats = stats(chat_times)
    full_stats = stats(full_times)

    result = {
        "n_warmup": N_WARMUP,
        "n_trials": N_TRIALS,
        "chat_only": chat_stats,
        "full_llm_path": full_stats,  # chat + embed
    }

    print(f"\nLLM path latency (N={N_TRIALS} warm trials):")
    print(f"  Chat call:  p50={chat_stats['p50_ms']:.1f}ms  p95={chat_stats['p95_ms']:.1f}ms  p99={chat_stats['p99_ms']:.1f}ms")
    print(f"  Full path:  p50={full_stats['p50_ms']:.1f}ms  p95={full_stats['p95_ms']:.1f}ms  p99={full_stats['p99_ms']:.1f}ms")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "llm_latency.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
