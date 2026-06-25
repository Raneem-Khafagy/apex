"""
Benchmark: Heuristic Gate Latency
==================================
Measures p50/p95/p99 latency of the HeuristicGate.match() call over N=1000 trials.
No Ollama required — uses pre-built numpy vectors.

Output: JSON written to ict_express/results/gate_latency.json
Usage:  uv run python scripts/bench_gate_latency.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# Make sure apex package is importable when run from apex/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex.adapters.base import SignalVector
from apex.inference.heuristic_gate import HeuristicGate

N_TRIALS = 1000
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # apex/
    "..", "ict_express", "results"
)


def _unit_vec(dim: int = 384) -> np.ndarray:
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_vector_table() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    table = {}
    for label in ("writing_document", "debugging_python", "reading_reference"):
        v = rng.standard_normal(384).astype(np.float32)
        table[label] = v / np.linalg.norm(v)
    return table


def _bench_signal(
    gate: HeuristicGate,
    activity_type: str,
    velocity: float,
    n: int,
) -> list[float]:
    signal = SignalVector(
        source_id="bench",
        content_hash="bench",
        activity_type=activity_type,
        velocity_metric=velocity,
        temporal_proximity=0.5,
        urgency_flag=False,
    )
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        gate.match(signal)
        times.append(time.perf_counter_ns() - t0)
    return times


def main() -> None:
    gate = HeuristicGate(vector_table=_make_vector_table())

    # Warm-up: 100 calls
    for _ in range(100):
        gate.match(SignalVector(
            source_id="w", content_hash="w",
            activity_type="writing", velocity_metric=0.8,
            temporal_proximity=0.5, urgency_flag=False,
        ))

    # Collect latencies across representative signals
    all_times: list[float] = []
    combos = [
        ("writing",   0.75),
        ("debugging", 0.80),
        ("reading",   0.65),
        ("writing",   0.45),
        ("debugging", 0.40),
        ("reading",   0.35),
    ]
    per_combo = N_TRIALS // len(combos)
    for activity, vel in combos:
        all_times.extend(_bench_signal(gate, activity, vel, per_combo))

    all_times.sort()
    n = len(all_times)

    def pct(p: float) -> float:
        return round(all_times[int(p * n / 100)] / 1_000_000, 4)  # ns → ms

    result = {
        "n_trials": n,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": round(all_times[0] / 1_000_000, 4),
        "max_ms": round(all_times[-1] / 1_000_000, 4),
        "mean_ms": round(sum(all_times) / n / 1_000_000, 4),
    }

    print(f"Gate latency (N={n}):")
    print(f"  p50  = {result['p50_ms']:.4f} ms")
    print(f"  p95  = {result['p95_ms']:.4f} ms")
    print(f"  p99  = {result['p99_ms']:.4f} ms")
    print(f"  mean = {result['mean_ms']:.4f} ms")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "gate_latency.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
