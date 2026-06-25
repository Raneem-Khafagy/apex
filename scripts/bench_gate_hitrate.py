"""
Benchmark: Heuristic Gate Hit Rate
====================================
Simulates N=500 mixed signals from all three domain adapters
and measures the gate hit rate by activity_type × velocity bucket.
No Ollama required.

Output: JSON written to ict_express/results/gate_hitrate.json
Usage:  uv run python scripts/bench_gate_hitrate.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex.adapters.base import SignalVector
from apex.adapters.factory import FactoryAdapter
from apex.adapters.productivity import ProductivityAdapter
from apex.adapters.research import ResearchAdapter
from apex.inference.heuristic_gate import HeuristicGate

N_TOTAL = 500
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "ict_express", "results"
)

VELOCITY_THRESHOLD = 0.6
MIN_VELOCITY = 0.3


def _make_vector_table() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    table = {}
    for label in ("writing_document", "debugging_python", "reading_reference"):
        v = rng.standard_normal(384).astype(np.float32)
        table[label] = v / np.linalg.norm(v)
    return table


def _synthetic_signals(rng: np.random.Generator, n: int) -> list[SignalVector]:
    """Generate a realistic mix of signals spanning all three domains."""
    signals = []

    # Productivity domain signals
    prod_activities = [
        ("writing",   0.3, 1.0),   # (activity, vel_min, vel_max)
        ("debugging", 0.3, 1.0),
        ("reading",   0.3, 1.0),
    ]
    # Factory domain signals (includes anomaly_response which gate misses)
    factory_activities = [
        ("anomaly_response", 0.5, 1.0),
        ("maintenance_check", 0.3, 0.8),
        ("sensor_reading", 0.3, 0.7),
    ]
    # Research domain signals
    research_activities = [
        ("reading",   0.3, 0.9),
        ("writing",   0.4, 0.9),
        ("reviewing", 0.3, 0.7),
    ]

    # Equal split across domains
    per_domain = n // 3
    extra = n - per_domain * 3

    def _make_signals(activities, count):
        result = []
        for _ in range(count):
            act, vmin, vmax = activities[rng.integers(0, len(activities))]
            vel = float(rng.uniform(vmin, vmax))
            result.append(SignalVector(
                source_id="bench",
                content_hash="bench",
                activity_type=act,
                velocity_metric=vel,
                temporal_proximity=float(rng.uniform(0.1, 0.9)),
                urgency_flag=False,
            ))
        return result

    signals.extend(_make_signals(prod_activities, per_domain))
    signals.extend(_make_signals(factory_activities, per_domain))
    signals.extend(_make_signals(research_activities, per_domain + extra))

    rng.shuffle(signals)
    return signals


def main() -> None:
    rng = np.random.default_rng(2024)
    gate = HeuristicGate(vector_table=_make_vector_table())
    signals = _synthetic_signals(rng, N_TOTAL)

    # Counters keyed by (activity_type, velocity_bucket): [hits, total]
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    total_hits = 0
    total_signals = 0

    for sig in signals:
        if sig.velocity_metric < MIN_VELOCITY:
            continue  # suppressed as idle

        vbucket = "high" if sig.velocity_metric >= VELOCITY_THRESHOLD else "low"
        key = (sig.activity_type, vbucket)
        result = gate.match(sig)

        counts[key][1] += 1
        total_signals += 1
        if result is not None:
            counts[key][0] += 1
            total_hits += 1

    # Build per-activity breakdown (merge high/low into one row each)
    activity_stats: dict[str, dict] = defaultdict(lambda: {"hits": 0, "total": 0, "high_hits": 0, "high_total": 0, "low_hits": 0, "low_total": 0})
    for (act, vbucket), (hits, total) in counts.items():
        activity_stats[act]["hits"] += hits
        activity_stats[act]["total"] += total
        activity_stats[act][f"{vbucket}_hits"] += hits
        activity_stats[act][f"{vbucket}_total"] += total

    rows = {}
    for act, s in activity_stats.items():
        rows[act] = {
            "hit_rate_pct": round(s["hits"] / s["total"] * 100, 1) if s["total"] > 0 else None,
            "high_vel_hit_rate_pct": round(s["high_hits"] / s["high_total"] * 100, 1) if s["high_total"] > 0 else None,
            "low_vel_hit_rate_pct": round(s["low_hits"] / s["low_total"] * 100, 1) if s["low_total"] > 0 else None,
            "total_signals": s["total"],
        }

    overall_hit_rate = round(total_hits / total_signals * 100, 1) if total_signals > 0 else 0.0

    result = {
        "n_total": N_TOTAL,
        "n_evaluated": total_signals,
        "total_hits": total_hits,
        "overall_hit_rate_pct": overall_hit_rate,
        "by_activity": rows,
    }

    print(f"Gate hit rate (N={total_signals} evaluated from {N_TOTAL} total):")
    print(f"  Overall: {overall_hit_rate:.1f}%")
    print()
    print(f"  {'Activity':<22} {'Hit rate':>10}  {'High-vel':>10}  {'Low-vel':>10}  {'N':>6}")
    print(f"  {'-'*22} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
    for act in sorted(rows.keys()):
        r = rows[act]
        hr = f"{r['hit_rate_pct']:.1f}%" if r["hit_rate_pct"] is not None else "---"
        hv = f"{r['high_vel_hit_rate_pct']:.1f}%" if r["high_vel_hit_rate_pct"] is not None else "---"
        lv = f"{r['low_vel_hit_rate_pct']:.1f}%" if r["low_vel_hit_rate_pct"] is not None else "---"
        print(f"  {act:<22} {hr:>10}  {hv:>10}  {lv:>10}  {r['total_signals']:>6}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "gate_hitrate.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
