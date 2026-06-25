"""
Phase 0 — Simulate a factory sensor writing state snapshots to a JSON file.

FactoryAdapter reads this file via _read_state(). This script lets you drive
the adapter through all three activity types (normal_operation,
maintenance_window, anomaly_event) without real MQTT hardware.

Usage:
    # Normal background noise
    uv run python scripts/simulate_factory_sensor.py

    # Ramp to an anomaly (triggers urgency_flag)
    uv run python scripts/simulate_factory_sensor.py --mode anomaly

    # Simulate approaching maintenance window
    uv run python scripts/simulate_factory_sensor.py --mode maintenance

Output file (default): /tmp/apex_factory_state.json
    Change with: --out /path/to/state.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def write_state(path: Path, deviation: float, maintenance: float,
                sensor_id: str, machine_id: str) -> None:
    state = {
        "deviation":           round(deviation, 4),
        "time_to_maintenance": round(maintenance, 4),
        "sensor_id":           sensor_id,
        "machine_id":          machine_id,
    }
    path.write_text(json.dumps(state, indent=2))


def simulate(mode: str, output: Path, interval: float) -> None:
    sensor_id  = "pressure_sensor_01"
    machine_id = "cnc_lathe_03"
    t = 0.0

    print(f"Writing sensor state → {output}  (mode={mode}, interval={interval}s)")
    print("Ctrl-C to stop.\n")

    try:
        while True:
            noise = random.gauss(0, 0.3)

            if mode == "normal":
                deviation = noise
                maintenance = 0.2 + 0.1 * math.sin(t / 30)

            elif mode == "anomaly":
                # Ramp from 0 → 4σ over 60 s, then hold
                ramp = min(4.0, t / 15.0)
                deviation = ramp + noise
                maintenance = 0.3

            elif mode == "maintenance":
                deviation = noise
                # maintenance_proximity rises from 0 → 1 over 120 s
                maintenance = min(1.0, t / 120.0)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            write_state(output, deviation, maintenance, sensor_id, machine_id)

            status = (
                f"  t={t:6.1f}s  dev={deviation:+.3f}σ  "
                f"maint={maintenance:.3f}  "
                f"{'⚠ ANOMALY' if abs(deviation) >= 3.0 else '      OK'}"
            )
            print(status, end="\r")

            time.sleep(interval)
            t += interval

    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate factory sensor state")
    parser.add_argument(
        "--mode", choices=["normal", "anomaly", "maintenance"], default="normal",
        help="Simulation mode (default: normal)",
    )
    parser.add_argument(
        "--out", default="/tmp/apex_factory_state.json",
        help="Output JSON file path (default: /tmp/apex_factory_state.json)",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Update interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()
    simulate(args.mode, Path(args.out), args.interval)


if __name__ == "__main__":
    main()
