"""
Factory / Industrial Signal Adapter — smart factory / IIoT domain.

Reads sensor telemetry from a local state file written by an MQTT bridge,
OPC-UA proxy, or simulation script. Never reads raw process data or PLC
program content — only deviation metrics and timing metadata.

Signal semantics
----------------
source_id          : SHA-256 of machine_id + sensor_id (identity hash)
content_hash       : SHA-256 of latest (deviation, maintenance_proximity)
                     snapshot — changes only when sensor state changes
activity_type      : "anomaly_event" | "maintenance_window" | "normal_operation"
velocity_metric    : normalized anomaly deviation ∈ [0, 1]
temporal_proximity : time-to-next-maintenance-window ∈ [0, 1] (1 = imminent)
urgency_flag       : True when deviation ≥ ANOMALY_THRESHOLD × baseline_sigma

The urgency_flag = True path is what makes APEX safety-critical-aware for the
factory domain. When an anomaly fires, the Speculative Retrieval Scheduler
forces τ → 0 and retrieves immediately — no waiting.

Privacy rule: this module never reads raw sensor data streams or PLC programs.
It reads only a JSON state snapshot produced by a separate MQTT/OPC-UA bridge.

Phase 0 (local integration): point sensor_state_path at a JSON file updated
by a simulation script (scripts/simulate_factory_sensor.py).
Phase 2+ (Jetson/hardware): point at the real MQTT bridge output.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from apex.adapters.base import SignalAdapter, SignalVector

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of standard deviations above baseline that triggers urgency_flag.
# At exactly this threshold, urgency_flag becomes True and τ → 0.
ANOMALY_THRESHOLD: float = 3.0

# Sentinel state used when the sensor file cannot be read.
_FALLBACK_STATE: dict = {
    "deviation": 0.0,
    "time_to_maintenance": 0.5,
    "sensor_id": "unknown",
    "machine_id": "unknown",
}


class FactoryAdapter(SignalAdapter):
    """
    Behavioral Signal Adapter for the smart factory / industrial domain.

    Parameters
    ----------
    sensor_state_path
        Path to a JSON file containing the latest sensor snapshot.
        Written by an external MQTT bridge, OPC-UA proxy, or simulation.
        Schema::

            {
                "deviation":            float,  # deviations from baseline (signed)
                "time_to_maintenance":  float,  # normalized [0, 1]; 1 = imminent
                "sensor_id":            str,    # e.g. "pressure_sensor_01"
                "machine_id":           str     # e.g. "cnc_lathe_03"
            }

    machine_id
        Fallback machine identifier used in source_id when the state file
        does not contain a machine_id field.
    baseline_sigma
        Baseline standard deviation of the sensor signal under normal
        operation. Controls the sensitivity of urgency_flag.
        Default 1.0 (deviation is already in σ units). Adjust if the
        sensor_state_path reports raw deviation in engineering units.
    """

    def __init__(
        self,
        sensor_state_path: str,
        machine_id: str = "machine_01",
        baseline_sigma: float = 1.0,
    ) -> None:
        self._state_path = Path(sensor_state_path)
        self._default_machine_id = machine_id
        self._baseline_sigma = max(baseline_sigma, 1e-6)  # guard division by zero
        self._last_state: dict = _FALLBACK_STATE.copy()
        logger.info(
            "FactoryAdapter: watching sensor state at '{}' (machine_id='{}')",
            self._state_path, machine_id,
        )

    # ── SignalAdapter contract ────────────────────────────────────────────────

    def observe(self) -> SignalVector:
        """
        Return a SignalVector snapshot of the current factory sensor state.
        Reads only the JSON state file — no raw sensor stream, no PLC content.
        """
        state = self._read_state()

        deviation: float = float(state.get("deviation", 0.0))
        maintenance_proximity: float = float(
            state.get("time_to_maintenance", 0.5)
        )
        sensor_id: str = str(state.get("sensor_id", "unknown"))
        machine_id: str = str(state.get("machine_id", self._default_machine_id))

        # velocity: normalized anomaly magnitude ∈ [0, 1]
        velocity = min(
            1.0,
            abs(deviation) / (ANOMALY_THRESHOLD * self._baseline_sigma),
        )

        # urgency_flag: True only when anomaly exceeds safety threshold
        urgency = abs(deviation) >= ANOMALY_THRESHOLD * self._baseline_sigma

        # source_id: deterministic identity hash for this machine + sensor pair
        source_id = hashlib.sha256(
            f"{machine_id}:{sensor_id}".encode()
        ).hexdigest()[:16]

        # content_hash: changes only when sensor state changes (not on every call)
        content_hash = hashlib.sha256(
            f"{deviation:.4f}:{maintenance_proximity:.4f}".encode()
        ).hexdigest()[:16]

        # activity_type: three mutually exclusive states
        if urgency:
            activity_type = "anomaly_event"
        elif maintenance_proximity >= 0.75:
            activity_type = "maintenance_window"
        else:
            activity_type = "normal_operation"

        logger.debug(
            "FactoryAdapter: machine='{}' sensor='{}' dev={:.3f} "
            "vel={:.3f} urgency={} activity='{}'",
            machine_id, sensor_id, deviation, velocity, urgency, activity_type,
        )

        return SignalVector(
            source_id=source_id,
            content_hash=content_hash,
            activity_type=activity_type,
            velocity_metric=velocity,
            temporal_proximity=maintenance_proximity,
            urgency_flag=urgency,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _read_state(self) -> dict:
        """
        Read the latest sensor state from the JSON file.
        Falls back to the last known good state on any I/O or parse error.
        Never raises — the pipeline must not crash on a missing sensor file.
        """
        try:
            raw = self._state_path.read_text(encoding="utf-8")
            state = json.loads(raw)
            self._last_state = state
            return state
        except FileNotFoundError:
            logger.warning(
                "FactoryAdapter: state file '{}' not found — using last known state",
                self._state_path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "FactoryAdapter: failed to read state file '{}': {} — using last known state",
                self._state_path, exc,
            )
        return self._last_state
