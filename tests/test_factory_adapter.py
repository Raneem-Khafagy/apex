"""
Tests for FactoryAdapter.

Real FactoryAdapter with a real temp state file.
No mocks — sensor_state_path points to a real JSON file written by the test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from apex.adapters.base import SignalAdapter, SignalVector
from apex.adapters.factory import ANOMALY_THRESHOLD, FactoryAdapter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_state(path: Path, deviation: float, maintenance: float,
                 sensor_id: str = "s01", machine_id: str = "m01") -> None:
    path.write_text(json.dumps({
        "deviation":           deviation,
        "time_to_maintenance": maintenance,
        "sensor_id":           sensor_id,
        "machine_id":          machine_id,
    }))


# ── Contract ──────────────────────────────────────────────────────────────────

class TestFactoryAdapterContract:
    def test_is_signal_adapter(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, 0.0, 0.3)
        adapter = FactoryAdapter(sensor_state_path=str(f))
        assert isinstance(adapter, SignalAdapter)

    def test_observe_returns_signal_vector(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, 0.5, 0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert isinstance(sv, SignalVector)

    def test_all_fields_present(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, 0.5, 0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.source_id
        assert sv.content_hash
        assert sv.activity_type
        assert isinstance(sv.velocity_metric, float)
        assert isinstance(sv.temporal_proximity, float)
        assert isinstance(sv.urgency_flag, bool)


# ── Activity classification ────────────────────────────────────────────────────

class TestActivityClassification:
    def test_normal_operation_when_low_deviation(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=0.5, maintenance=0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.activity_type == "normal_operation"

    def test_maintenance_window_when_proximity_high(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=0.0, maintenance=0.9)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.activity_type == "maintenance_window"

    def test_anomaly_event_when_deviation_exceeds_threshold(self, tmp_path):
        f = tmp_path / "state.json"
        # deviation = ANOMALY_THRESHOLD * baseline_sigma (default 1.0)
        _write_state(f, deviation=ANOMALY_THRESHOLD, maintenance=0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.activity_type == "anomaly_event"

    def test_anomaly_takes_priority_over_maintenance(self, tmp_path):
        """Anomaly + maintenance_window → anomaly_event wins."""
        f = tmp_path / "state.json"
        _write_state(f, deviation=ANOMALY_THRESHOLD, maintenance=0.95)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.activity_type == "anomaly_event"

    def test_negative_deviation_also_triggers_anomaly(self, tmp_path):
        """Deviation is checked by abs() — negative excursions are anomalies too."""
        f = tmp_path / "state.json"
        _write_state(f, deviation=-ANOMALY_THRESHOLD, maintenance=0.2)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.activity_type == "anomaly_event"


# ── urgency_flag ─────────────────────────────────────────────────────────────

class TestUrgencyFlag:
    def test_urgency_false_for_normal(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=0.5, maintenance=0.2)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.urgency_flag is False

    def test_urgency_true_at_threshold(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=ANOMALY_THRESHOLD, maintenance=0.2)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.urgency_flag is True

    def test_urgency_false_just_below_threshold(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=ANOMALY_THRESHOLD - 0.01, maintenance=0.2)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.urgency_flag is False

    def test_urgency_respects_baseline_sigma(self, tmp_path):
        """With baseline_sigma=2.0, threshold doubles."""
        f = tmp_path / "state.json"
        # deviation=5.0 < ANOMALY_THRESHOLD(3) * sigma(2) = 6.0 → no urgency
        _write_state(f, deviation=5.0, maintenance=0.2)
        sv = FactoryAdapter(sensor_state_path=str(f), baseline_sigma=2.0).observe()
        assert sv.urgency_flag is False

    def test_urgency_true_above_scaled_threshold(self, tmp_path):
        f = tmp_path / "state.json"
        # deviation=6.1 > 3.0 * 2.0 = 6.0 → urgency
        _write_state(f, deviation=6.1, maintenance=0.2)
        sv = FactoryAdapter(sensor_state_path=str(f), baseline_sigma=2.0).observe()
        assert sv.urgency_flag is True


# ── Velocity metric ───────────────────────────────────────────────────────────

class TestVelocityMetric:
    def test_velocity_zero_for_zero_deviation(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=0.0, maintenance=0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.velocity_metric == pytest.approx(0.0)

    def test_velocity_one_at_threshold(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=ANOMALY_THRESHOLD, maintenance=0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.velocity_metric == pytest.approx(1.0)

    def test_velocity_capped_at_one(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=ANOMALY_THRESHOLD * 10, maintenance=0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv.velocity_metric == pytest.approx(1.0)

    def test_velocity_bounded_zero_to_one(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=1.5, maintenance=0.3)
        sv = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert 0.0 <= sv.velocity_metric <= 1.0


# ── source_id determinism ─────────────────────────────────────────────────────

class TestSourceId:
    def test_source_id_is_deterministic_for_same_machine_sensor(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=0.5, maintenance=0.3,
                     sensor_id="s01", machine_id="m01")
        a = FactoryAdapter(sensor_state_path=str(f)).observe()
        b = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert a.source_id == b.source_id

    def test_source_id_differs_for_different_machine(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, 0.5, 0.3, sensor_id="s01", machine_id="m01")
        sv_a = FactoryAdapter(sensor_state_path=str(f)).observe()
        _write_state(f, 0.5, 0.3, sensor_id="s01", machine_id="m99")
        sv_b = FactoryAdapter(sensor_state_path=str(f)).observe()
        assert sv_a.source_id != sv_b.source_id


# ── Resilience ────────────────────────────────────────────────────────────────

class TestResilience:
    def test_missing_state_file_does_not_raise(self, tmp_path):
        adapter = FactoryAdapter(sensor_state_path=str(tmp_path / "missing.json"))
        sv = adapter.observe()   # must not raise
        assert isinstance(sv, SignalVector)

    def test_corrupted_json_falls_back_gracefully(self, tmp_path):
        f = tmp_path / "state.json"
        f.write_text("{ not valid json !!!")
        adapter = FactoryAdapter(sensor_state_path=str(f))
        sv = adapter.observe()
        assert isinstance(sv, SignalVector)

    def test_state_updates_are_reflected_on_next_observe(self, tmp_path):
        f = tmp_path / "state.json"
        _write_state(f, deviation=0.0, maintenance=0.2)
        adapter = FactoryAdapter(sensor_state_path=str(f))
        sv1 = adapter.observe()
        assert sv1.urgency_flag is False

        _write_state(f, deviation=ANOMALY_THRESHOLD, maintenance=0.2)
        sv2 = adapter.observe()
        assert sv2.urgency_flag is True
