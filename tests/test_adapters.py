"""
Tests for SignalVector schema and ProductivityAdapter.observe() contract.
Real components only — no mocks, no patches.
app_detector is set via constructor injection (a lambda) for determinism; this is
dependency injection, not mocking.
"""
import time

import pytest

from apex.adapters.base import SignalAdapter, SignalVector
from apex.adapters.productivity import ProductivityAdapter


# ── SignalVector schema ──────────────────────────────────────────────────────

class TestSignalVectorSchema:
    def test_all_fields_present(self):
        sv = SignalVector(
            source_id="abc",
            content_hash="def",
            activity_type="writing",
            velocity_metric=0.5,
            temporal_proximity=0.3,
            urgency_flag=False,
        )
        assert sv.source_id == "abc"
        assert sv.content_hash == "def"
        assert sv.activity_type == "writing"
        assert sv.velocity_metric == 0.5
        assert sv.temporal_proximity == 0.3
        assert sv.urgency_flag is False

    def test_urgency_flag_defaults_false(self):
        sv = SignalVector(
            source_id="x",
            content_hash="y",
            activity_type="idle",
            velocity_metric=0.0,
            temporal_proximity=0.0,
        )
        assert sv.urgency_flag is False

    def test_velocity_metric_is_float(self):
        sv = SignalVector("a", "b", "writing", 1, 0.0)
        assert isinstance(sv.velocity_metric, (int, float))

    def test_source_id_is_str(self):
        sv = SignalVector("source", "hash", "reading", 0.1, 0.5)
        assert isinstance(sv.source_id, str)

    def test_urgency_flag_is_bool(self):
        sv = SignalVector("a", "b", "anomaly_event", 0.9, 0.0, True)
        assert isinstance(sv.urgency_flag, bool)
        assert sv.urgency_flag is True


# ── SignalAdapter ABC ────────────────────────────────────────────────────────

class TestSignalAdapterABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            SignalAdapter()  # type: ignore

    def test_concrete_subclass_must_implement_observe(self):
        class Incomplete(SignalAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass_is_instantiable(self):
        class Minimal(SignalAdapter):
            def observe(self) -> SignalVector:
                return SignalVector("s", "h", "idle", 0.0, 0.0)

        sv = Minimal().observe()
        assert isinstance(sv, SignalVector)


# ── ProductivityAdapter ──────────────────────────────────────────────────────

class TestProductivityAdapter:
    """
    Constructor injection (app_detector=lambda) is used for determinism.
    No mocking framework is involved — the lambda is a real callable.
    """

    def _make_adapter(self, app_name: str = "Xcode", last_change_delta: float = 2.0):
        adapter = ProductivityAdapter(
            watch_path="/tmp",
            app_detector=lambda: app_name,
        )
        adapter._last_change_time = time.time() - last_change_delta
        return adapter

    def test_observe_returns_signal_vector(self):
        sv = self._make_adapter().observe()
        assert isinstance(sv, SignalVector)

    def test_source_id_is_nonempty_string(self):
        sv = self._make_adapter().observe()
        assert isinstance(sv.source_id, str) and len(sv.source_id) > 0

    def test_source_id_is_deterministic_for_same_app(self):
        adapter = self._make_adapter(app_name="Safari")
        assert adapter.observe().source_id == adapter.observe().source_id

    def test_source_id_changes_with_different_app(self):
        sv1 = self._make_adapter(app_name="Xcode").observe()
        sv2 = self._make_adapter(app_name="Safari").observe()
        assert sv1.source_id != sv2.source_id

    def test_velocity_metric_bounded_zero_to_one(self):
        sv = self._make_adapter(last_change_delta=2.0).observe()
        assert 0.0 <= sv.velocity_metric <= 1.0

    def test_recent_change_gives_high_velocity(self):
        sv = self._make_adapter(last_change_delta=0.5).observe()
        assert sv.velocity_metric >= 0.7

    def test_stale_change_gives_low_velocity(self):
        sv = self._make_adapter(last_change_delta=60.0).observe()
        assert sv.velocity_metric <= 0.2

    def test_activity_type_is_known_string(self):
        sv = self._make_adapter(app_name="Xcode").observe()
        assert sv.activity_type in {"writing", "debugging", "reading", "idle"}

    def test_xcode_maps_to_debugging(self):
        sv = self._make_adapter(app_name="Xcode").observe()
        assert sv.activity_type == "debugging"

    def test_pages_maps_to_writing(self):
        sv = self._make_adapter(app_name="Pages").observe()
        assert sv.activity_type == "writing"

    def test_safari_maps_to_reading(self):
        sv = self._make_adapter(app_name="Safari").observe()
        assert sv.activity_type == "reading"

    def test_unknown_app_idle_when_velocity_low(self):
        # No recent file changes → velocity near 0 → idle (no file-change override)
        sv = self._make_adapter(app_name="SomeObscureApp", last_change_delta=30.0).observe()
        assert sv.activity_type == "idle"

    def test_unknown_app_writing_when_velocity_high(self):
        # Recent file changes → velocity high → override to "writing"
        # This handles eval/demo scenarios where Terminal edits files in the vault.
        sv = self._make_adapter(app_name="SomeObscureApp", last_change_delta=2.0).observe()
        assert sv.activity_type == "writing"

    def test_urgency_flag_false_for_productivity_domain(self):
        sv = self._make_adapter().observe()
        assert sv.urgency_flag is False

    def test_content_hash_is_hex_string(self):
        sv = self._make_adapter().observe()
        # Must be a hex digest of fixed length — never raw file content
        assert len(sv.content_hash) == 16
        int(sv.content_hash, 16)  # raises ValueError if not valid hex

    def test_real_macos_app_detector_returns_string(self):
        """Smoke test: real osascript app detection returns a non-empty string."""
        adapter = ProductivityAdapter(watch_path="/tmp")
        sv = adapter.observe()
        assert isinstance(sv.source_id, str) and len(sv.source_id) > 0
        assert isinstance(sv, SignalVector)
