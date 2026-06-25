"""
Tests for SpeculativeScheduler.

All tests use real components — no mocks, no patches.
The scheduler is pure-Python with no I/O, so tests are synchronous and fast.
q̂ is a real numpy array (the scheduler must not touch it, but it must accept it).

Decision matrix covered:
  urgency_flag=True  → RETRIEVE unconditionally
  c >= τ, buffer miss → RETRIEVE
  c < τ             → WAIT
  buffer hit         → WAIT (even if c >= τ)
  battery saver      → τ raised to 0.80
  update_tau         → live threshold change
  τ boundary (c == τ) → RETRIEVE (>= not >)
"""
import numpy as np
import pytest

from apex.scheduler.speculative import (
    RetrievalAction,
    SchedulerDecision,
    SpeculativeScheduler,
)

EMBED_DIM = 384


def _vec() -> np.ndarray:
    """Real unit vector — same kind the IIE would produce."""
    rng = np.random.default_rng(42)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ── RetrievalAction enum ─────────────────────────────────────────────────────

class TestRetrievalAction:
    def test_retrieve_and_wait_exist(self):
        assert RetrievalAction.RETRIEVE
        assert RetrievalAction.WAIT

    def test_actions_are_distinct(self):
        assert RetrievalAction.RETRIEVE != RetrievalAction.WAIT


# ── SchedulerDecision dataclass ──────────────────────────────────────────────

class TestSchedulerDecision:
    def test_fields_present(self):
        d = SchedulerDecision(
            action=RetrievalAction.RETRIEVE,
            tau_used=0.65,
            reason="test",
        )
        assert d.action == RetrievalAction.RETRIEVE
        assert d.tau_used == 0.65
        assert d.reason == "test"

    def test_is_frozen(self):
        d = SchedulerDecision(RetrievalAction.WAIT, 0.65, "test")
        with pytest.raises((AttributeError, TypeError)):
            d.action = RetrievalAction.RETRIEVE  # type: ignore


# ── Default state ────────────────────────────────────────────────────────────

class TestSchedulerDefaults:
    def test_default_tau_is_0_65(self):
        s = SpeculativeScheduler()
        assert s.tau == 0.65

    def test_custom_tau_at_init(self):
        s = SpeculativeScheduler(tau=0.75)
        assert s.tau == 0.75

    def test_battery_saver_off_by_default(self):
        # With tau=0.65 and c=0.70, should RETRIEVE (not blocked by battery saver)
        s = SpeculativeScheduler()
        d = s.decide(_vec(), c=0.70, label="writing_document")
        assert d.action == RetrievalAction.RETRIEVE


# ── Core decision policy ─────────────────────────────────────────────────────

class TestDecisionPolicy:
    def setup_method(self):
        self.s = SpeculativeScheduler(tau=0.65)
        self.q = _vec()

    def test_retrieve_when_c_above_tau_and_buffer_miss(self):
        d = self.s.decide(self.q, c=0.80, label="debugging_python", buffer_hit=False)
        assert d.action == RetrievalAction.RETRIEVE

    def test_retrieve_when_c_exactly_at_tau(self):
        """c >= τ means exact equality must also fire retrieval."""
        d = self.s.decide(self.q, c=0.65, label="writing_document", buffer_hit=False)
        assert d.action == RetrievalAction.RETRIEVE

    def test_wait_when_c_below_tau(self):
        d = self.s.decide(self.q, c=0.50, label="writing_document", buffer_hit=False)
        assert d.action == RetrievalAction.WAIT

    def test_wait_when_c_just_below_tau(self):
        d = self.s.decide(self.q, c=0.64, label="writing_document", buffer_hit=False)
        assert d.action == RetrievalAction.WAIT

    def test_wait_when_buffer_hit_even_if_c_high(self):
        """Subscriber already has warm context — no redundant retrieval."""
        d = self.s.decide(self.q, c=0.95, label="debugging_python", buffer_hit=True)
        assert d.action == RetrievalAction.WAIT

    def test_wait_when_buffer_hit_and_c_below_tau(self):
        d = self.s.decide(self.q, c=0.40, label="reading_reference", buffer_hit=True)
        assert d.action == RetrievalAction.WAIT


# ── urgency_flag override ────────────────────────────────────────────────────

class TestUrgencyOverride:
    def setup_method(self):
        self.s = SpeculativeScheduler(tau=0.65)
        self.q = _vec()

    def test_urgency_retrieves_even_with_low_confidence(self):
        d = self.s.decide(self.q, c=0.10, label="anomaly_event", urgency_flag=True)
        assert d.action == RetrievalAction.RETRIEVE

    def test_urgency_retrieves_even_with_buffer_hit(self):
        """urgency_flag overrides buffer hit — safety-critical domains cannot wait."""
        d = self.s.decide(self.q, c=0.10, label="anomaly_event",
                          urgency_flag=True, buffer_hit=True)
        assert d.action == RetrievalAction.RETRIEVE

    def test_urgency_retrieves_even_with_c_zero(self):
        d = self.s.decide(self.q, c=0.0, label="anomaly_event", urgency_flag=True)
        assert d.action == RetrievalAction.RETRIEVE

    def test_urgency_tau_used_is_zero(self):
        """When urgency fires, the logged τ must be 0.0 — not the calibrated value."""
        d = self.s.decide(self.q, c=0.50, label="anomaly_event", urgency_flag=True)
        assert d.tau_used == 0.0

    def test_urgency_overrides_battery_saver(self):
        """urgency_flag wins even when battery saver is active (τ = 0.80)."""
        self.s.set_battery_saver(True)
        d = self.s.decide(self.q, c=0.10, label="anomaly_event", urgency_flag=True)
        assert d.action == RetrievalAction.RETRIEVE
        assert d.tau_used == 0.0


# ── Battery saver mode ───────────────────────────────────────────────────────

class TestBatterySaver:
    def setup_method(self):
        self.s = SpeculativeScheduler(tau=0.65)
        self.q = _vec()

    def test_battery_saver_raises_tau_to_0_80(self):
        self.s.set_battery_saver(True)
        # c=0.70 would normally RETRIEVE (> 0.65), but must WAIT under battery saver
        d = self.s.decide(self.q, c=0.70, label="writing_document")
        assert d.action == RetrievalAction.WAIT

    def test_battery_saver_tau_used_is_0_80(self):
        self.s.set_battery_saver(True)
        d = self.s.decide(self.q, c=0.70, label="writing_document")
        assert d.tau_used == 0.80

    def test_battery_saver_still_retrieves_above_0_80(self):
        self.s.set_battery_saver(True)
        d = self.s.decide(self.q, c=0.85, label="debugging_python")
        assert d.action == RetrievalAction.RETRIEVE

    def test_battery_saver_off_restores_calibrated_tau(self):
        self.s.set_battery_saver(True)
        self.s.set_battery_saver(False)
        # c=0.70 > 0.65 → should RETRIEVE again
        d = self.s.decide(self.q, c=0.70, label="writing_document")
        assert d.action == RetrievalAction.RETRIEVE


# ── update_tau — live calibration ────────────────────────────────────────────

class TestUpdateTau:
    def setup_method(self):
        self.s = SpeculativeScheduler(tau=0.65)
        self.q = _vec()

    def test_update_tau_takes_effect_immediately(self):
        # c=0.70 currently RETRIEVES (> 0.65)
        d_before = self.s.decide(self.q, c=0.70, label="writing_document")
        assert d_before.action == RetrievalAction.RETRIEVE

        self.s.update_tau(0.80)

        # same c=0.70 must now WAIT (< 0.80)
        d_after = self.s.decide(self.q, c=0.70, label="writing_document")
        assert d_after.action == RetrievalAction.WAIT

    def test_update_tau_reflected_in_tau_property(self):
        self.s.update_tau(0.72)
        assert self.s.tau == 0.72

    def test_update_tau_zero_always_retrieves(self):
        self.s.update_tau(0.0)
        d = self.s.decide(self.q, c=0.0, label="writing_document")
        assert d.action == RetrievalAction.RETRIEVE

    def test_update_tau_one_never_retrieves_except_urgency(self):
        self.s.update_tau(1.0)
        d = self.s.decide(self.q, c=0.999, label="writing_document")
        assert d.action == RetrievalAction.WAIT

    def test_update_tau_validates_above_one(self):
        with pytest.raises(ValueError):
            self.s.update_tau(1.5)

    def test_update_tau_validates_below_zero(self):
        with pytest.raises(ValueError):
            self.s.update_tau(-0.1)

    def test_update_tau_accepts_boundary_zero(self):
        self.s.update_tau(0.0)
        assert self.s.tau == 0.0

    def test_update_tau_accepts_boundary_one(self):
        self.s.update_tau(1.0)
        assert self.s.tau == 1.0


# ── tau_used logging ─────────────────────────────────────────────────────────

class TestTauUsedLogging:
    """tau_used in the decision must reflect what τ was actually used."""

    def setup_method(self):
        self.q = _vec()

    def test_tau_used_matches_calibrated_tau(self):
        s = SpeculativeScheduler(tau=0.72)
        d = s.decide(self.q, c=0.80, label="writing_document")
        assert d.tau_used == 0.72

    def test_tau_used_is_0_80_in_battery_saver(self):
        s = SpeculativeScheduler(tau=0.65)
        s.set_battery_saver(True)
        d = s.decide(self.q, c=0.85, label="debugging_python")
        assert d.tau_used == 0.80

    def test_tau_used_is_0_0_for_urgency(self):
        s = SpeculativeScheduler(tau=0.65)
        d = s.decide(self.q, c=0.50, label="anomaly_event", urgency_flag=True)
        assert d.tau_used == 0.0


# ── Scheduler does not modify q̂ ──────────────────────────────────────────────

class TestQHatPassThrough:
    def test_scheduler_does_not_modify_q_hat(self):
        """q̂ must arrive at the retrieval engine unchanged."""
        s = SpeculativeScheduler()
        q = _vec()
        original = q.copy()
        s.decide(q, c=0.80, label="writing_document")
        np.testing.assert_array_equal(q, original)

    def test_scheduler_accepts_any_ndarray_shape(self):
        """The scheduler must not make assumptions about q̂ content."""
        s = SpeculativeScheduler()
        for dim in [128, 384, 768]:
            q = np.random.default_rng(dim).random(dim).astype(np.float32)
            d = s.decide(q, c=0.80, label="writing_document")
            assert d.action in (RetrievalAction.RETRIEVE, RetrievalAction.WAIT)
