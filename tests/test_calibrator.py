"""
Tests for TauCalibrator.

TauCalibrator reads claim rate per confidence bucket from DuckDB
and calls SpeculativeScheduler.update_tau() when a new estimate
is available.

No mocks — uses real AnalyticsStore (DuckDB :memory:) and real
SpeculativeScheduler.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from apex.analytics.store import AnalyticsStore
from apex.inference.tau_calibrator import TauCalibrator
from apex.scheduler.speculative import SpeculativeScheduler


# ── Helpers ──────────────────────────────────────────────────────────────────

def _store() -> AnalyticsStore:
    return AnalyticsStore(db_path=":memory:")


def _scheduler(tau: float = 0.65) -> SpeculativeScheduler:
    return SpeculativeScheduler(tau=tau)


def _seed_events(
    store: AnalyticsStore,
    session_id: str,
    *,
    c: float,
    n_total: int,
    n_claimed: int,
) -> None:
    """Seed n_total prefetch events with given confidence; claim n_claimed of them."""
    assert n_claimed <= n_total
    event_ids = []
    for _ in range(n_total):
        eid = store.log_prefetch(
            session_id=session_id,
            subscriber_id="sub1",
            label="debugging_python",
            c=c,
            tau_used=0.65,
        )
        event_ids.append(eid)
    for eid in event_ids[:n_claimed]:
        store.log_claim(eid)


# ── Construction ─────────────────────────────────────────────────────────────

class TestConstruction:
    def test_calibrator_constructs(self):
        cal = TauCalibrator(store=_store(), scheduler=_scheduler())
        assert cal is not None

    def test_default_min_events(self):
        cal = TauCalibrator(store=_store(), scheduler=_scheduler())
        assert cal.min_events >= 5

    def test_calibrator_accepts_custom_session_id(self):
        cal = TauCalibrator(
            store=_store(),
            scheduler=_scheduler(),
            session_id="mysession",
        )
        assert cal.session_id == "mysession"


# ── calibrate() — core method ─────────────────────────────────────────────────

class TestCalibrateMethod:
    def test_no_op_when_store_empty(self):
        """With no events, τ must not change."""
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=_store(), scheduler=sched)
        cal.calibrate()
        assert sched.tau == pytest.approx(0.65)

    def test_no_op_below_min_events(self):
        """With fewer events than min_events threshold, τ must not change."""
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=10)
        # Seed only 3 events — below min_events=10
        _seed_events(store, "s1", c=0.9, n_total=3, n_claimed=3)
        cal.calibrate()
        assert sched.tau == pytest.approx(0.65)

    def test_tau_decreases_when_sub_threshold_events_all_claimed(self):
        """
        Events at c=0.45 (below current τ=0.65), all claimed.

        This is the correct "over-cautious" signal: the IIE infers intent at
        confidence 0.45 and those prefetches are reliably useful (claim_rate=1.0
        >= TARGET_PRP=0.65).  Algorithm A finds the lowest qualifying bucket
        (bucket 4, c_mid≈0.45) and sets τ there — which is lower than 0.65.

        Note: data-driven calibration can ONLY lower τ when there is evidence
        of reliable retrievals at confidence levels below the current τ.
        """
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=5)
        # 20 events below current τ, all claimed → reliable at low confidence
        _seed_events(store, "s1", c=0.45, n_total=20, n_claimed=20)
        old_tau = sched.tau
        cal.calibrate()
        # τ should decrease to bucket 4's c_mid ≈ 0.45
        assert sched.tau < old_tau

    def test_tau_increases_when_low_confidence_mostly_unclaimed(self):
        """
        Events at c=0.67 (just above τ=0.65) mostly unclaimed → τ too low.
        New τ should be higher than 0.65.
        """
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=5)
        # 20 events just above τ, only 2 claimed (10% claim rate — well below 0.65)
        _seed_events(store, "s1", c=0.67, n_total=20, n_claimed=2)
        old_tau = sched.tau
        cal.calibrate()
        assert sched.tau > old_tau

    def test_tau_clamped_to_min(self):
        """Even if all events are claimed at every confidence level, τ stays >= TAU_MIN."""
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=5)
        # Many events at very low confidence, all claimed
        _seed_events(store, "s1", c=0.1, n_total=50, n_claimed=50)
        cal.calibrate()
        assert sched.tau >= cal.tau_min

    def test_tau_clamped_to_max(self):
        """Even if all events at all confidence levels unclaimed, τ stays <= TAU_MAX."""
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=5)
        # Many events, none claimed
        _seed_events(store, "s1", c=0.7, n_total=50, n_claimed=0)
        cal.calibrate()
        assert sched.tau <= cal.tau_max

    def test_tau_update_is_idempotent_given_same_data(self):
        """Calling calibrate() twice with the same data produces the same τ."""
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=5)
        _seed_events(store, "s1", c=0.9, n_total=20, n_claimed=18)
        cal.calibrate()
        tau_after_first = sched.tau
        cal.calibrate()
        assert sched.tau == pytest.approx(tau_after_first)

    def test_session_isolation(self):
        """
        Events from a different session must not affect calibration for our session.
        """
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="mine", min_events=5)
        # Seed a different session with 50 unclaimed events at c=0.7
        _seed_events(store, "other", c=0.7, n_total=50, n_claimed=0)
        cal.calibrate()
        # Should be no-op — "mine" session has no events
        assert sched.tau == pytest.approx(0.65)


# ── last_tau property ─────────────────────────────────────────────────────────

class TestLastTau:
    def test_last_tau_none_before_calibrate(self):
        cal = TauCalibrator(store=_store(), scheduler=_scheduler())
        assert cal.last_tau is None

    def test_last_tau_set_after_calibrate_changes_tau(self):
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=5)
        _seed_events(store, "s1", c=0.95, n_total=20, n_claimed=20)
        cal.calibrate()
        assert cal.last_tau is not None
        assert isinstance(cal.last_tau, float)

    def test_last_tau_none_when_no_op(self):
        """last_tau stays None when calibrate() is a no-op (too few events)."""
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(store=store, scheduler=sched, session_id="s1", min_events=10)
        _seed_events(store, "s1", c=0.9, n_total=2, n_claimed=2)
        cal.calibrate()
        assert cal.last_tau is None


# ── Async run loop ────────────────────────────────────────────────────────────

class TestRunLoop:
    @pytest.mark.asyncio
    async def test_run_and_stop(self):
        """run() must exit cleanly after stop() is called."""
        cal = TauCalibrator(
            store=_store(),
            scheduler=_scheduler(),
            interval=0.05,  # very fast for tests
        )
        task = asyncio.create_task(cal.run())
        await asyncio.sleep(0.2)
        cal.stop()
        await asyncio.wait_for(task, timeout=2.0)
        assert task.done()

    @pytest.mark.asyncio
    async def test_run_calls_calibrate_at_least_once(self):
        """After running briefly, calibrate() must have been called."""
        store = _store()
        sched = _scheduler(tau=0.65)
        cal = TauCalibrator(
            store=store,
            scheduler=sched,
            session_id="s1",
            min_events=5,
            interval=0.05,
        )
        _seed_events(store, "s1", c=0.95, n_total=20, n_claimed=20)

        task = asyncio.create_task(cal.run())
        await asyncio.sleep(0.3)
        cal.stop()
        await asyncio.wait_for(task, timeout=2.0)
        # τ should have changed from the initial 0.65
        assert sched.tau != pytest.approx(0.65)
