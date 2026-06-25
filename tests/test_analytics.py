"""
Tests for the DuckDB analytics store.

Pure Python / DuckDB — no Ollama, no mocks.
All stores use :memory: databases so tests are isolated and fast.

Metrics under test:
  PRP  = claimed_prefetches / total_prefetches  (target > 0.65)
  LtC  = mean latency_ms when claimed           (target: negative mean)
  BI   = computed externally from battery_events
  DPS  = mean(relevance_score + format_score)/2
"""
import time

import pytest

from apex.analytics.store import AnalyticsStore


def _store() -> AnalyticsStore:
    return AnalyticsStore(db_path=":memory:")


# ── Schema / setup ────────────────────────────────────────────────────────────

class TestSchema:
    def test_store_initialises_without_error(self):
        s = _store()
        assert s is not None

    def test_prefetch_events_table_exists(self):
        s = _store()
        result = s.con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'prefetch_events'"
        ).fetchone()
        assert result is not None

    def test_battery_events_table_exists(self):
        s = _store()
        result = s.con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'battery_events'"
        ).fetchone()
        assert result is not None

    def test_delivery_events_table_exists(self):
        s = _store()
        result = s.con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'delivery_events'"
        ).fetchone()
        assert result is not None


# ── log_prefetch ──────────────────────────────────────────────────────────────

class TestLogPrefetch:
    def test_log_prefetch_returns_event_id(self):
        s = _store()
        eid = s.log_prefetch("sess1", "sub1", "debugging_python", c=0.9, tau_used=0.65)
        assert isinstance(eid, int) and eid >= 0

    def test_log_prefetch_creates_row(self):
        s = _store()
        s.log_prefetch("sess1", "sub1", "debugging_python", c=0.9, tau_used=0.65)
        count = s.con.execute("SELECT COUNT(*) FROM prefetch_events").fetchone()[0]
        assert count == 1

    def test_log_prefetch_unclaimed_by_default(self):
        s = _store()
        s.log_prefetch("sess1", "sub1", "debugging_python", c=0.9, tau_used=0.65)
        row = s.con.execute(
            "SELECT claimed FROM prefetch_events"
        ).fetchone()
        assert row[0] is False

    def test_log_prefetch_stores_fields(self):
        s = _store()
        s.log_prefetch("s1", "u1", "writing_document", c=0.75, tau_used=0.65)
        row = s.con.execute(
            "SELECT session_id, subscriber_id, label, confidence, tau_used "
            "FROM prefetch_events"
        ).fetchone()
        assert row[0] == "s1"
        assert row[1] == "u1"
        assert row[2] == "writing_document"
        assert abs(row[3] - 0.75) < 1e-6
        assert abs(row[4] - 0.65) < 1e-6


# ── log_claim ─────────────────────────────────────────────────────────────────

class TestLogClaim:
    def test_log_claim_marks_event_claimed(self):
        s = _store()
        eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
        s.log_claim(eid)
        row = s.con.execute(
            "SELECT claimed FROM prefetch_events WHERE id = ?", [eid]
        ).fetchone()
        assert row[0] is True

    def test_log_claim_sets_latency_ms(self):
        s = _store()
        eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
        time.sleep(0.05)  # simulate time passing before claim
        s.log_claim(eid)
        row = s.con.execute(
            "SELECT latency_ms FROM prefetch_events WHERE id = ?", [eid]
        ).fetchone()
        assert row[0] is not None
        # latency is t_claimed - t_available in ms
        assert isinstance(row[0], float)

    def test_log_claim_unknown_id_is_safe(self):
        s = _store()
        s.log_claim(99999)  # must not raise

    def test_log_claim_sets_t_claimed(self):
        s = _store()
        eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
        before = time.time()
        s.log_claim(eid)
        after = time.time()
        row = s.con.execute(
            "SELECT t_claimed FROM prefetch_events WHERE id = ?", [eid]
        ).fetchone()
        assert before <= row[0] <= after


# ── log_battery ───────────────────────────────────────────────────────────────

class TestLogBattery:
    def test_log_battery_creates_row(self):
        s = _store()
        s.log_battery("s1", mw_reading=250.0, apex_running=True)
        count = s.con.execute("SELECT COUNT(*) FROM battery_events").fetchone()[0]
        assert count == 1

    def test_log_battery_stores_fields(self):
        s = _store()
        s.log_battery("s1", mw_reading=300.5, apex_running=False)
        row = s.con.execute(
            "SELECT session_id, mw_reading, apex_running FROM battery_events"
        ).fetchone()
        assert row[0] == "s1"
        assert abs(row[1] - 300.5) < 0.01
        assert row[2] is False


# ── log_delivery ──────────────────────────────────────────────────────────────

class TestLogDelivery:
    def test_log_delivery_creates_row(self):
        s = _store()
        s.log_delivery("s1", "u1", relevance_score=0.8, format_score=0.9)
        count = s.con.execute("SELECT COUNT(*) FROM delivery_events").fetchone()[0]
        assert count == 1

    def test_log_delivery_stores_scores(self):
        s = _store()
        s.log_delivery("s1", "u1", relevance_score=0.75, format_score=0.85)
        row = s.con.execute(
            "SELECT relevance_score, format_score FROM delivery_events"
        ).fetchone()
        assert abs(row[0] - 0.75) < 1e-6
        assert abs(row[1] - 0.85) < 1e-6


# ── PRP metric ────────────────────────────────────────────────────────────────

class TestPRP:
    def test_prp_all_claimed(self):
        s = _store()
        for _ in range(3):
            eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
            s.log_claim(eid)
        prp = s.compute_prp("s1")
        assert abs(prp - 1.0) < 1e-6

    def test_prp_none_claimed(self):
        s = _store()
        for _ in range(4):
            s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
        prp = s.compute_prp("s1")
        assert prp == 0.0

    def test_prp_partial_claim(self):
        s = _store()
        for i in range(4):
            eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
            if i < 2:
                s.log_claim(eid)
        prp = s.compute_prp("s1")
        assert abs(prp - 0.5) < 1e-6

    def test_prp_empty_session_returns_none(self):
        s = _store()
        prp = s.compute_prp("empty_session")
        assert prp is None

    def test_prp_session_isolated(self):
        s = _store()
        # Session A: 1/1 claimed
        eid = s.log_prefetch("sA", "u1", "debugging_python", c=0.9, tau_used=0.65)
        s.log_claim(eid)
        # Session B: 0/1 claimed
        s.log_prefetch("sB", "u1", "debugging_python", c=0.9, tau_used=0.65)
        assert abs(s.compute_prp("sA") - 1.0) < 1e-6
        assert s.compute_prp("sB") == 0.0


# ── LtC metric ────────────────────────────────────────────────────────────────

class TestLtC:
    def test_ltc_returns_float_for_claimed_events(self):
        s = _store()
        eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
        s.log_claim(eid)
        ltc = s.compute_mean_ltc("s1")
        assert isinstance(ltc, float)

    def test_ltc_is_negative_for_proactive_prefetch(self):
        # APEX prefetched, then user claimed later — LtC = t_available - t_claimed < 0
        # Negative LtC means APEX was ready before the user needed it (proactive).
        s = _store()
        eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
        time.sleep(0.05)
        s.log_claim(eid)
        ltc = s.compute_mean_ltc("s1")
        assert ltc is not None
        assert ltc < 0  # proactive: APEX was ready before claim

    def test_ltc_empty_returns_none(self):
        s = _store()
        assert s.compute_mean_ltc("empty") is None

    def test_ltc_unclaimed_not_included(self):
        s = _store()
        eid = s.log_prefetch("s1", "u1", "debugging_python", c=0.9, tau_used=0.65)
        s.log_claim(eid)
        s.log_prefetch("s1", "u1", "writing_document", c=0.7, tau_used=0.65)
        ltc = s.compute_mean_ltc("s1")
        assert ltc is not None


# ── claim_via_pull ────────────────────────────────────────────────────────────

class TestClaimViaPull:
    """
    Pull-mode supervision: the vault agent calls GET /context/{id} at T_need.
    claim_via_pull() finds the most recent unclaimed prefetch within the window
    and claims it, recording LtC = t_available - t_need (negative = proactive).
    """

    def test_claims_recent_unclaimed_prefetch(self):
        s = _store()
        eid = s.log_prefetch("s1", "u1", "writing_document", c=0.8, tau_used=0.65)
        time.sleep(0.02)
        t_need = time.time()
        n = s.claim_via_pull(session_id="s1", subscriber_id="u1", t_need=t_need)
        assert n == 1
        row = s.con.execute(
            "SELECT claimed FROM prefetch_events WHERE id=?", [eid]
        ).fetchone()
        assert row[0] is True

    def test_ltc_is_negative_after_pull_claim(self):
        # Prefetch logged → some time passes → pull at t_need
        # LtC = t_available - t_need → negative (APEX was ready before need)
        s = _store()
        s.log_prefetch("s1", "u1", "writing_document", c=0.8, tau_used=0.65)
        time.sleep(0.05)
        t_need = time.time()
        s.claim_via_pull(session_id="s1", subscriber_id="u1", t_need=t_need)
        ltc = s.compute_mean_ltc("s1")
        assert ltc is not None
        assert ltc < 0  # proactive lead — negative is good

    def test_no_claims_when_all_prefetches_already_claimed(self):
        s = _store()
        eid = s.log_prefetch("s1", "u1", "writing_document", c=0.8, tau_used=0.65)
        s.log_claim(eid)
        n = s.claim_via_pull(session_id="s1", subscriber_id="u1", t_need=time.time())
        assert n == 0

    def test_no_claims_for_wrong_subscriber(self):
        s = _store()
        s.log_prefetch("s1", "u1", "writing_document", c=0.8, tau_used=0.65)
        t_need = time.time()
        n = s.claim_via_pull(session_id="s1", subscriber_id="u2", t_need=t_need)
        assert n == 0

    def test_no_claims_outside_window(self):
        s = _store()
        s.log_prefetch("s1", "u1", "writing_document", c=0.8, tau_used=0.65)
        # t_need is far in the future — outside the window
        t_need = time.time() + 3600
        n = s.claim_via_pull(
            session_id="s1", subscriber_id="u1", t_need=t_need, window_s=60.0
        )
        assert n == 0

    def test_claims_at_most_one_per_call(self):
        # Three prefetches for same subscriber — one pull should claim only one
        s = _store()
        for _ in range(3):
            s.log_prefetch("s1", "u1", "writing_document", c=0.8, tau_used=0.65)
        n = s.claim_via_pull(session_id="s1", subscriber_id="u1", t_need=time.time())
        assert n == 1
        total_claimed = s.con.execute(
            "SELECT COUNT(*) FROM prefetch_events WHERE claimed=TRUE"
        ).fetchone()[0]
        assert total_claimed == 1

    def test_claims_most_recent_not_oldest(self):
        # Two prefetches — pull should claim the most recent one
        s = _store()
        eid_old = s.log_prefetch("s1", "u1", "writing_document", c=0.8, tau_used=0.65)
        time.sleep(0.01)
        eid_new = s.log_prefetch("s1", "u1", "writing_document", c=0.9, tau_used=0.65)
        s.claim_via_pull(session_id="s1", subscriber_id="u1", t_need=time.time())
        old_row = s.con.execute(
            "SELECT claimed FROM prefetch_events WHERE id=?", [eid_old]
        ).fetchone()
        new_row = s.con.execute(
            "SELECT claimed FROM prefetch_events WHERE id=?", [eid_new]
        ).fetchone()
        assert old_row[0] is False   # older one untouched
        assert new_row[0] is True    # most recent one claimed


# ── DPS metric ────────────────────────────────────────────────────────────────

class TestDPS:
    def test_dps_returns_float(self):
        s = _store()
        s.log_delivery("s1", "u1", relevance_score=0.8, format_score=0.9)
        dps = s.compute_dps("s1")
        assert isinstance(dps, float)

    def test_dps_is_mean_of_scores(self):
        s = _store()
        s.log_delivery("s1", "u1", relevance_score=0.6, format_score=0.8)
        s.log_delivery("s1", "u1", relevance_score=0.8, format_score=0.6)
        dps = s.compute_dps("s1")
        # mean relevance = 0.7, mean format = 0.7, DPS = 0.7
        assert abs(dps - 0.7) < 1e-4

    def test_dps_empty_returns_none(self):
        s = _store()
        assert s.compute_dps("empty") is None
