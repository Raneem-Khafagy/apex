"""
DuckDB Analytics Store — thesis evaluation event logger.

Records four event types for computing the four thesis metrics:
  PRP  = claimed_prefetches / total_prefetches       (target > 0.65)
  LtC  = mean latency_ms for claimed prefetches      (target: negative mean)
  BI   = derived externally from battery_events      (target < 15% overhead)
  DPS  = mean(relevance_score + format_score) / 2    (target > 0.75)

All tables use :memory: by default so unit tests are isolated and fast.
"""
from __future__ import annotations

import time
from typing import Optional

import duckdb
from loguru import logger


class AnalyticsStore:
    """
    Thin DuckDB wrapper for APEX thesis evaluation events.

    Parameters
    ----------
    db_path
        Path to the DuckDB file, or ":memory:" for an in-process store.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.con = duckdb.connect(db_path)
        self._create_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS prefetch_events (
                id            INTEGER PRIMARY KEY,
                session_id    VARCHAR NOT NULL,
                subscriber_id VARCHAR NOT NULL,
                label         VARCHAR NOT NULL,
                confidence    DOUBLE  NOT NULL,
                tau_used      DOUBLE  NOT NULL,
                t_available   DOUBLE  NOT NULL,
                t_claimed     DOUBLE,
                claimed       BOOLEAN NOT NULL DEFAULT FALSE,
                latency_ms    DOUBLE,
                t_signal      DOUBLE,
                t_iie         DOUBLE,
                t_retrieval   DOUBLE,
                t_push        DOUBLE,
                iie_ms        DOUBLE,
                retrieval_ms  DOUBLE,
                push_ms       DOUBLE,
                multi_sub_overhead_ms DOUBLE,
                subscriber_count INTEGER
            )
        """)
        # Sync sequence past any rows that survived a previous crash.
        _max_p = self.con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM prefetch_events"
        ).fetchone()[0]
        self.con.execute(
            f"CREATE OR REPLACE SEQUENCE prefetch_events_id_seq START {_max_p + 1}"
        )

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS battery_events (
                id           INTEGER PRIMARY KEY,
                session_id   VARCHAR NOT NULL,
                ts           DOUBLE  NOT NULL,
                mw_reading   DOUBLE  NOT NULL,
                apex_running BOOLEAN NOT NULL
            )
        """)
        _max_b = self.con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM battery_events"
        ).fetchone()[0]
        self.con.execute(
            f"CREATE OR REPLACE SEQUENCE battery_events_id_seq START {_max_b + 1}"
        )

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS delivery_events (
                id               INTEGER PRIMARY KEY,
                session_id       VARCHAR NOT NULL,
                subscriber_id    VARCHAR NOT NULL,
                ts               DOUBLE  NOT NULL,
                relevance_score  DOUBLE  NOT NULL,
                format_score     DOUBLE  NOT NULL
            )
        """)
        _max_d = self.con.execute(
            "SELECT COALESCE(MAX(id), 0) FROM delivery_events"
        ).fetchone()[0]
        self.con.execute(
            f"CREATE OR REPLACE SEQUENCE delivery_events_id_seq START {_max_d + 1}"
        )

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_prefetch(
        self,
        session_id: str,
        subscriber_id: str,
        label: str,
        c: float,
        tau_used: float,
        t_signal: Optional[float] = None,
        t_iie: Optional[float] = None,
        t_retrieval: Optional[float] = None,
        t_push: Optional[float] = None,
        multi_sub_overhead_ms: Optional[float] = None,
        subscriber_count: Optional[int] = None,
    ) -> int:
        """
        Record a prefetch attempt.

        Parameters
        ----------
        t_signal, t_iie, t_retrieval, t_push
            Four timestamps for latency profiling. If provided, automatically
            computes iie_ms, retrieval_ms, push_ms stage latencies.
        multi_sub_overhead_ms
            Multi-subscriber formatting overhead in milliseconds.
            Time from retrieval complete to all subscribers formatted.
        subscriber_count
            Number of subscribers that were processed for this retrieval.

        Returns
        -------
        Row ID of the new prefetch event (used to claim it later).
        """
        t_available = time.time()
        event_id = self.con.execute(
            "SELECT nextval('prefetch_events_id_seq')"
        ).fetchone()[0]

        # Compute stage latencies if timestamps are provided
        iie_ms = (t_iie - t_signal) * 1000.0 if t_signal and t_iie else None
        retrieval_ms = (t_retrieval - t_iie) * 1000.0 if t_iie and t_retrieval else None
        push_ms = (t_push - t_retrieval) * 1000.0 if t_retrieval and t_push else None

        self.con.execute(
            """
            INSERT INTO prefetch_events
                (id, session_id, subscriber_id, label, confidence,
                 tau_used, t_available, t_claimed, claimed, latency_ms,
                 t_signal, t_iie, t_retrieval, t_push,
                 iie_ms, retrieval_ms, push_ms,
                 multi_sub_overhead_ms, subscriber_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, FALSE, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [event_id, session_id, subscriber_id, label, c, tau_used, t_available,
             t_signal, t_iie, t_retrieval, t_push, iie_ms, retrieval_ms, push_ms,
             multi_sub_overhead_ms, subscriber_count],
        )
        logger.debug(
            "Analytics: log_prefetch id={} session='{}' label='{}' c={:.3f}",
            event_id, session_id, label, c,
        )
        return event_id

    def log_claim(self, event_id: int) -> None:
        """
        Mark a prefetch event as claimed by a consumer.

        Sets claimed=True, records t_claimed, and computes
        latency_ms = (t_available - t_claimed) * 1000.

        Sign convention (matches CLAUDE.md LtC definition):
            negative  → APEX was ready BEFORE the consumer claimed  (proactive ✓)
            near-zero → push delivery, claim ~= prefetch time       (baseline)
            positive  → consumer claimed BEFORE prefetch logged     (impossible normally)

        Safe to call with an unknown event_id (no-op).
        """
        t_claimed = time.time()
        # Only claim if not already claimed — prevents push delivery from
        # overwriting a pull-supervision claim that arrived earlier.
        self.con.execute(
            """
            UPDATE prefetch_events
            SET claimed    = TRUE,
                t_claimed  = ?,
                latency_ms = (t_available - ?) * 1000.0
            WHERE id = ?
              AND claimed = FALSE
            """,
            [t_claimed, t_claimed, event_id],
        )
        logger.debug("Analytics: log_claim id={}", event_id)

    def update_push_timing(self, event_id: int, t_push: float) -> None:
        """
        Update t_push timestamp and recompute push_ms for an existing prefetch event.

        Called after push delivery is complete to finalize timing metrics.
        """
        self.con.execute(
            """
            UPDATE prefetch_events
            SET t_push = ?,
                push_ms = CASE
                    WHEN t_retrieval IS NOT NULL THEN (? - t_retrieval) * 1000.0
                    ELSE NULL
                END
            WHERE id = ?
            """,
            [t_push, t_push, event_id],
        )
        logger.debug("Analytics: update_push_timing id={} t_push={:.6f}", event_id, t_push)

    def claim_via_pull(
        self,
        session_id: str,
        subscriber_id: str,
        t_need: float,
        window_s: float = 60.0,
    ) -> int:
        """
        Pull-mode supervision claim.

        Called when a subscriber explicitly pulls context (GET /context/{id}).
        Finds the most recent unclaimed prefetch for this subscriber that
        arrived within ``window_s`` seconds before ``t_need``, and claims it.

        Parameters
        ----------
        session_id
            Current daemon session (must match coordinator.session_id).
        subscriber_id
            Subscriber performing the pull.
        t_need
            Unix timestamp of the pull request — treated as the "need" time.
        window_s
            How far back to look for unclaimed prefetches. Default: 60 s.

        Returns
        -------
        Number of events claimed (0 or 1). Zero means no warm prefetch was
        available when the user pulled — a proactive miss.
        """
        t_window_start = t_need - window_s
        row = self.con.execute(
            """
            SELECT id
            FROM prefetch_events
            WHERE session_id    = ?
              AND subscriber_id = ?
              AND claimed       = FALSE
              AND t_available  >= ?
              AND t_available  <= ?
            ORDER BY t_available DESC
            LIMIT 1
            """,
            [session_id, subscriber_id, t_window_start, t_need],
        ).fetchone()

        if row is None:
            # Diagnostic: distinguish "wrong session" / "all claimed" / "push never fired"
            debug = self.con.execute(
                "SELECT COUNT(*) FROM prefetch_events WHERE subscriber_id = ? AND claimed = FALSE",
                [subscriber_id],
            ).fetchone()
            unclaimed_any = debug[0] if debug else 0
            logger.debug(
                "Analytics: claim_via_pull — 0 found for session='{}' sub='{}' window={}s "
                "| {} unclaimed rows exist across ALL sessions",
                session_id, subscriber_id, window_s, unclaimed_any,
            )
            return 0

        event_id = row[0]
        self.con.execute(
            """
            UPDATE prefetch_events
            SET claimed    = TRUE,
                t_claimed  = ?,
                latency_ms = (t_available - ?) * 1000.0
            WHERE id = ?
            """,
            [t_need, t_need, event_id],
        )
        logger.debug(
            "Analytics: claim_via_pull id={} sub='{}' t_need={:.3f}",
            event_id, subscriber_id, t_need,
        )
        return 1

    def log_battery(
        self,
        session_id: str,
        mw_reading: float,
        apex_running: bool,
    ) -> None:
        """Record a battery power sample."""
        ts = time.time()
        event_id = self.con.execute(
            "SELECT nextval('battery_events_id_seq')"
        ).fetchone()[0]
        self.con.execute(
            """
            INSERT INTO battery_events (id, session_id, ts, mw_reading, apex_running)
            VALUES (?, ?, ?, ?, ?)
            """,
            [event_id, session_id, ts, mw_reading, apex_running],
        )
        logger.debug(
            "Analytics: log_battery session='{}' mW={:.1f} apex={}",
            session_id, mw_reading, apex_running,
        )

    def log_delivery(
        self,
        session_id: str,
        subscriber_id: str,
        relevance_score: float,
        format_score: float,
    ) -> None:
        """Record a human-annotated delivery quality score."""
        ts = time.time()
        event_id = self.con.execute(
            "SELECT nextval('delivery_events_id_seq')"
        ).fetchone()[0]
        self.con.execute(
            """
            INSERT INTO delivery_events
                (id, session_id, subscriber_id, ts, relevance_score, format_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [event_id, session_id, subscriber_id, ts, relevance_score, format_score],
        )
        logger.debug(
            "Analytics: log_delivery session='{}' rel={:.3f} fmt={:.3f}",
            session_id, relevance_score, format_score,
        )

    # ── Metric queries ────────────────────────────────────────────────────────

    def compute_prp(self, session_id: str) -> Optional[float]:
        """
        Proactive Retrieval Precision for a session.

        Returns claimed_prefetches / total_prefetches, or None if the
        session has no prefetch events.
        """
        row = self.con.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE claimed) * 1.0 / NULLIF(COUNT(*), 0)
            FROM prefetch_events
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])

    def compute_mean_ltc(self, session_id: str) -> Optional[float]:
        """
        Mean Latency-to-Context for claimed prefetch events.

        Returns mean latency_ms (positive = context arrived after need),
        or None if there are no claimed events for the session.
        """
        row = self.con.execute(
            """
            SELECT AVG(latency_ms)
            FROM prefetch_events
            WHERE session_id = ? AND claimed = TRUE
            """,
            [session_id],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])

    def compute_dps(self, session_id: str) -> Optional[float]:
        """
        Delivery Precision Score — mean of (relevance + format) / 2 per session.

        Formula: AVG((relevance_score + format_score) / 2)
        Returns None if no delivery events for the session.
        """
        row = self.con.execute(
            """
            SELECT AVG((relevance_score + format_score) / 2.0)
            FROM delivery_events
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])

    def compute_stage_latencies(self, session_id: str) -> dict[str, Optional[float]]:
        """
        Compute mean stage latencies for Claim 2 analysis.

        Returns
        -------
        Dictionary with 'iie_ms', 'retrieval_ms', 'push_ms' mean values.
        None values indicate no data available for that stage.
        """
        rows = self.con.execute(
            """
            SELECT
                AVG(iie_ms) as mean_iie_ms,
                AVG(retrieval_ms) as mean_retrieval_ms,
                AVG(push_ms) as mean_push_ms
            FROM prefetch_events
            WHERE session_id = ?
              AND iie_ms IS NOT NULL
              AND retrieval_ms IS NOT NULL
              AND push_ms IS NOT NULL
            """,
            [session_id],
        ).fetchone()

        if rows is None:
            return {"iie_ms": None, "retrieval_ms": None, "push_ms": None}

        return {
            "iie_ms": float(rows[0]) if rows[0] is not None else None,
            "retrieval_ms": float(rows[1]) if rows[1] is not None else None,
            "push_ms": float(rows[2]) if rows[2] is not None else None,
        }

    def compute_multi_subscriber_overhead(self, session_id: str) -> dict[str, Optional[float]]:
        """
        Compute multi-subscriber overhead statistics for Claim 3 analysis.

        Returns mean overhead by subscriber count and overhead scaling metrics.
        """
        # Get overhead data grouped by subscriber count
        rows = self.con.execute(
            """
            SELECT
                subscriber_count,
                AVG(multi_sub_overhead_ms) as mean_overhead_ms,
                COUNT(*) as event_count
            FROM prefetch_events
            WHERE session_id = ?
              AND multi_sub_overhead_ms IS NOT NULL
              AND subscriber_count IS NOT NULL
            GROUP BY subscriber_count
            ORDER BY subscriber_count
            """,
            [session_id],
        ).fetchall()

        if not rows:
            return {
                "overhead_by_count": {},
                "scaling_factor": None,
                "single_sub_mean_ms": None,
                "multi_sub_mean_ms": None
            }

        overhead_by_count = {}
        single_sub_overhead = None
        multi_sub_overhead = None

        for subscriber_count, mean_overhead, event_count in rows:
            count = int(subscriber_count)
            overhead = float(mean_overhead)
            overhead_by_count[count] = {
                "mean_overhead_ms": overhead,
                "event_count": int(event_count)
            }

            if count == 1:
                single_sub_overhead = overhead
            elif count > 1:
                # Use the highest subscriber count as representative multi-subscriber
                multi_sub_overhead = overhead

        # Compute scaling factor (multi-sub / single-sub)
        scaling_factor = None
        if single_sub_overhead and multi_sub_overhead:
            scaling_factor = multi_sub_overhead / single_sub_overhead

        return {
            "overhead_by_count": overhead_by_count,
            "scaling_factor": scaling_factor,
            "single_sub_mean_ms": single_sub_overhead,
            "multi_sub_mean_ms": multi_sub_overhead
        }
