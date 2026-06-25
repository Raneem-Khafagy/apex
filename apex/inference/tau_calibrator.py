"""
τ Calibration Feedback Loop.

Reads claim rate per confidence bucket from DuckDB (via AnalyticsStore)
and calls SpeculativeScheduler.update_tau() when a more accurate threshold
can be estimated from observed claim rates.

This is APEX's defining adaptive behavior: τ starts at 0.65 and converges
toward the minimum confidence level at which retrieved context is reliably
useful to the subscriber. Every session generates labeled training data
with zero manual annotation.

Algorithm
---------
1. Query all prefetch_events for the current session.
2. Exit without updating if fewer than min_events rows exist.
3. Bucket events by confidence into N_BUCKETS equal-width intervals over [0, 1].
4. For each bucket (ascending), compute claim_rate = claimed / total.
5. The new τ is the midpoint of the LOWEST bucket whose claim_rate >= TARGET_PRP.
   Intuition: if events at confidence c are claimed 65%+ of the time, then c is
   a reliable threshold — fire retrieval whenever the IIE reaches that confidence.
6. If no bucket meets the target (all claim rates < TARGET), raise τ by one step
   (reduce spurious retrievals). If every bucket meets the target, lower τ by one
   step (we can afford to be more aggressive).
7. Clamp the result to [TAU_MIN, TAU_MAX] to prevent degenerate states.
8. Call scheduler.update_tau(new_tau) only when new_tau ≠ current tau.

The calibrator runs as an async background task, calling calibrate() every
`interval` seconds. Stop it with stop().

Privacy: only confidence scores and claim flags are read. No document content.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from loguru import logger

from apex.analytics.store import AnalyticsStore
from apex.scheduler.speculative import SpeculativeScheduler

# ── Constants ─────────────────────────────────────────────────────────────────

N_BUCKETS: int = 10          # confidence bins: [0.0,0.1), [0.1,0.2), …, [0.9,1.0]
TARGET_PRP: float = 0.65     # claim rate required for a bucket to qualify
TAU_MIN: float = 0.30        # never lower τ below this
TAU_MAX: float = 0.90        # never raise τ above this
DEFAULT_INTERVAL: float = 60.0   # seconds between calibration passes
DEFAULT_MIN_EVENTS: int = 10     # skip calibration if fewer events exist


class TauCalibrator:
    """
    Continuous calibration loop for the Speculative Retrieval Scheduler.

    Parameters
    ----------
    store
        AnalyticsStore providing access to the DuckDB prefetch_events table.
    scheduler
        SpeculativeScheduler whose τ will be updated.
    session_id
        DuckDB session scope for the calibration query.
        Typically the subscriber_id or a Phase 0 session label.
    interval
        Seconds between calibration passes.
    min_events
        Minimum number of prefetch events required before calibration runs.
        Prevents premature τ updates from statistically thin data.
    """

    def __init__(
        self,
        store: AnalyticsStore,
        scheduler: SpeculativeScheduler,
        session_id: str = "default",
        interval: float = DEFAULT_INTERVAL,
        min_events: int = DEFAULT_MIN_EVENTS,
        fixed_tau: Optional[bool] = None,
    ) -> None:
        self._store = store
        self._scheduler = scheduler
        self._session_id = session_id
        self._interval = interval
        self._min_events = min_events
        self._running = False
        self._last_tau: Optional[float] = None

        # Check for fixed τ mode via environment variable or explicit parameter
        if fixed_tau is None:
            self._fixed_tau = os.getenv("APEX_TAU_FIXED", "0").lower() in ("1", "true", "yes")
        else:
            self._fixed_tau = fixed_tau

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def min_events(self) -> int:
        return self._min_events

    @property
    def last_tau(self) -> Optional[float]:
        """The most recent τ value written to the scheduler, or None if never calibrated."""
        return self._last_tau

    @property
    def tau_min(self) -> float:
        return TAU_MIN

    @property
    def tau_max(self) -> float:
        return TAU_MAX

    @property
    def fixed_tau(self) -> bool:
        """Whether τ calibration is disabled (fixed at DEFAULT_TAU = 0.65)."""
        return self._fixed_tau

    # ── Core calibration ─────────────────────────────────────────────────────

    def calibrate(self) -> None:
        """
        Run one calibration pass.

        Reads prefetch_events for the current session, computes per-domain
        bucket claim rates, estimates optimal τ per domain, and calls
        scheduler.update_domain_tau() for each domain with sufficient data.

        If fixed_tau mode is enabled, this is a no-op and τ stays at the
        initial value (0.65) for the entire session.

        Safe to call repeatedly — idempotent given the same underlying data.
        """
        # Skip calibration if in fixed τ mode (for baseline comparison)
        if self._fixed_tau:
            logger.debug("TauCalibrator: fixed τ mode enabled — skipping calibration")
            return
        domain_buckets = self._compute_bucket_claim_rates_per_domain()
        if not domain_buckets:
            logger.debug("TauCalibrator: no events for session='{}' — skipping", self._session_id)
            return

        updated_domains = 0
        for label, buckets in domain_buckets.items():
            total_events = sum(b["total"] for b in buckets)
            if total_events < self._min_events:
                logger.debug(
                    "TauCalibrator: only {} events for session='{}' domain='{}' — need {} — skipping",
                    total_events, self._session_id, label, self._min_events,
                )
                continue

            new_tau = self._estimate_tau(buckets)
            current_tau = self._scheduler.get_tau(label)

            if abs(new_tau - current_tau) < 1e-6:
                logger.debug(
                    "TauCalibrator: τ unchanged at {:.4f} for session='{}' domain='{}'",
                    current_tau, self._session_id, label,
                )
                continue

            self._scheduler.update_domain_tau(label, new_tau)
            updated_domains += 1
            logger.info(
                "TauCalibrator: τ updated {:.4f} → {:.4f} for session='{}' domain='{}' ({} events)",
                current_tau, new_tau, self._session_id, label, total_events,
            )

        # Also update global τ based on aggregated data for backward compatibility
        self._update_global_tau(domain_buckets)

        if updated_domains == 0:
            logger.debug("TauCalibrator: no domains had sufficient events for calibration")

    # ── Async run loop ────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Run the calibration loop until stop() is called.

        Calls calibrate() immediately on startup, then every `interval` seconds.
        """
        self._running = True
        mode = "FIXED" if self._fixed_tau else "ADAPTIVE"
        logger.info(
            "TauCalibrator: started for session='{}' interval={}s mode={}",
            self._session_id, self._interval, mode,
        )
        while self._running:
            try:
                self.calibrate()
            except Exception as exc:
                # Never let a DuckDB or arithmetic error kill the daemon
                logger.error("TauCalibrator: calibrate() raised {}: {} — continuing", type(exc).__name__, exc)
            # Sleep in small slices so stop() is responsive
            slept = 0.0
            slice_s = min(0.05, self._interval)
            while self._running and slept < self._interval:
                await asyncio.sleep(slice_s)
                slept += slice_s
        logger.info("TauCalibrator: stopped for session='{}'", self._session_id)

    def stop(self) -> None:
        """Signal the calibration loop to exit after the current sleep slice."""
        self._running = False

    # ── Internal ─────────────────────────────────────────────────────────────

    def _compute_bucket_claim_rates_per_domain(self) -> dict[str, list[dict]]:
        """
        Group prefetch events by label and confidence bucket, compute claim rates.

        Returns a dict mapping label to bucket data:
            {"debugging_python": [bucket_dict, ...],
             "drafting_research": [bucket_dict, ...], ...}

        Each bucket_dict has the same structure as _compute_bucket_claim_rates().
        """
        bucket_width = 1.0 / N_BUCKETS

        # Group by label first, then by confidence bucket
        rows = self._store.con.execute(
            """
            SELECT
                label,
                LEAST(FLOOR(confidence * ?), ? - 1) AS bucket,
                COUNT(*)                              AS total,
                COUNT(*) FILTER (WHERE claimed)       AS claimed
            FROM prefetch_events
            WHERE session_id = ?
            GROUP BY label, bucket
            ORDER BY label, bucket
            """,
            [N_BUCKETS, N_BUCKETS, self._session_id],
        ).fetchall()

        result = {}
        for (label, bucket, total, claimed) in rows:
            bucket = int(bucket)
            c_low = bucket * bucket_width
            c_mid = c_low + bucket_width / 2.0
            claim_rate = claimed / total if total > 0 else 0.0

            bucket_data = {
                "bucket": bucket,
                "c_low": c_low,
                "c_mid": c_mid,
                "total": int(total),
                "claimed": int(claimed),
                "claim_rate": claim_rate,
            }

            if label not in result:
                result[label] = []
            result[label].append(bucket_data)

        return result

    def _compute_bucket_claim_rates(self) -> list[dict]:
        """
        Group prefetch events by confidence bucket and compute claim rates.
        Legacy method for backward compatibility.

        Returns a list of dicts sorted by bucket index (ascending confidence):
            [{"bucket": int, "c_low": float, "c_mid": float,
              "total": int, "claimed": int, "claim_rate": float}, ...]

        Buckets with zero events are omitted.
        """
        bucket_width = 1.0 / N_BUCKETS

        # Use integer bucket index (FLOOR(c * N_BUCKETS)) to avoid FP-division
        # edge cases. Confidence=1.0 would give bucket=10 → clamp to 9.
        rows = self._store.con.execute(
            """
            SELECT
                LEAST(FLOOR(confidence * ?), ? - 1) AS bucket,
                COUNT(*)                              AS total,
                COUNT(*) FILTER (WHERE claimed)       AS claimed
            FROM prefetch_events
            WHERE session_id = ?
            GROUP BY 1
            ORDER BY 1
            """,
            [N_BUCKETS, N_BUCKETS, self._session_id],
        ).fetchall()

        result = []
        for (bucket, total, claimed) in rows:
            bucket = int(bucket)
            c_low = bucket * bucket_width
            c_mid = c_low + bucket_width / 2.0
            claim_rate = claimed / total if total > 0 else 0.0
            result.append({
                "bucket": bucket,
                "c_low": c_low,
                "c_mid": c_mid,
                "total": int(total),
                "claimed": int(claimed),
                "claim_rate": claim_rate,
            })
        return result

    def _update_global_tau(self, domain_buckets: dict[str, list[dict]]) -> None:
        """
        Update the global τ based on aggregated data across all domains.
        For backward compatibility and fallback when no domain-specific τ exists.
        """
        # Aggregate all buckets across domains
        aggregated_buckets = {}
        for buckets in domain_buckets.values():
            for bucket_data in buckets:
                bucket_idx = bucket_data["bucket"]
                if bucket_idx not in aggregated_buckets:
                    aggregated_buckets[bucket_idx] = {
                        "bucket": bucket_idx,
                        "c_low": bucket_data["c_low"],
                        "c_mid": bucket_data["c_mid"],
                        "total": 0,
                        "claimed": 0,
                        "claim_rate": 0.0,
                    }
                aggregated_buckets[bucket_idx]["total"] += bucket_data["total"]
                aggregated_buckets[bucket_idx]["claimed"] += bucket_data["claimed"]

        # Recompute claim rates
        aggregated_list = []
        for bucket_data in aggregated_buckets.values():
            if bucket_data["total"] > 0:
                bucket_data["claim_rate"] = bucket_data["claimed"] / bucket_data["total"]
                aggregated_list.append(bucket_data)

        if not aggregated_list:
            return

        aggregated_list.sort(key=lambda x: x["bucket"])
        total_events = sum(b["total"] for b in aggregated_list)

        if total_events >= self._min_events:
            new_tau = self._estimate_tau(aggregated_list)
            current_tau = self._scheduler.tau

            if abs(new_tau - current_tau) >= 1e-6:
                self._scheduler.update_tau(new_tau)
                self._last_tau = new_tau
                logger.info(
                    "TauCalibrator: global τ updated {:.4f} → {:.4f} for session='{}' ({} events)",
                    current_tau, new_tau, self._session_id, total_events,
                )

    def _estimate_tau(self, buckets: list[dict]) -> float:
        """
        Estimate the optimal τ from per-bucket claim rates.

        Strategy:
        - Find the lowest-confidence bucket where claim_rate >= TARGET_PRP.
          That is the minimum confidence at which retrieval is reliably useful.
          Set τ = c_mid of that bucket.
        - If ALL buckets are below TARGET (every retrieval too speculative):
          raise τ by one bucket width (be more conservative).
        - If ALL buckets are at or above TARGET (even low-confidence useful):
          lower τ by one bucket width (be more aggressive).
        - Clamp result to [TAU_MIN, TAU_MAX].
        """
        bucket_width = 1.0 / N_BUCKETS
        current_tau = self._scheduler.tau
        qualifying = [b for b in buckets if b["claim_rate"] >= TARGET_PRP]

        if not qualifying:
            # All claim rates below target → too many spurious retrievals
            new_tau = current_tau + bucket_width
        elif len(qualifying) == len(buckets):
            # All buckets qualify → we can fire retrieval at lower confidence
            new_tau = qualifying[0]["c_mid"]  # lowest qualifying bucket
        else:
            # Normal case: find the lowest-confidence qualifying bucket
            new_tau = qualifying[0]["c_mid"]

        return max(TAU_MIN, min(TAU_MAX, new_tau))
