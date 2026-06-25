"""
Speculative Retrieval Scheduler (SRS).

Implements the Goldilocks timing principle (Seo et al., CHI 2025):
retrieve early enough that context is ready when the user needs it,
but not so early that retrieved context expires unused.

Role in pipeline
----------------
Sits between the Intent Inference Engine and the Retrieval Engine.
Receives the intent triple (q̂, c, ℓ) and a buffer-hit signal, and
decides whether retrieval should fire NOW or WAIT.

The SRS does NOT perform retrieval. It returns a SchedulerDecision.
The pipeline coordinator checks decision.action and, if RETRIEVE,
passes (q̂, ℓ) to the retrieval engine.

τ is never hardcoded
---------------------
τ is a mutable, per-user calibrated threshold. The default (0.65) is
used only until the calibration feedback loop (analytics/store.py)
provides a learned value. Call update_tau() when the calibration store
emits a new estimate.

Priority ladder for effective τ
---------------------------------
1. urgency_flag = True  →  τ = 0.00  (unconditional; safety-critical)
2. battery_saver = True →  τ = 0.80  (hard override; reduce wasted retrieval)
3. otherwise            →  τ = self._tau  (calibrated, default 0.65)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    pass


class RetrievalAction(Enum):
    """Possible decisions the SRS can emit."""
    RETRIEVE = "RETRIEVE"
    WAIT = "WAIT"


@dataclass(frozen=True)
class SchedulerDecision:
    """
    Immutable decision record returned by SpeculativeScheduler.decide().

    Attributes
    ----------
    action
        RETRIEVE: fire retrieval now using (q̂, ℓ) held by the caller.
        WAIT: do nothing; re-evaluate on the next signal.
    tau_used
        The effective τ value that was in effect when this decision was made.
        Logged to analytics/store.py for calibration curve fitting.
    reason
        Human-readable trace of why this decision was made.
        Used in structured logs and the live terminal display.
    """
    action: RetrievalAction
    tau_used: float
    reason: str


class SpeculativeScheduler:
    """
    Goldilocks timing scheduler for speculative context retrieval.

    Parameters
    ----------
    tau
        Initial confidence threshold. Retrieval fires when c >= τ.
        Must be in [0.0, 1.0]. Defaults to 0.65.
        Will be overwritten by calibration feedback over time.
    """

    DEFAULT_TAU: float = 0.65
    BATTERY_SAVER_TAU: float = 0.80

    def __init__(self, tau: float = DEFAULT_TAU) -> None:
        self._validate_tau(tau)
        self._tau: float = tau                          # Global fallback τ
        self._domain_tau: dict[str, float] = {}         # Per-domain τ values
        self._battery_saver: bool = False

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def tau(self) -> float:
        """Current calibrated base threshold (excluding overrides)."""
        return self._tau

    def get_tau(self, label: str) -> float:
        """
        Get the τ threshold for a specific domain label.

        Returns the domain-specific τ if available, otherwise falls back
        to the global τ. This is the method used by the calibrator and
        coordinator to get domain-specific thresholds.

        Parameters
        ----------
        label
            Domain label (e.g., "debugging_python", "drafting_research")

        Returns
        -------
        Domain-specific τ or global τ fallback
        """
        return self._domain_tau.get(label, self._tau)

    def decide(
        self,
        q_hat: np.ndarray,
        c: float,
        label: str,
        urgency_flag: bool = False,
        buffer_hit: bool = False,
    ) -> SchedulerDecision:
        """
        Apply the Goldilocks policy and return a retrieval decision.

        Parameters
        ----------
        q_hat
            Dense intent vector from the IIE. Passed through unchanged;
            the SRS does not read, copy, or transform it.
        c
            Confidence score from the IIE, ∈ [0, 1].
        label
            Task context label from the IIE, e.g. "debugging_python".
        urgency_flag
            When True, retrieval is unconditional. τ is forced to 0.0.
            This is the safety-critical override for factory/medical/emergency domains.
        buffer_hit
            When True, the ContextBuffer already has warm context for this
            subscriber. Retrieval would be redundant — return WAIT.
        """
        effective_tau = self._effective_tau(urgency_flag, label)

        # ── Priority 1: urgency override ────────────────────────────────────
        # Must be checked first, before any τ or buffer comparison.
        if urgency_flag:
            reason = f"urgency_flag=True → τ forced to 0.0 (c={c:.3f})"
            logger.debug("SRS RETRIEVE [urgency]: {}", reason)
            return SchedulerDecision(
                action=RetrievalAction.RETRIEVE,
                tau_used=effective_tau,
                reason=reason,
            )

        # ── Priority 2: normal policy ────────────────────────────────────────
        if c >= effective_tau and not buffer_hit:
            reason = (
                f"c={c:.3f} >= τ={effective_tau:.3f}, buffer_miss, label='{label}'"
            )
            logger.debug("SRS RETRIEVE: {}", reason)
            return SchedulerDecision(
                action=RetrievalAction.RETRIEVE,
                tau_used=effective_tau,
                reason=reason,
            )

        # ── Priority 3: WAIT ─────────────────────────────────────────────────
        if buffer_hit:
            reason = f"buffer_hit=True, label='{label}' already warm (c={c:.3f})"
        else:
            reason = f"c={c:.3f} < τ={effective_tau:.3f}, label='{label}'"
        logger.debug("SRS WAIT: {}", reason)
        return SchedulerDecision(
            action=RetrievalAction.WAIT,
            tau_used=effective_tau,
            reason=reason,
        )

    def update_tau(self, new_tau: float) -> None:
        """
        Update the global calibrated confidence threshold.

        Called by the calibration feedback loop when a new global τ estimate
        is available. Takes effect immediately for domains without specific τ.

        Parameters
        ----------
        new_tau
            New threshold value. Must be in [0.0, 1.0].

        Raises
        ------
        ValueError
            If new_tau is outside [0.0, 1.0].
        """
        self._validate_tau(new_tau)
        old = self._tau
        self._tau = new_tau
        logger.info("SRS global τ updated: {:.3f} → {:.3f}", old, new_tau)

    def update_domain_tau(self, label: str, new_tau: float) -> None:
        """
        Update the domain-specific calibrated confidence threshold.

        Called by the calibration feedback loop when a new per-domain τ
        estimate is available. Takes effect immediately for that domain.

        Parameters
        ----------
        label
            Domain label (e.g., "debugging_python", "drafting_research")
        new_tau
            New threshold value. Must be in [0.0, 1.0].

        Raises
        ------
        ValueError
            If new_tau is outside [0.0, 1.0].
        """
        self._validate_tau(new_tau)
        old = self._domain_tau.get(label, self._tau)
        self._domain_tau[label] = new_tau
        logger.info("SRS domain τ updated for '{}': {:.3f} → {:.3f}", label, old, new_tau)

    def set_battery_saver(self, enabled: bool) -> None:
        """
        Activate or deactivate battery saver mode.

        When enabled, τ is hard-overridden to BATTERY_SAVER_TAU (0.80),
        reducing the rate of speculative retrievals to conserve power.
        Does not affect urgency_flag override.

        Parameters
        ----------
        enabled
            True to activate battery saver, False to restore calibrated τ.
        """
        self._battery_saver = enabled
        state = "ON" if enabled else "OFF"
        logger.info(
            "SRS battery saver {}: τ = {:.2f}",
            state,
            self.BATTERY_SAVER_TAU if enabled else self._tau,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _effective_tau(self, urgency_flag: bool, label: str) -> float:
        """
        Resolve the τ value in effect for this decision tick.

        Uses domain-specific τ if available, otherwise falls back to global τ.
        """
        if urgency_flag:
            return 0.0
        if self._battery_saver:
            return self.BATTERY_SAVER_TAU
        return self.get_tau(label)

    @staticmethod
    def _validate_tau(tau: float) -> None:
        if not (0.0 <= tau <= 1.0):
            raise ValueError(
                f"τ must be in [0.0, 1.0], got {tau}. "
                "Use urgency_flag=True for unconditional retrieval."
            )
