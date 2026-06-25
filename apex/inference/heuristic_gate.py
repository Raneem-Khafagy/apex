"""
Heuristic Gate — fast (<1 ms) pattern matcher for the Intent Inference Engine.

For signals that match a known high-confidence pattern, the gate returns the
pre-computed intent triple (q̂, c, ℓ) immediately, bypassing the LLM entirely.

Design principles
-----------------
- Pure Python dict lookup — no I/O, no model calls.
- q̂ is ALWAYS a numpy ndarray injected via vector_table at construction time.
  It is never generated or synthesized here.
- Returns None for ambiguous or unknown signals; the IIE then falls through
  to the small LLM reasoner.
- Deterministic: same signal → same output every time.

Architecture rule: this file must never formulate a text string and treat it
as a retrieval query. q̂ must always be a dense vector.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from apex.adapters.base import SignalVector

# Type alias for the gate output triple
IntentTriple = tuple[np.ndarray, float, str]

# ── Pattern table ────────────────────────────────────────────────────────────
# Maps (activity_type, velocity_bucket) → (vector_table_key, confidence)
#
# velocity_bucket: "high" if velocity_metric >= VELOCITY_THRESHOLD, else "low".
# Only "high" and "low" buckets are recognized — the LLM handles the rest.

VELOCITY_THRESHOLD = 0.6

# Minimum velocity below which no heuristic fires (prevents idle noise).
MIN_VELOCITY = 0.3

_PATTERN_TABLE: dict[tuple[str, str], tuple[str, float]] = {
    ("writing",   "high"): ("writing_document",  0.90),
    ("writing",   "low"):  ("writing_document",  0.87),
    ("debugging", "high"): ("debugging_python",  0.90),
    ("debugging", "low"):  ("debugging_python",  0.87),
    ("reading",   "high"): ("reading_reference", 0.90),
    ("reading",   "low"):  ("reading_reference", 0.87),
    # "idle" is intentionally absent — idle signals always fall through to WAIT.
}


class HeuristicGate:
    """
    Fast pattern matcher for the IIE's first-pass gate.

    Parameters
    ----------
    vector_table
        Pre-computed embedding vectors keyed by task-context label.
        Must contain an entry for every label referenced in _PATTERN_TABLE.
        Built by IntentEngine at startup using the Ollama embedding model.
    """

    def __init__(self, vector_table: dict[str, np.ndarray]) -> None:
        self._vectors = vector_table

    def match(self, signal: SignalVector) -> Optional[IntentTriple]:
        """
        Attempt a fast pattern match on the signal.

        Returns
        -------
        (q̂, c, ℓ) if the signal matches a known pattern, else None.
        q̂ is always a numpy ndarray — never a string.
        """
        # Bail out immediately for idle or near-zero velocity — no retrieval needed
        if signal.activity_type == "idle" or signal.velocity_metric < MIN_VELOCITY:
            return None

        velocity_bucket = "high" if signal.velocity_metric >= VELOCITY_THRESHOLD else "low"
        key = (signal.activity_type, velocity_bucket)

        entry = _PATTERN_TABLE.get(key)
        if entry is None:
            logger.debug(
                "HeuristicGate miss: activity='{}' vel={:.2f}",
                signal.activity_type, signal.velocity_metric,
            )
            return None

        label, confidence = entry

        q_hat = self._vectors.get(label)
        if q_hat is None:
            logger.warning(
                "HeuristicGate: label '{}' not in vector_table — falling through to LLM",
                label,
            )
            return None

        logger.debug(
            "HeuristicGate hit: label='{}' c={:.2f}", label, confidence
        )
        return q_hat, confidence, label
