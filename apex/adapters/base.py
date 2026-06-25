"""
Signal Adapter base — domain-blind contracts.

SignalVector: the normalized signal schema emitted by every adapter.
SignalAdapter: the abstract base class every domain adapter must implement.

Architecture rule: this file must never contain domain-specific logic.
All six fields in SignalVector are fixed. To extend for a new domain,
add a field here with a sensible default — do NOT branch on it inside the core.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SignalVector:
    """
    Normalized signal snapshot emitted by a SignalAdapter at each timestep.

    Fields
    ------
    source_id
        Hash of active application + context identifier.
        Deterministic for the same (app, context) pair.
    content_hash
        Hash of open file names / context type — NEVER raw file content.
    activity_type
        Semantic activity label, e.g. "writing", "debugging", "anomaly_event".
    velocity_metric
        Normalized [0, 1] intensity of current activity.
        Semantics are domain-specific (keystrokes/sec, sensor deviation, etc.).
    temporal_proximity
        Normalized [0, 1] closeness to next event / deadline / maintenance window.
        Higher = more urgent from a time-sensitivity standpoint.
    urgency_flag
        When True, the Speculative Retrieval Scheduler forces τ → 0 and
        retrieves immediately. Reserved for safety-critical domains.
        Default False.
    """
    source_id: str
    content_hash: str
    activity_type: str
    velocity_metric: float
    temporal_proximity: float
    urgency_flag: bool = False


class SignalAdapter(ABC):
    """
    Abstract base class for all domain-specific signal adapters.

    Each adapter owns exactly one method: observe().
    observe() returns a normalized SignalVector snapshot of the current
    environment state, derived from OS-level metadata only.

    Adding a new domain = subclassing SignalAdapter and implementing observe().
    Zero changes to the core pipeline.
    """

    @abstractmethod
    def observe(self) -> SignalVector:
        """
        Collect a snapshot of the current environment state and return it
        as a normalized SignalVector.

        Implementations MUST:
        - Read only structural metadata (file names, app state, sensor readings).
        - Never read file content.
        - Return within a few milliseconds (this is called in a hot loop).
        - Be idempotent — calling observe() twice in quick succession should
          return signals that are semantically consistent.
        """
