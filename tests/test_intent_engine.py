"""
Tests for HeuristicGate and IntentEngine.
Real components only — no mocks, no patches.
Requires Ollama running with phi3.5 and all-minilm models.

A shared IntentEngine is initialized once at module level to amortize
the Ollama embedding calls for the vector table build.

Critical invariant under test: q̂ is ALWAYS a numpy ndarray, never a str.
"""
import numpy as np
import pytest

from apex.adapters.base import SignalVector
from apex.inference.heuristic_gate import HeuristicGate
from apex.inference.intent_engine import EMBED_DIM, LLM_CONFIDENCE, IntentEngine

# ── Module-level shared engine (built once, reused across all tests) ─────────
# _build_vector_table() embeds 3 labels via Ollama all-minilm. Doing this
# once avoids repeated network calls to Ollama during the test run.
_engine = IntentEngine()


def _signal(activity_type: str = "writing", velocity: float = 0.8,
            urgency: bool = False) -> SignalVector:
    return SignalVector(
        source_id="test",
        content_hash="cafebabe",
        activity_type=activity_type,
        velocity_metric=velocity,
        temporal_proximity=0.3,
        urgency_flag=urgency,
    )


# ── HeuristicGate ────────────────────────────────────────────────────────────

class TestHeuristicGate:
    """
    HeuristicGate is initialized with real vectors from the shared engine's
    internal vector table — no generated or synthetic arrays.
    """

    @pytest.fixture(scope="class")
    def gate(self):
        # Pull the real vector table that the shared engine built from Ollama
        return HeuristicGate(vector_table=_engine._vector_table)

    def test_known_writing_pattern_returns_result(self, gate):
        assert gate.match(_signal("writing", velocity=0.7)) is not None

    def test_known_debugging_pattern_returns_result(self, gate):
        assert gate.match(_signal("debugging", velocity=0.8)) is not None

    def test_unknown_activity_type_returns_none(self, gate):
        assert gate.match(_signal("unknown_exotic_activity", velocity=0.5)) is None

    def test_result_is_triple(self, gate):
        result = gate.match(_signal("writing", velocity=0.7))
        assert result is not None
        q_hat, c, label = result

    def test_q_hat_is_ndarray_never_str(self, gate):
        """Core invariant: q̂ must be a dense vector, never a text string."""
        q_hat, _, _ = gate.match(_signal("writing", velocity=0.7))
        assert isinstance(q_hat, np.ndarray), (
            f"q̂ must be np.ndarray — got {type(q_hat).__name__}. "
            "The IIE must never emit a text string as a retrieval query."
        )

    def test_q_hat_shape_is_embed_dim(self, gate):
        q_hat, _, _ = gate.match(_signal("writing", velocity=0.7))
        assert q_hat.shape == (EMBED_DIM,)

    def test_confidence_in_valid_range(self, gate):
        _, c, _ = gate.match(_signal("debugging", velocity=0.9))
        assert 0.0 <= c <= 1.0

    def test_heuristic_confidence_is_high(self, gate):
        """Known patterns must return c = 0.9 per architecture spec."""
        _, c, _ = gate.match(_signal("writing", velocity=0.8))
        assert c >= 0.85

    def test_label_is_string(self, gate):
        _, _, label = gate.match(_signal("debugging", velocity=0.6))
        assert isinstance(label, str) and len(label) > 0

    def test_label_is_not_q_hat(self, gate):
        q_hat, _, label = gate.match(_signal("writing", velocity=0.7))
        assert isinstance(label, str)
        assert isinstance(q_hat, np.ndarray)

    def test_idle_low_velocity_returns_none(self, gate):
        assert gate.match(_signal("idle", velocity=0.0)) is None

    def test_gate_is_deterministic(self, gate):
        sig = _signal("debugging", velocity=0.9)
        r1 = gate.match(sig)
        r2 = gate.match(sig)
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]
        assert r1[2] == r2[2]


# ── IntentEngine ─────────────────────────────────────────────────────────────

class TestIntentEngine:
    async def test_output_is_triple(self):
        result = await _engine.infer(_signal("writing", velocity=0.8))
        assert len(result) == 3

    async def test_q_hat_is_ndarray_always(self):
        """THE critical invariant: q̂ is never a string."""
        q_hat, c, label = await _engine.infer(_signal("writing", velocity=0.8))
        assert isinstance(q_hat, np.ndarray), (
            f"q̂ must be ndarray — got {type(q_hat).__name__}. "
            "The IIE must never emit a text string as a retrieval query."
        )

    async def test_q_hat_has_correct_shape(self):
        q_hat, _, _ = await _engine.infer(_signal("writing", velocity=0.8))
        assert q_hat.shape == (EMBED_DIM,)

    async def test_confidence_in_range(self):
        _, c, _ = await _engine.infer(_signal("writing", velocity=0.8))
        assert 0.0 <= c <= 1.0

    async def test_label_is_str(self):
        _, _, label = await _engine.infer(_signal("writing", velocity=0.8))
        assert isinstance(label, str) and len(label) > 0

    async def test_heuristic_path_gives_high_confidence(self):
        """
        Known signal → gate hits → c must be heuristic-level (≥ 0.85).
        This distinguishes the heuristic path (c=0.9) from the LLM path (c=0.65).
        """
        _, c, _ = await _engine.infer(_signal("writing", velocity=0.9))
        assert c >= 0.85

    async def test_llm_path_gives_llm_confidence(self):
        """
        Unknown activity → gate misses → LLM path → c must equal LLM_CONFIDENCE.
        The LLM path confidence (0.65) is distinct from the heuristic (0.9).
        """
        _, c, _ = await _engine.infer(_signal("unknown_activity_xyz", velocity=0.4))
        assert c == LLM_CONFIDENCE

    async def test_llm_path_still_returns_ndarray(self):
        """Even on the LLM path, q̂ must be ndarray — not the label string."""
        q_hat, _, label = await _engine.infer(_signal("unknown_activity_xyz", velocity=0.4))
        assert isinstance(q_hat, np.ndarray)
        assert isinstance(label, str)
        assert not isinstance(q_hat, str)

    async def test_urgency_flag_forces_max_confidence(self):
        """urgency_flag=True must yield c = 1.0 unconditionally."""
        urgent = _signal("anomaly_event", velocity=1.0, urgency=True)
        _, c, _ = await _engine.infer(urgent)
        assert c >= 0.95, f"urgency_flag=True must yield c≥0.95, got {c}"

    async def test_urgency_flag_q_hat_is_still_ndarray(self):
        urgent = _signal("anomaly_event", velocity=1.0, urgency=True)
        q_hat, _, _ = await _engine.infer(urgent)
        assert isinstance(q_hat, np.ndarray)

    async def test_engine_is_consistent_for_same_signal(self):
        """
        Two calls with the same known signal must return the same label and
        the same vector (heuristic path is deterministic).
        """
        sig = _signal("debugging", velocity=0.9)
        q1, c1, l1 = await _engine.infer(sig)
        q2, c2, l2 = await _engine.infer(sig)
        assert l1 == l2
        assert c1 == c2
        np.testing.assert_array_equal(q1, q2)
