"""
Tests for PipelineCoordinator.

The coordinator wires:
  SignalVector → IIE → SRS → RetrievalEngine → ContextBuffer

Tests use:
- Real IntentEngine (Ollama all-minilm + phi3.5)
- Real RetrievalEngine with pre-seeded chunks
- Real ContextBuffer
- Real SpeculativeScheduler

No mocks.
"""
from __future__ import annotations

import asyncio

import numpy as np
import ollama
import pytest

from apex.adapters.base import SignalVector
from apex.analytics.store import AnalyticsStore
from apex.buffer.context_buffer import ContextBuffer
from apex.inference.intent_engine import IntentEngine
from apex.pipeline.coordinator import PipelineCoordinator
from apex.retrieval.rrf import Chunk, RetrievalEngine
from apex.scheduler.speculative import SpeculativeScheduler


# ── Shared fixtures ──────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    r = ollama.embed(model="all-minilm", input=text)
    v = np.array(r.embeddings[0], dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def _seeded_engine() -> RetrievalEngine:
    """Return a RetrievalEngine with one debugging and one writing chunk."""
    engine = RetrievalEngine()
    for chunk_id, text, label in [
        ("debug_1", "Python traceback: AttributeError in module foo", "debugging_python"),
        ("write_1", "Draft introduction for the research paper", "writing_document"),
    ]:
        chunk = Chunk(chunk_id=chunk_id, text=text, source=f"/{chunk_id}.txt", label=label)
        engine.add_chunk(chunk, _embed(text))
    return engine


def _coordinator(subscriber_id: str = "sub1") -> PipelineCoordinator:
    engine = _seeded_engine()
    buffer = ContextBuffer()
    iie = IntentEngine()
    scheduler = SpeculativeScheduler()
    store = AnalyticsStore(db_path=":memory:")
    return PipelineCoordinator(
        retrieval_engine=engine,
        buffer=buffer,
        intent_engine=iie,
        scheduler=scheduler,
        store=store,
        subscriber_ids=[subscriber_id],
    )


# ── Construction ─────────────────────────────────────────────────────────────

class TestConstruction:
    def test_coordinator_constructs(self):
        coord = _coordinator()
        assert coord is not None

    def test_coordinator_exposes_buffer(self):
        coord = _coordinator()
        assert isinstance(coord.buffer, ContextBuffer)


# ── process_signal ────────────────────────────────────────────────────────────

class TestProcessSignal:
    @pytest.mark.asyncio
    async def test_high_confidence_signal_populates_buffer(self):
        """A heuristic-path signal (c=0.9) should trigger retrieval."""
        coord = _coordinator("sub1")
        signal = SignalVector(
            source_id="app_vscode",
            content_hash="abc123",
            activity_type="debugging",
            velocity_metric=0.8,      # high velocity → heuristic path, c=0.9
            temporal_proximity=0.5,
            urgency_flag=False,
        )
        await coord.process_signal(signal, subscriber_id="sub1")
        chunks = coord.buffer.get("sub1")
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_urgency_flag_always_retrieves(self):
        """urgency_flag=True must fire retrieval regardless of confidence."""
        coord = _coordinator("sub1")
        signal = SignalVector(
            source_id="factory_sensor",
            content_hash="xyz",
            activity_type="anomaly_event",
            velocity_metric=0.1,   # low — would normally WAIT
            temporal_proximity=0.0,
            urgency_flag=True,
        )
        await coord.process_signal(signal, subscriber_id="sub1")
        chunks = coord.buffer.get("sub1")
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_low_confidence_signal_does_not_override_buffer_hit(self):
        """If buffer already has warm context, SRS should WAIT (buffer_hit=True)."""
        coord = _coordinator("sub1")

        # Pre-warm the buffer with a debugging chunk
        warm_chunk = Chunk(
            chunk_id="warm_1",
            text="Existing context for debugging",
            source="/warm.py",
            label="debugging_python",
            score=0.9,
        )
        coord.buffer.put("sub1", [warm_chunk])

        signal = SignalVector(
            source_id="app_vscode",
            content_hash="abc",
            activity_type="debugging",
            velocity_metric=0.8,
            temporal_proximity=0.5,
            urgency_flag=False,
        )
        # Buffer hit → SRS should WAIT, no new chunks added from retrieval
        before = len(coord.buffer.get("sub1"))
        await coord.process_signal(signal, subscriber_id="sub1")
        after = len(coord.buffer.get("sub1"))
        # Buffer should still have chunks (at minimum the pre-warmed one)
        assert after >= 1
        # When there is a buffer hit, no new retrieval happens — count stays same
        assert after == before

    @pytest.mark.asyncio
    async def test_process_signal_logs_prefetch_to_analytics(self):
        """Every RETRIEVE decision must log a prefetch event."""
        coord = _coordinator("sub1")
        signal = SignalVector(
            source_id="app_vscode",
            content_hash="abc",
            activity_type="debugging",
            velocity_metric=0.8,
            temporal_proximity=0.5,
            urgency_flag=False,
        )
        await coord.process_signal(signal, subscriber_id="sub1")
        # At least one prefetch logged for this session (keyed by coordinator.session_id)
        prp = coord.store.compute_prp(coord.session_id)
        assert prp is not None

    @pytest.mark.asyncio
    async def test_multiple_subscribers_isolated(self):
        """Each subscriber must get their own buffer partition."""
        engine = _seeded_engine()
        buffer = ContextBuffer()
        iie = IntentEngine()
        scheduler = SpeculativeScheduler()
        store = AnalyticsStore(db_path=":memory:")
        coord = PipelineCoordinator(
            retrieval_engine=engine,
            buffer=buffer,
            intent_engine=iie,
            scheduler=scheduler,
            store=store,
            subscriber_ids=["subA", "subB"],
        )

        signal = SignalVector(
            source_id="app_vscode",
            content_hash="abc",
            activity_type="debugging",
            velocity_metric=0.8,
            temporal_proximity=0.5,
            urgency_flag=False,
        )

        await coord.process_signal(signal, subscriber_id="subA")

        # subA has chunks, subB has none
        assert len(coord.buffer.get("subA")) >= 1
        assert len(coord.buffer.get("subB")) == 0
