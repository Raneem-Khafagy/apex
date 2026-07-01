#!/usr/bin/env python3
"""
Test multi-subscriber overhead measurement in PipelineCoordinator.

Validates that process_signal_all() correctly measures max(t_push_N) - t_retrieval_complete
when processing multiple subscribers for the same behavioral signal.
"""

import asyncio
import os
import tempfile
import time

import pytest

from apex.adapters.base import SignalVector
from apex.analytics.store import AnalyticsStore
from apex.buffer.context_buffer import ContextBuffer
from apex.inference.intent_engine import IntentEngine
from apex.pipeline.coordinator import PipelineCoordinator
from apex.retrieval.rrf import Chunk, RetrievalEngine
from apex.scheduler.speculative import RetrievalAction, SchedulerDecision, SpeculativeScheduler


# Deterministic seam doubles injected at the PipelineCoordinator's constructor
# boundaries (intent_engine=, scheduler=, engine=). These are the defined API
# seams described in ADR-001 ("Interface-boundary stubs"), NOT Ollama mocks — the
# LLM is never patched. They isolate the multi-subscriber push/format overhead,
# which is what this test measures, from inference and retrieval variability.

class FixedIntentEngine:
    """Deterministic IIE seam double — returns a fixed (q̂, c, ℓ) triple."""

    async def infer(self, signal: SignalVector):
        import numpy as np
        q_hat = np.ones(384, dtype=np.float32) / np.sqrt(384)  # fixed unit vector (all-MiniLM dim)
        c = 0.85
        label = signal.activity_type
        return q_hat, c, label


class AlwaysRetrieveScheduler:
    """Scheduler seam double that always decides to RETRIEVE."""

    def decide(self, q_hat, c, label, urgency_flag=False, buffer_hit=False):
        return SchedulerDecision(
            action=RetrievalAction.RETRIEVE,
            tau_used=0.65,
            reason="test seam - always retrieve"
        )


class StaticRetrievalEngine:
    """Retrieval seam double that returns a fixed set of chunks."""

    def search(self, q_hat, label=None, k=5):
        return [
            Chunk(
                chunk_id=f"{label}_chunk_{i}",
                text=f"Static content for {label} domain",
                source=f"docs/{label}_guide.md",
                label=label,
                score=0.9 - (i * 0.1)
            )
            for i in range(k)
        ]


@pytest.mark.asyncio
async def test_single_subscriber_uses_normal_processing():
    """Test that single subscriber case falls back to normal processing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        store = AnalyticsStore(db_path)
        buffer = ContextBuffer()
        coordinator = PipelineCoordinator(
            retrieval_engine=StaticRetrievalEngine(),
            buffer=buffer,
            intent_engine=FixedIntentEngine(),
            scheduler=AlwaysRetrieveScheduler(),
            store=store,
            subscriber_ids=["test_subscriber"],
            retrieval_k=3
        )

        signal = SignalVector(
            source_id="test_app",
            content_hash="hash123",
            activity_type="testing",
            velocity_metric=1.5,
            temporal_proximity=0.8,
            urgency_flag=False
        )

        await coordinator.process_signal_all(signal)

        # Should have one prefetch event with normal timing
        events = store.con.execute(
            "SELECT multi_sub_overhead_ms, subscriber_count FROM prefetch_events"
        ).fetchall()

        assert len(events) == 1
        overhead_ms, sub_count = events[0]

        # Single subscriber should not record multi-subscriber overhead
        assert overhead_ms is None  # Normal processing path doesn't set this
        assert sub_count is None    # Normal processing path doesn't set this


@pytest.mark.asyncio
async def test_multi_subscriber_overhead_measurement():
    """Test multi-subscriber overhead tracking with simulated push delays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        store = AnalyticsStore(db_path)
        buffer = ContextBuffer()

        # Push callback that records different simulated delivery times
        push_delays = {"subscriber_1": 0.01, "subscriber_2": 0.02, "subscriber_3": 0.03}

        async def recording_push_callback(subscriber_id: str) -> bool:
            """Simulate variable push delivery times."""
            await asyncio.sleep(push_delays[subscriber_id])
            return True  # Always successful delivery

        coordinator = PipelineCoordinator(
            retrieval_engine=StaticRetrievalEngine(),
            buffer=buffer,
            intent_engine=FixedIntentEngine(),
            scheduler=AlwaysRetrieveScheduler(),
            store=store,
            subscriber_ids=["subscriber_1", "subscriber_2", "subscriber_3"],
            retrieval_k=3,
            push_callback=recording_push_callback
        )

        signal = SignalVector(
            source_id="test_app",
            content_hash="hash456",
            activity_type="writing",
            velocity_metric=2.0,
            temporal_proximity=0.5,
            urgency_flag=False
        )

        start_time = time.perf_counter()
        await coordinator.process_signal_all(signal)
        total_time = time.perf_counter() - start_time

        # Should have three prefetch events (one per subscriber)
        events = store.con.execute(
            """
            SELECT subscriber_id, multi_sub_overhead_ms, subscriber_count, claimed
            FROM prefetch_events
            ORDER BY subscriber_id
            """
        ).fetchall()

        assert len(events) == 3

        # All events should have the same overhead and subscriber count
        for i, (sub_id, overhead_ms, sub_count, claimed) in enumerate(events):
            assert sub_id == f"subscriber_{i+1}"
            assert sub_count == 3
            assert overhead_ms is not None
            assert overhead_ms > 0  # Should have measurable overhead
            assert claimed is True  # push callback always succeeds

        # Overhead should include the maximum delay plus processing time
        # The measured overhead is max(t_push_N) - t_retrieval_complete
        # This includes asyncio sleep times plus buffer operations
        overhead_ms = events[0][1]  # All should be the same

        # Should be at least the maximum sleep time (30ms) plus processing overhead
        # Allow generous tolerance for system variance and processing time
        assert overhead_ms >= 30.0  # At least the max sleep delay
        assert overhead_ms <= 100.0  # Reasonable upper bound including processing


@pytest.mark.asyncio
async def test_multi_subscriber_shared_retrieval():
    """Test that multi-subscriber processing shares retrieval but logs per subscriber."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        store = AnalyticsStore(db_path)
        buffer = ContextBuffer()

        # Track retrieval calls
        retrieval_calls = []

        class TrackingRetrievalEngine(StaticRetrievalEngine):
            def search(self, q_hat, label=None, k=5):
                retrieval_calls.append((label, k))
                return super().search(q_hat, label, k)

        coordinator = PipelineCoordinator(
            retrieval_engine=TrackingRetrievalEngine(),
            buffer=buffer,
            intent_engine=FixedIntentEngine(),
            scheduler=AlwaysRetrieveScheduler(),
            store=store,
            subscriber_ids=["sub_1", "sub_2", "sub_3"],
            retrieval_k=5
        )

        signal = SignalVector(
            source_id="test_app",
            content_hash="hash789",
            activity_type="debugging",
            velocity_metric=1.0,
            temporal_proximity=0.9,
            urgency_flag=False
        )

        await coordinator.process_signal_all(signal)

        # Should have called retrieval exactly once (shared)
        assert len(retrieval_calls) == 1
        assert retrieval_calls[0] == ("debugging", 5)

        # But should have three prefetch events logged
        events = store.con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT subscriber_id) FROM prefetch_events"
        ).fetchone()
        total_events, unique_subscribers = events
        assert total_events == 3
        assert unique_subscribers == 3

        # All should have the same timing metadata (shared retrieval)
        timing_data = store.con.execute(
            """
            SELECT DISTINCT t_signal, t_iie, t_retrieval, label, confidence
            FROM prefetch_events
            """
        ).fetchall()
        assert len(timing_data) == 1  # All events should have identical shared timing


@pytest.mark.asyncio
async def test_overhead_scaling_analysis():
    """Test the compute_multi_subscriber_overhead() analysis method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        store = AnalyticsStore(db_path)
        session_id = "test_session"

        # Insert fixed data for different subscriber counts
        # Single subscriber (baseline)
        store.log_prefetch(
            session_id=session_id,
            subscriber_id="single_sub",
            label="testing",
            c=0.8,
            tau_used=0.65,
            multi_sub_overhead_ms=5.0,  # Minimal overhead
            subscriber_count=1
        )

        # Three subscribers
        for i in range(3):
            store.log_prefetch(
                session_id=session_id,
                subscriber_id=f"multi_sub_{i}",
                label="testing",
                c=0.8,
                tau_used=0.65,
                multi_sub_overhead_ms=25.0,  # Higher overhead
                subscriber_count=3
            )

        # Analyze overhead scaling
        overhead_stats = store.compute_multi_subscriber_overhead(session_id)

        # Should have data for both subscriber counts
        assert "overhead_by_count" in overhead_stats
        assert 1 in overhead_stats["overhead_by_count"]
        assert 3 in overhead_stats["overhead_by_count"]

        # Check scaling metrics
        assert overhead_stats["single_sub_mean_ms"] == 5.0
        assert overhead_stats["multi_sub_mean_ms"] == 25.0
        assert overhead_stats["scaling_factor"] == 5.0  # 25ms / 5ms = 5x overhead

        # Verify event counts
        assert overhead_stats["overhead_by_count"][1]["event_count"] == 1
        assert overhead_stats["overhead_by_count"][3]["event_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])