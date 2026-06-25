"""
Pipeline Coordinator — wires BSM → IIE → SRS → Retrieval → Buffer.

The coordinator is the single place that connects every pipeline component.
It has no domain logic. It orchestrates, routes, and logs.

Responsibilities
----------------
1. Receive a SignalVector from the Behavioral Signal Monitor (or directly).
2. Run it through the Intent Inference Engine → (q̂, c, ℓ).
3. Pass the triple to the Speculative Retrieval Scheduler → SchedulerDecision.
4. If RETRIEVE: query the RetrievalEngine, put results into the ContextBuffer,
   and log the prefetch event to the AnalyticsStore.
5. If WAIT: do nothing.

The coordinator does NOT push context to subscribers. Pushing is done by the
MCP server layer (_push_context in server.py) which reads from the buffer.

Privacy rule: no document content is logged. Only labels, c values, and
subscriber_ids appear in log output.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from apex.adapters.base import SignalVector
from apex.analytics.store import AnalyticsStore
from apex.buffer.context_buffer import ContextBuffer
from apex.inference.intent_engine import IntentEngine
from apex.retrieval.rrf import RetrievalEngine
from apex.scheduler.speculative import RetrievalAction, SpeculativeScheduler

if TYPE_CHECKING:
    from apex.monitor.live import LiveDisplay


class PipelineCoordinator:
    """
    Wires the six pipeline stages for a single pass of a behavioral signal.

    Parameters
    ----------
    retrieval_engine
        Populated RetrievalEngine (HNSW + BM25 + RRF).
    buffer
        Shared ContextBuffer instance. Populated partitions are read by the
        MCP push layer.
    intent_engine
        IntentEngine to infer (q̂, c, ℓ) from a SignalVector.
    scheduler
        SpeculativeScheduler that decides RETRIEVE or WAIT.
    store
        AnalyticsStore for DuckDB event logging (thesis metrics).
    subscriber_ids
        List of registered subscriber IDs. Each RETRIEVE fires a separate
        buffer.put() for every subscriber so they get isolated partitions.
    retrieval_k
        Number of top-k chunks to retrieve per subscriber per retrieval.
    session_id
        Run-level identifier written to every DuckDB prefetch event.
        Defaults to a fresh UUID so each daemon startup produces a new session.
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        buffer: ContextBuffer,
        intent_engine: IntentEngine,
        scheduler: SpeculativeScheduler,
        store: AnalyticsStore,
        subscriber_ids: Optional[list[str]] = None,
        retrieval_k: int = 5,
        push_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        session_id: Optional[str] = None,
        display: Optional["LiveDisplay"] = None,
    ) -> None:
        self._engine = retrieval_engine
        self._buffer = buffer
        self._iie = intent_engine
        self._scheduler = scheduler
        self._store = store
        self._subscriber_ids: list[str] = list(subscriber_ids or [])
        self._k = retrieval_k
        self._push_callback = push_callback
        self.session_id: str = session_id or str(uuid.uuid4())
        self._display = display
        # SSE broadcast callable injected by server.init_app()
        self._sse_broadcast: Optional[Callable[[str, Any], None]] = None

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def buffer(self) -> ContextBuffer:
        return self._buffer

    @property
    def store(self) -> AnalyticsStore:
        return self._store

    # ── Subscriber registration ───────────────────────────────────────────────

    def add_subscriber(self, subscriber_id: str) -> None:
        """Register a new subscriber to receive retrieved context."""
        if subscriber_id not in self._subscriber_ids:
            self._subscriber_ids.append(subscriber_id)
            logger.info("Coordinator: subscriber '{}' added", subscriber_id)

    def remove_subscriber(self, subscriber_id: str) -> None:
        """Unregister a subscriber and clear their buffer partition."""
        if subscriber_id in self._subscriber_ids:
            self._subscriber_ids.remove(subscriber_id)
            self._buffer.clear_subscriber(subscriber_id)
            logger.info("Coordinator: subscriber '{}' removed", subscriber_id)

    # ── Main pipeline entry point ─────────────────────────────────────────────

    async def process_signal(
        self,
        signal: SignalVector,
        subscriber_id: str,
    ) -> None:
        """
        Run one behavioral signal through the full pipeline for one subscriber.

        Parameters
        ----------
        signal
            Normalized SignalVector from the Behavioral Signal Monitor.
        subscriber_id
            The subscriber for whom context should be retrieved and buffered.
            Must have been registered via add_subscriber() or in __init__.
        """
        # Four-timestamp latency profiling for Claim 2
        t_signal = time.perf_counter()

        # ── Stage 1: Intent Inference ────────────────────────────────────────
        q_hat, c, label = await self._iie.infer(signal)
        t_iie = time.perf_counter()
        logger.debug(
            "Coordinator: IIE → label='{}' c={:.3f} urgency={}",
            label, c, signal.urgency_flag,
        )

        # Update terminal display + SSE dashboard with signal info
        if self._display is not None:
            self._display.update_signal(
                activity_type=signal.activity_type,
                velocity=signal.velocity_metric,
                urgency=signal.urgency_flag,
                label=label,
                confidence=c,
            )
        if self._sse_broadcast is not None:
            self._sse_broadcast("signal", {
                "activity_type": signal.activity_type,
                "velocity": round(signal.velocity_metric, 3),
                "urgency": signal.urgency_flag,
                "label": label,
                "confidence": round(c, 3),
            })

        # ── Stage 2: Scheduler decision ──────────────────────────────────────
        buffer_hit = self._buffer.has_warm_context(subscriber_id, label)
        decision = self._scheduler.decide(
            q_hat,
            c,
            label,
            urgency_flag=signal.urgency_flag,
            buffer_hit=buffer_hit,
        )
        logger.debug(
            "Coordinator: SRS → {} (τ={:.3f}) reason='{}'",
            decision.action.value, decision.tau_used, decision.reason,
        )

        # Update terminal display + SSE dashboard with scheduler decision
        if self._display is not None:
            self._display.update_pipeline(
                action=decision.action.value,
                label=label,
                tau=decision.tau_used,
                reason=decision.reason,
            )
        if self._sse_broadcast is not None:
            self._sse_broadcast("pipeline", {
                "action": decision.action.value,
                "label": label,
                "tau": round(decision.tau_used, 3),
                "reason": decision.reason,
            })

        if decision.action == RetrievalAction.WAIT:
            return

        # ── Stage 3: Retrieval ───────────────────────────────────────────────
        chunks = self._engine.search(q_hat, label=label, k=self._k)
        t_retrieval = time.perf_counter()
        logger.info(
            "Coordinator: retrieved {} chunk(s) for label='{}' subscriber='{}'",
            len(chunks), label, subscriber_id,
        )

        if not chunks:
            return

        # ── Stage 4: Buffer update ───────────────────────────────────────────
        self._buffer.put(subscriber_id, chunks)

        # Update buffer state in display + dashboard
        if self._display is not None or self._sse_broadcast is not None:
            partitions = {sid: len(self._buffer.get(sid)) for sid in self._subscriber_ids}
            if self._display is not None:
                self._display.update_buffer(partitions)
            if self._sse_broadcast is not None:
                self._sse_broadcast("buffer", partitions)

        # ── Stage 5: Analytics logging (with latency profiling) ──────────────
        event_id = self._store.log_prefetch(
            session_id=self.session_id,
            subscriber_id=subscriber_id,
            label=label,
            c=c,
            tau_used=decision.tau_used,
            t_signal=t_signal,
            t_iie=t_iie,
            t_retrieval=t_retrieval,
            t_push=None,  # Set after push completes
        )
        logger.debug("Coordinator: logged prefetch event_id={}", event_id)

        # Update metrics in display + dashboard after every prefetch
        if self._display is not None or self._sse_broadcast is not None:
            prp = self._store.compute_prp(self.session_id)
            ltc = self._store.compute_mean_ltc(self.session_id)
            if self._display is not None:
                self._display.update_metrics(prp=prp, mean_ltc=ltc)
                self._display.refresh()
            if self._sse_broadcast is not None:
                self._sse_broadcast("metrics", {"prp": prp, "ltc": ltc, "dps": None})

        # ── Stage 6: Push to subscriber + claim on delivery ──────────────────
        if self._push_callback is not None:
            delivered = await self._push_callback(subscriber_id)
            t_push = time.perf_counter()
            # Update timing information now that push is complete
            self._store.update_push_timing(event_id, t_push)
            if delivered:
                self._store.log_claim(event_id)
                logger.debug("Coordinator: claimed event_id={} (delivered via push)", event_id)
        else:
            t_push = time.perf_counter()  # Complete timing even if no push callback
            self._store.update_push_timing(event_id, t_push)

    async def process_signal_all(self, signal: SignalVector) -> None:
        """
        Run a signal through the pipeline for ALL registered subscribers.

        Called by the Behavioral Signal Monitor when a global event fires
        (e.g. a file change that may be relevant to any subscriber).

        Measures multi-subscriber overhead for Claim 3 analysis when N > 1.
        """
        subscriber_ids = list(self._subscriber_ids)
        if len(subscriber_ids) <= 1:
            # Single or no subscribers - use normal processing
            for subscriber_id in subscriber_ids:
                await self.process_signal(signal, subscriber_id=subscriber_id)
            return

        # Multi-subscriber case - measure formatting overhead
        # Process inference and retrieval once, then format for all subscribers
        t_signal = time.perf_counter()

        # ── Stage 1: Intent Inference (shared) ────────────────────────────────
        q_hat, c, label = await self._iie.infer(signal)
        t_iie = time.perf_counter()

        # ── Stage 2: Scheduler decision (check first subscriber for representative decision) ──
        buffer_hit = self._buffer.has_warm_context(subscriber_ids[0], label)
        decision = self._scheduler.decide(
            q_hat,
            c,
            label,
            urgency_flag=signal.urgency_flag,
            buffer_hit=buffer_hit,
        )

        if decision.action == RetrievalAction.WAIT:
            return

        # ── Stage 3: Retrieval (shared) ───────────────────────────────────────
        chunks = self._engine.search(q_hat, label=label, k=self._k)
        t_retrieval = time.perf_counter()

        if not chunks:
            return

        # ── Stage 4: Multi-subscriber formatting + buffering ─────────────────
        t_format_start = time.perf_counter()
        push_times = []
        event_ids = []

        for subscriber_id in subscriber_ids:
            # Buffer update per subscriber
            self._buffer.put(subscriber_id, chunks)

            # Log prefetch event per subscriber
            event_id = self._store.log_prefetch(
                session_id=self.session_id,
                subscriber_id=subscriber_id,
                label=label,
                c=c,
                tau_used=decision.tau_used,
                t_signal=t_signal,
                t_iie=t_iie,
                t_retrieval=t_retrieval,
                t_push=None,  # Will be updated after push
                multi_sub_overhead_ms=None,  # Will be computed after all pushes
                subscriber_count=len(subscriber_ids),
            )
            event_ids.append(event_id)

            # Push to subscriber if callback available
            if self._push_callback is not None:
                delivered = await self._push_callback(subscriber_id)
                t_push_individual = time.perf_counter()
                push_times.append(t_push_individual)

                # Update individual push timing
                self._store.update_push_timing(event_id, t_push_individual)
                if delivered:
                    self._store.log_claim(event_id)
            else:
                t_push_individual = time.perf_counter()
                push_times.append(t_push_individual)
                self._store.update_push_timing(event_id, t_push_individual)

        # Compute multi-subscriber overhead: max(t_push_N) - t_retrieval_complete
        if push_times:
            t_max_push = max(push_times)
            multi_sub_overhead_ms = (t_max_push - t_retrieval) * 1000.0

            # Update all events with the computed overhead
            for event_id in event_ids:
                self._store.con.execute(
                    "UPDATE prefetch_events SET multi_sub_overhead_ms = ? WHERE id = ?",
                    [multi_sub_overhead_ms, event_id]
                )

        # Update displays as in normal processing
        if self._display is not None or self._sse_broadcast is not None:
            partitions = {sid: len(self._buffer.get(sid)) for sid in self._subscriber_ids}
            if self._display is not None:
                self._display.update_buffer(partitions)
            if self._sse_broadcast is not None:
                self._sse_broadcast("buffer", partitions)

            prp = self._store.compute_prp(self.session_id)
            ltc = self._store.compute_mean_ltc(self.session_id)
            if self._display is not None:
                self._display.update_metrics(prp=prp, mean_ltc=ltc)
                self._display.refresh()
            if self._sse_broadcast is not None:
                self._sse_broadcast("metrics", {"prp": prp, "ltc": ltc, "dps": None})
