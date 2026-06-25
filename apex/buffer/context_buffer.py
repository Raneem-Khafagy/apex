"""
Context Buffer — TTL-managed per-subscriber chunk store.

Role in the pipeline
--------------------
Receives retrieved Chunk objects from the Retrieval Engine and holds them
for each subscriber until they are consumed (pushed via MCP) or expire.

The buffer serves two consumers:
  1. Speculative Retrieval Scheduler — queries has_warm_context() to decide
     whether to skip a retrieval (buffer_hit).
  2. MCP push layer — calls get() to retrieve warm chunks for delivery.

Design properties
-----------------
Per-subscriber isolation
    Internal state is partitioned by subscriber_id. get(A) can never return
    content put() for subscriber B. Enforced structurally — no conditional
    logic involved.

TTL expiry
    Every buffered chunk has an insertion timestamp and a TTL (seconds).
    Chunks older than their TTL are excluded from get() and has_warm_context().
    They are physically removed by evict_expired().

Dirty-flag (stale) mechanism
    When the SignalMonitor detects a file change at a given source path, it
    calls mark_stale(source). All buffered chunks whose chunk.source matches
    are flagged stale=True and immediately excluded from reads. They are
    physically removed on the next evict_expired() call.

get() is read-only
    get() filters and returns; it does not mutate the partition.
    evict_expired() is the only operation that physically removes entries.

max_chunks_per_subscriber
    When a put() would exceed the per-subscriber cap, the oldest buffered
    chunks (by insertion time) are dropped first to make room.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from apex.retrieval.rrf import Chunk


# ── Internal entry ───────────────────────────────────────────────────────────

@dataclass
class _BufferedChunk:
    """
    Internal wrapper around a Chunk with TTL and staleness metadata.
    Not exposed outside this module.
    """
    chunk: Chunk
    inserted_at: float        # time.time() at insertion
    ttl: float                # seconds until expiry
    stale: bool = False       # True = dirty-flagged by mark_stale()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.inserted_at) > self.ttl

    @property
    def is_usable(self) -> bool:
        """A chunk is usable if it is neither expired nor stale."""
        return not self.is_expired and not self.stale


# ── ContextBuffer ─────────────────────────────────────────────────────────────

class ContextBuffer:
    """
    TTL-managed, per-subscriber context cache.

    Parameters
    ----------
    default_ttl
        Default TTL in seconds for chunks whose put() call does not specify
        a TTL. Default 60 s per CLAUDE.md.
    max_chunks_per_subscriber
        Hard cap on buffered chunks per subscriber. When exceeded, the
        oldest chunks (by insertion time) are evicted to make room.
    """

    DEFAULT_TTL: float = 60.0
    DEFAULT_MAX_CHUNKS: int = 50

    def __init__(
        self,
        default_ttl: float = DEFAULT_TTL,
        max_chunks_per_subscriber: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        self._default_ttl = default_ttl
        self._max_chunks = max_chunks_per_subscriber
        # partition: subscriber_id → list[_BufferedChunk], insertion order
        self._partitions: dict[str, list[_BufferedChunk]] = {}

    @property
    def default_ttl(self) -> float:
        return self._default_ttl

    # ── Write path ────────────────────────────────────────────────────────────

    def put(
        self,
        subscriber_id: str,
        chunks: list[Chunk],
        ttl: Optional[float] = None,
    ) -> None:
        """
        Add chunks to a subscriber's partition.

        Parameters
        ----------
        subscriber_id
            Target subscriber. Content is isolated to this partition.
        chunks
            Chunk objects from the Retrieval Engine.
        ttl
            Seconds until these chunks expire. Defaults to self.default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        partition = self._partitions.setdefault(subscriber_id, [])
        for chunk in chunks:
            partition.append(_BufferedChunk(
                chunk=chunk,
                inserted_at=now,
                ttl=effective_ttl,
            ))

        # Enforce max_chunks: drop oldest entries if over the cap
        if len(partition) > self._max_chunks:
            # Sort by insertion time ascending, keep the most recent max_chunks
            partition.sort(key=lambda bc: bc.inserted_at)
            dropped = len(partition) - self._max_chunks
            del partition[:dropped]
            logger.debug(
                "ContextBuffer[{}]: dropped {} oldest chunk(s) to stay under cap={}",
                subscriber_id, dropped, self._max_chunks,
            )

        logger.debug(
            "ContextBuffer[{}]: put {} chunk(s), partition size={}",
            subscriber_id, len(chunks), len(partition),
        )

    # ── Read path ─────────────────────────────────────────────────────────────

    def get(self, subscriber_id: str) -> list[Chunk]:
        """
        Return all usable (non-expired, non-stale) chunks for a subscriber,
        sorted by RRF score descending.

        This method is read-only — it does not modify the partition.
        Call evict_expired() for physical cleanup.

        Parameters
        ----------
        subscriber_id
            The subscriber whose context to fetch.

        Returns
        -------
        Sorted list of Chunk objects. Empty if the subscriber is unknown or
        has no usable chunks.
        """
        partition = self._partitions.get(subscriber_id, [])
        usable = [bc.chunk for bc in partition if bc.is_usable]
        usable.sort(key=lambda c: c.score, reverse=True)
        return usable

    def has_warm_context(self, subscriber_id: str, label: str) -> bool:
        """
        Return True if the subscriber has at least one usable chunk for label.

        Used by the Speculative Retrieval Scheduler to determine buffer_hit.

        Parameters
        ----------
        subscriber_id
            Target subscriber.
        label
            Task context label (e.g. "debugging_python").
        """
        partition = self._partitions.get(subscriber_id, [])
        return any(
            bc.is_usable and bc.chunk.label == label
            for bc in partition
        )

    # ── Dirty-flag path ───────────────────────────────────────────────────────

    def mark_stale(self, source: str) -> int:
        """
        Mark all buffered chunks whose chunk.source == source as stale.

        Called by SignalMonitor when a file-change event fires.
        Stale chunks are immediately excluded from get() and has_warm_context()
        without waiting for TTL expiry.

        Parameters
        ----------
        source
            File path or document identifier that changed.

        Returns
        -------
        Number of chunks marked stale.
        """
        count = 0
        for partition in self._partitions.values():
            for bc in partition:
                if bc.chunk.source == source and not bc.stale:
                    bc.stale = True
                    count += 1
        if count:
            logger.info("ContextBuffer: marked {} chunk(s) stale for source='{}'", count, source)
        return count

    # ── Eviction path ─────────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """
        Physically remove all expired and stale entries from all partitions.

        Returns
        -------
        Total number of entries removed across all subscriber partitions.
        """
        total_removed = 0
        for subscriber_id, partition in self._partitions.items():
            before = len(partition)
            self._partitions[subscriber_id] = [
                bc for bc in partition if bc.is_usable
            ]
            removed = before - len(self._partitions[subscriber_id])
            if removed:
                logger.debug(
                    "ContextBuffer[{}]: evicted {} expired/stale chunk(s)",
                    subscriber_id, removed,
                )
            total_removed += removed
        return total_removed

    def clear_subscriber(self, subscriber_id: str) -> None:
        """
        Remove the entire partition for a subscriber.

        Called when a subscriber unregisters from the MCP server.

        Parameters
        ----------
        subscriber_id
            The subscriber to remove.
        """
        if subscriber_id in self._partitions:
            count = len(self._partitions.pop(subscriber_id))
            logger.info("ContextBuffer[{}]: cleared {} chunk(s)", subscriber_id, count)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Total buffered entries (including expired/stale, before eviction)."""
        return sum(len(p) for p in self._partitions.values())

    def subscriber_count(self) -> int:
        """Number of active subscriber partitions."""
        return len(self._partitions)
