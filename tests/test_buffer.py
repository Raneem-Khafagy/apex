"""
Tests for ContextBuffer.

Real Chunk objects, real time.time(). No mocks, no patches.

Short TTLs (0.05 s) + time.sleep() are used to trigger expiry in the TTL
tests. These sleeps are intentionally tiny — the goal is correctness, not speed.

Subscriber IDs used throughout:
    "alice"  — primary test subscriber
    "bob"    — secondary subscriber for isolation tests
"""
import time

import pytest

from apex.buffer.context_buffer import ContextBuffer
from apex.retrieval.rrf import Chunk


# ── Helpers ──────────────────────────────────────────────────────────────────

def _chunk(
    chunk_id: str = "c1",
    text: str = "some text",
    source: str = "doc.md",
    label: str = "debugging_python",
    score: float = 0.8,
) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source=source,
                 label=label, score=score)


def _chunks(n: int, label: str = "debugging_python", source: str = "doc.md") -> list[Chunk]:
    return [_chunk(chunk_id=f"c{i}", label=label, source=source, score=1.0 - i * 0.05)
            for i in range(n)]


# ── Basic put / get ──────────────────────────────────────────────────────────

class TestPutGet:
    def test_put_then_get_returns_chunk(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk()])
        results = buf.get("alice")
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    def test_get_unknown_subscriber_returns_empty(self):
        buf = ContextBuffer()
        assert buf.get("nobody") == []

    def test_get_returns_all_usable_chunks(self):
        buf = ContextBuffer()
        buf.put("alice", _chunks(3))
        assert len(buf.get("alice")) == 3

    def test_get_returns_chunks_sorted_by_score_descending(self):
        buf = ContextBuffer()
        c_low  = _chunk("low",  score=0.3)
        c_high = _chunk("high", score=0.9)
        c_mid  = _chunk("mid",  score=0.6)
        buf.put("alice", [c_low, c_high, c_mid])
        scores = [c.score for c in buf.get("alice")]
        assert scores == sorted(scores, reverse=True)

    def test_put_multiple_times_accumulates(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("c1")])
        buf.put("alice", [_chunk("c2")])
        assert len(buf.get("alice")) == 2

    def test_get_returns_chunk_text(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(text="the actual content")])
        assert buf.get("alice")[0].text == "the actual content"


# ── Subscriber isolation ─────────────────────────────────────────────────────

class TestSubscriberIsolation:
    def test_alice_cannot_see_bobs_chunks(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("alice_chunk")])
        buf.put("bob",   [_chunk("bob_chunk")])
        alice_ids = {c.chunk_id for c in buf.get("alice")}
        assert "bob_chunk" not in alice_ids

    def test_bob_cannot_see_alices_chunks(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("alice_chunk")])
        buf.put("bob",   [_chunk("bob_chunk")])
        bob_ids = {c.chunk_id for c in buf.get("bob")}
        assert "alice_chunk" not in bob_ids

    def test_clear_alice_does_not_affect_bob(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("a1")])
        buf.put("bob",   [_chunk("b1")])
        buf.clear_subscriber("alice")
        assert len(buf.get("alice")) == 0
        assert len(buf.get("bob")) == 1

    def test_evict_on_one_subscriber_does_not_affect_other(self):
        buf = ContextBuffer(default_ttl=0.05)
        buf.put("alice", [_chunk("a1")])
        buf.put("bob",   [_chunk("b1")], ttl=60.0)  # bob's chunks are long-lived
        time.sleep(0.1)
        buf.evict_expired()
        assert len(buf.get("bob")) == 1


# ── TTL expiry ───────────────────────────────────────────────────────────────

class TestTTLExpiry:
    def test_chunk_visible_before_ttl(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk()], ttl=60.0)
        assert len(buf.get("alice")) == 1

    def test_chunk_hidden_after_ttl(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk()], ttl=0.05)
        time.sleep(0.1)
        assert len(buf.get("alice")) == 0

    def test_mixed_ttl_only_fresh_returned(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("stale_c")], ttl=0.05)
        time.sleep(0.1)
        buf.put("alice", [_chunk("fresh_c")], ttl=60.0)
        results = buf.get("alice")
        ids = {c.chunk_id for c in results}
        assert "fresh_c" in ids
        assert "stale_c" not in ids

    def test_default_ttl_is_sixty_seconds(self):
        buf = ContextBuffer()
        assert buf.default_ttl == 60.0

    def test_custom_default_ttl(self):
        buf = ContextBuffer(default_ttl=30.0)
        assert buf.default_ttl == 30.0

    def test_per_call_ttl_overrides_default(self):
        buf = ContextBuffer(default_ttl=60.0)
        buf.put("alice", [_chunk()], ttl=0.05)
        time.sleep(0.1)
        assert len(buf.get("alice")) == 0  # short ttl overrode the 60s default


# ── Dirty-flag (mark_stale) ──────────────────────────────────────────────────

class TestDirtyFlag:
    def test_chunk_hidden_after_mark_stale(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(source="notes.py")])
        buf.mark_stale("notes.py")
        assert len(buf.get("alice")) == 0

    def test_chunk_from_other_source_unaffected(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("keep", source="other.py")])
        buf.put("alice", [_chunk("drop", source="notes.py")])
        buf.mark_stale("notes.py")
        results = buf.get("alice")
        ids = {c.chunk_id for c in results}
        assert "keep" in ids
        assert "drop" not in ids

    def test_mark_stale_affects_all_subscribers_with_same_source(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(source="shared.py")])
        buf.put("bob",   [_chunk(source="shared.py")])
        buf.mark_stale("shared.py")
        assert len(buf.get("alice")) == 0
        assert len(buf.get("bob")) == 0

    def test_mark_stale_returns_count_marked(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("c1", source="file.py"),
                          _chunk("c2", source="file.py"),
                          _chunk("c3", source="other.py")])
        count = buf.mark_stale("file.py")
        assert count == 2

    def test_mark_stale_unknown_source_returns_zero(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(source="doc.md")])
        count = buf.mark_stale("nonexistent.py")
        assert count == 0

    def test_stale_chunk_still_removed_by_evict(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(source="file.py")])
        buf.mark_stale("file.py")
        removed = buf.evict_expired()
        assert removed >= 1
        assert len(buf.get("alice")) == 0


# ── has_warm_context ─────────────────────────────────────────────────────────

class TestHasWarmContext:
    def test_true_when_usable_chunk_exists_for_label(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(label="debugging_python")])
        assert buf.has_warm_context("alice", "debugging_python") is True

    def test_false_for_unknown_subscriber(self):
        buf = ContextBuffer()
        assert buf.has_warm_context("nobody", "debugging_python") is False

    def test_false_for_different_label(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(label="debugging_python")])
        assert buf.has_warm_context("alice", "writing_document") is False

    def test_false_after_ttl_expiry(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(label="debugging_python")], ttl=0.05)
        time.sleep(0.1)
        assert buf.has_warm_context("alice", "debugging_python") is False

    def test_false_after_mark_stale(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(label="debugging_python", source="code.py")])
        buf.mark_stale("code.py")
        assert buf.has_warm_context("alice", "debugging_python") is False

    def test_true_only_for_owning_subscriber(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(label="debugging_python")])
        assert buf.has_warm_context("alice", "debugging_python") is True
        assert buf.has_warm_context("bob",   "debugging_python") is False

    def test_true_when_at_least_one_usable_chunk(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("c1", label="debugging_python")], ttl=0.05)
        buf.put("alice", [_chunk("c2", label="debugging_python")], ttl=60.0)
        time.sleep(0.1)
        # c1 expired but c2 is still warm
        assert buf.has_warm_context("alice", "debugging_python") is True


# ── evict_expired ────────────────────────────────────────────────────────────

class TestEvictExpired:
    def test_evict_removes_expired_entries(self):
        buf = ContextBuffer()
        buf.put("alice", _chunks(3), ttl=0.05)
        time.sleep(0.1)
        removed = buf.evict_expired()
        assert removed == 3

    def test_evict_does_not_remove_fresh_entries(self):
        buf = ContextBuffer()
        buf.put("alice", _chunks(3), ttl=60.0)
        removed = buf.evict_expired()
        assert removed == 0
        assert len(buf.get("alice")) == 3

    def test_evict_removes_stale_entries(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk(source="file.py")])
        buf.mark_stale("file.py")
        removed = buf.evict_expired()
        assert removed >= 1

    def test_evict_returns_correct_total_across_subscribers(self):
        buf = ContextBuffer()
        buf.put("alice", _chunks(2), ttl=0.05)
        buf.put("bob",   _chunks(3), ttl=0.05)
        time.sleep(0.1)
        removed = buf.evict_expired()
        assert removed == 5

    def test_evict_mixed_keeps_fresh(self):
        buf = ContextBuffer()
        buf.put("alice", [_chunk("expire")], ttl=0.05)
        buf.put("alice", [_chunk("keep")],   ttl=60.0)
        time.sleep(0.1)
        buf.evict_expired()
        ids = {c.chunk_id for c in buf.get("alice")}
        assert "keep" in ids
        assert "expire" not in ids


# ── clear_subscriber ─────────────────────────────────────────────────────────

class TestClearSubscriber:
    def test_clear_removes_all_chunks(self):
        buf = ContextBuffer()
        buf.put("alice", _chunks(5))
        buf.clear_subscriber("alice")
        assert buf.get("alice") == []

    def test_clear_unknown_subscriber_is_safe(self):
        buf = ContextBuffer()
        buf.clear_subscriber("nobody")  # must not raise

    def test_clear_subscriber_partition_gone(self):
        buf = ContextBuffer()
        buf.put("alice", _chunks(3))
        buf.clear_subscriber("alice")
        assert buf.has_warm_context("alice", "debugging_python") is False


# ── max_chunks_per_subscriber ─────────────────────────────────────────────────

class TestMaxChunks:
    def test_exceeding_max_drops_oldest(self):
        buf = ContextBuffer(max_chunks_per_subscriber=3)
        for i in range(5):
            buf.put("alice", [_chunk(chunk_id=f"c{i}")])
        # Only 3 most-recent chunks should remain
        assert len(buf.get("alice")) == 3

    def test_max_chunks_keeps_newest(self):
        buf = ContextBuffer(max_chunks_per_subscriber=2)
        buf.put("alice", [_chunk("old")])
        buf.put("alice", [_chunk("newer")])
        buf.put("alice", [_chunk("newest")])
        ids = {c.chunk_id for c in buf.get("alice")}
        assert "newest" in ids
        assert "old" not in ids

    def test_max_chunks_per_subscriber_respected_independently(self):
        buf = ContextBuffer(max_chunks_per_subscriber=2)
        buf.put("alice", _chunks(3))
        buf.put("bob",   _chunks(3))
        assert len(buf.get("alice")) == 2
        assert len(buf.get("bob")) == 2
