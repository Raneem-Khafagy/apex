"""
Tests for ResearchAdapter.

Real ResearchAdapter with real temp directories.
No mocks — uses os.stat() metadata only, which is the privacy contract.
notify_change() is called directly to control velocity for determinism.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from apex.adapters.base import SignalAdapter, SignalVector
from apex.adapters.research import (
    ResearchAdapter,
    _DRAFTING_THRESHOLD,
    _REVISING_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _adapter(tmp_path: Path, **kwargs) -> ResearchAdapter:
    """Return a ResearchAdapter watching tmp_path."""
    return ResearchAdapter(draft_dir=str(tmp_path), **kwargs)


def _touch(path: Path, content: str = "x") -> None:
    path.write_text(content)


# ── Contract ──────────────────────────────────────────────────────────────────

class TestResearchAdapterContract:
    def test_is_signal_adapter(self, tmp_path):
        assert isinstance(_adapter(tmp_path), SignalAdapter)

    def test_observe_returns_signal_vector(self, tmp_path):
        sv = _adapter(tmp_path).observe()
        assert isinstance(sv, SignalVector)

    def test_urgency_flag_always_false(self, tmp_path):
        adapter = _adapter(tmp_path)
        adapter.notify_change()
        sv = adapter.observe()
        assert sv.urgency_flag is False

    def test_all_fields_present(self, tmp_path):
        sv = _adapter(tmp_path).observe()
        assert sv.source_id
        assert sv.content_hash
        assert sv.activity_type
        assert isinstance(sv.velocity_metric, float)
        assert isinstance(sv.temporal_proximity, float)


# ── Velocity ──────────────────────────────────────────────────────────────────

class TestVelocity:
    def test_velocity_high_immediately_after_change(self, tmp_path):
        adapter = _adapter(tmp_path, typing_window_sec=30.0)
        adapter.notify_change()
        sv = adapter.observe()
        assert sv.velocity_metric > 0.9

    def test_velocity_decays_over_time(self, tmp_path):
        adapter = _adapter(tmp_path, typing_window_sec=30.0)
        adapter.notify_change()
        # Simulate 40s of inactivity by backdating the timestamp directly.
        # This is equivalent to the passage of time and avoids flaky sleeps.
        adapter._last_change_time -= 40.0
        sv = adapter.observe()
        assert sv.velocity_metric == pytest.approx(0.0, abs=0.05)

    def test_velocity_bounded_zero_to_one(self, tmp_path):
        adapter = _adapter(tmp_path)
        adapter.notify_change()
        sv = adapter.observe()
        assert 0.0 <= sv.velocity_metric <= 1.0


# ── Activity classification ────────────────────────────────────────────────────

class TestActivityClassification:
    def test_idle_when_no_recent_change(self, tmp_path):
        adapter = _adapter(tmp_path, typing_window_sec=30.0)
        # Simulate 40s of inactivity by backdating the timestamp directly.
        adapter._last_change_time -= 40.0
        sv = adapter.observe()
        assert sv.activity_type == "idle"

    def test_drafting_when_high_velocity(self, tmp_path):
        adapter = _adapter(tmp_path, typing_window_sec=30.0)
        adapter.notify_change()
        sv = adapter.observe()
        # velocity ≈ 1.0 > DRAFTING_THRESHOLD
        assert sv.activity_type == "drafting"

    def test_revising_when_low_velocity_no_citations(self, tmp_path):
        # Set window very short so velocity is low-but-nonzero
        adapter = _adapter(tmp_path, typing_window_sec=0.5)
        adapter.notify_change()
        time.sleep(0.35)   # velocity drops to ~0.30 — above 0, below DRAFTING
        sv = adapter.observe()
        assert sv.activity_type in ("revising", "reviewing_lit")

    def test_reviewing_lit_when_mid_velocity_and_citations_recent(self, tmp_path):
        bib = tmp_path / "refs.bib"
        bib.write_text("@article{a,}")
        adapter = _adapter(
            tmp_path,
            citations_path=str(bib),
            typing_window_sec=1.0,
        )
        adapter.notify_change()
        time.sleep(0.4)   # velocity ≈ 0.6 — in [REVISING, DRAFTING) range
        sv = adapter.observe()
        # citations were just written (recent) → reviewing_lit
        assert sv.activity_type == "reviewing_lit"


# ── content_hash ──────────────────────────────────────────────────────────────

class TestContentHash:
    def test_content_hash_is_hex_string(self, tmp_path):
        sv = _adapter(tmp_path).observe()
        assert len(sv.content_hash) == 16
        int(sv.content_hash, 16)   # raises if not hex

    def test_content_hash_changes_when_draft_file_saved(self, tmp_path):
        adapter = _adapter(tmp_path)
        sv1 = adapter.observe()
        _touch(tmp_path / "draft.md")
        sv2 = adapter.observe()
        assert sv1.content_hash != sv2.content_hash

    def test_content_hash_stable_without_changes(self, tmp_path):
        _touch(tmp_path / "draft.md")
        adapter = _adapter(tmp_path)
        sv1 = adapter.observe()
        sv2 = adapter.observe()
        assert sv1.content_hash == sv2.content_hash


# ── source_id ────────────────────────────────────────────────────────────────

class TestSourceId:
    def test_source_id_stable_for_same_dir(self, tmp_path):
        a = _adapter(tmp_path).observe()
        b = _adapter(tmp_path).observe()
        assert a.source_id == b.source_id

    def test_source_id_differs_for_different_dir(self, tmp_path):
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()
        sv_a = ResearchAdapter(draft_dir=str(dir_a)).observe()
        sv_b = ResearchAdapter(draft_dir=str(dir_b)).observe()
        assert sv_a.source_id != sv_b.source_id


# ── deadline_file ─────────────────────────────────────────────────────────────

class TestDeadlineFile:
    def test_temporal_proximity_from_deadline_file(self, tmp_path):
        deadline = tmp_path / "deadline"
        deadline.touch()   # just touched → proximity ≈ 1.0
        adapter = _adapter(tmp_path, deadline_file=str(deadline))
        sv = adapter.observe()
        assert sv.temporal_proximity > 0.99

    def test_temporal_proximity_low_for_old_deadline_file(self, tmp_path):
        deadline = tmp_path / "deadline"
        deadline.touch()
        # Set mtime to 10 days ago (beyond 7-day window → proximity = 0)
        import os
        old_time = time.time() - 10 * 24 * 3600
        os.utime(deadline, (old_time, old_time))
        adapter = _adapter(tmp_path, deadline_file=str(deadline))
        sv = adapter.observe()
        assert sv.temporal_proximity == pytest.approx(0.0)

    def test_missing_deadline_file_falls_back_to_velocity(self, tmp_path):
        adapter = _adapter(tmp_path,
                           deadline_file=str(tmp_path / "no_such_file"),
                           typing_window_sec=30.0)
        adapter.notify_change()
        sv = adapter.observe()
        # proximity falls back to velocity, which is high after notify_change
        assert sv.temporal_proximity > 0.9


# ── Privacy sentinel ──────────────────────────────────────────────────────────

class TestPrivacy:
    def test_observe_never_reads_file_content(self, tmp_path):
        """open() must never be called by observe() — stat() only."""
        draft = tmp_path / "thesis.md"
        draft.write_text("This content must never be read by the adapter.")
        adapter = _adapter(tmp_path)

        real_open = open
        opened_files: list[str] = []

        def spy_open(file, *args, **kwargs):
            opened_files.append(str(file))
            return real_open(file, *args, **kwargs)

        import builtins
        builtins.open = spy_open  # type: ignore[assignment]
        try:
            adapter.observe()
        finally:
            builtins.open = real_open  # type: ignore[assignment]

        draft_opened = any(str(draft) in f for f in opened_files)
        assert not draft_opened, (
            f"observe() read file content — privacy violation.\n"
            f"Opened: {opened_files}"
        )
