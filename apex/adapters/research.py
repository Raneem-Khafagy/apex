"""
PureSearch / Academic Research Signal Adapter.

Reads draft state from filesystem metadata only — file sizes, modification
timestamps, and citation file size as a count proxy. Never reads document
content, bibliography entries, or author names.

Signal semantics
----------------
source_id          : SHA-256 of draft_dir path (stable per project)
content_hash       : SHA-256 of (filename, mtime, size) tuples in draft_dir
                     — changes when any draft file is saved
activity_type      : "drafting" | "reviewing_lit" | "revising" | "idle"
velocity_metric    : typing burst intensity — recency of last file change ∈ [0, 1]
temporal_proximity : deadline proximity ∈ [0, 1] — set externally or via file
urgency_flag       : always False (research domain is not safety-critical)

Activity classification
-----------------------
Velocity is the primary axis. Citation file size growth is the secondary axis:

  velocity ≥ 0.7                       → "drafting"      (active writing burst)
  velocity ∈ [0.2, 0.7) + cit growing → "reviewing_lit"  (reading + annotating)
  velocity ∈ (0, 0.2)                  → "revising"       (slow, deliberate edits)
  velocity = 0                         → "idle"

Privacy rule: this module never calls open() on any draft file.
All signals are derived from stat() metadata only.

Phase 0 (local integration): point draft_dir at a local LaTeX or Markdown
thesis directory. Set deadline_file to a plain file whose mtime you update
manually to simulate deadline pressure.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from apex.adapters.base import SignalAdapter, SignalVector

# ── Constants ─────────────────────────────────────────────────────────────────

# Draft file extensions monitored for change detection.
_DRAFT_EXTENSIONS: frozenset[str] = frozenset({
    ".tex", ".md", ".txt", ".rst", ".typ",
})

# Citation file extensions (size used as a proxy for citation count).
_CITATION_EXTENSIONS: frozenset[str] = frozenset({
    ".bib", ".json", ".yaml", ".yml",
})

# Velocity decays to 0 after this many seconds of inactivity.
_VELOCITY_DECAY_SECONDS: float = 60.0

# Velocity thresholds for activity classification.
_DRAFTING_THRESHOLD: float = 0.7
_REVISING_THRESHOLD: float = 0.2


class ResearchAdapter(SignalAdapter):
    """
    Behavioral Signal Adapter for the PureSearch / academic research domain.

    Parameters
    ----------
    draft_dir
        Directory containing draft files (.tex, .md, .rst, etc.).
        Monitored via stat() metadata only — content is never read.
    citations_path
        Path to a citations file (.bib, .json, etc.) or a directory
        containing citation files. Size is used as a count proxy.
    deadline_file
        Optional path to a file whose mtime encodes deadline proximity.
        When provided, temporal_proximity is derived from how recently
        this file was touched relative to a 7-day window.
        If None, temporal_proximity mirrors velocity_metric.
    typing_window_sec
        Number of seconds over which typing velocity decays.
        Shorter window = more responsive to bursts; longer = smoother.
    """

    def __init__(
        self,
        draft_dir: str,
        citations_path: Optional[str] = None,
        deadline_file: Optional[str] = None,
        typing_window_sec: float = _VELOCITY_DECAY_SECONDS,
    ) -> None:
        self._draft_dir = Path(draft_dir)
        self._citations_path = Path(citations_path) if citations_path else None
        self._deadline_file = Path(deadline_file) if deadline_file else None
        self._typing_window = max(typing_window_sec, 1.0)
        self._last_change_time: float = time.time()
        logger.info(
            "ResearchAdapter: watching draft_dir='{}' citations='{}'",
            self._draft_dir,
            self._citations_path or "(none)",
        )

    # ── Called by SignalMonitor on file change events ─────────────────────────

    def notify_change(self) -> None:
        """Record the time of the most recent draft file change event."""
        self._last_change_time = time.time()

    # ── SignalAdapter contract ────────────────────────────────────────────────

    def observe(self) -> SignalVector:
        """
        Return a SignalVector snapshot of the current research activity.
        Reads only filesystem metadata — never file content.
        """
        # ── Velocity: typing burst intensity ─────────────────────────────────
        elapsed = time.time() - self._last_change_time
        velocity = max(0.0, 1.0 - elapsed / self._typing_window)

        # ── Temporal proximity: deadline pressure ─────────────────────────────
        temporal_proximity = self._compute_deadline_proximity(velocity)

        # ── Content hash: changes when any draft file is saved ───────────────
        content_hash = self._compute_content_hash()

        # ── Citation size: proxy for "in reading / annotation mode" ──────────
        citations_growing = self._citations_changed()

        # ── Source ID: stable per draft project ──────────────────────────────
        source_id = hashlib.sha256(
            str(self._draft_dir.resolve()).encode()
        ).hexdigest()[:16]

        # ── Activity classification ───────────────────────────────────────────
        activity_type = self._classify(velocity, citations_growing)

        logger.debug(
            "ResearchAdapter: vel={:.3f} proximity={:.3f} activity='{}' cit_growing={}",
            velocity, temporal_proximity, activity_type, citations_growing,
        )

        return SignalVector(
            source_id=source_id,
            content_hash=content_hash,
            activity_type=activity_type,
            velocity_metric=velocity,
            temporal_proximity=temporal_proximity,
            urgency_flag=False,  # research domain is never safety-critical
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _compute_content_hash(self) -> str:
        """
        Hash (filename, mtime, size) for all draft files in draft_dir.
        Never reads file content — stat() only.
        """
        try:
            entries = sorted(
                (e.name, e.stat().st_mtime, e.stat().st_size)
                for e in os.scandir(self._draft_dir)
                if Path(e.name).suffix.lower() in _DRAFT_EXTENSIONS
            )
        except (PermissionError, FileNotFoundError):
            entries = []
        return hashlib.sha256(str(entries).encode()).hexdigest()[:16]

    def _citations_changed(self) -> bool:
        """
        Return True if the citations file/directory has been modified recently
        (within the last typing_window_sec). Used for activity classification.
        """
        if self._citations_path is None:
            return False
        try:
            if self._citations_path.is_dir():
                mtime = max(
                    (e.stat().st_mtime for e in os.scandir(self._citations_path)
                     if Path(e.name).suffix.lower() in _CITATION_EXTENSIONS),
                    default=0.0,
                )
            else:
                mtime = self._citations_path.stat().st_mtime
            return (time.time() - mtime) < self._typing_window
        except (PermissionError, FileNotFoundError):
            return False

    def _compute_deadline_proximity(self, velocity: float) -> float:
        """
        Return a normalized [0, 1] deadline proximity score.

        If deadline_file is set, its recency within a 7-day window determines
        the score (recently touched = deadline is close).
        Otherwise falls back to velocity as a proxy.
        """
        if self._deadline_file is None:
            return velocity  # proxy: high activity ≈ user feels time pressure
        try:
            mtime = self._deadline_file.stat().st_mtime
            elapsed_since_touch = time.time() - mtime
            seven_days_sec = 7 * 24 * 3600
            # 1.0 = touched within last second; 0.0 = touched > 7 days ago
            return max(0.0, 1.0 - elapsed_since_touch / seven_days_sec)
        except (PermissionError, FileNotFoundError):
            return velocity

    @staticmethod
    def _classify(velocity: float, citations_growing: bool) -> str:
        """Classify the current research activity from velocity and citation state."""
        if velocity >= _DRAFTING_THRESHOLD:
            return "drafting"
        if velocity >= _REVISING_THRESHOLD:
            # Mid-velocity: distinguish review (citations active) from revision
            return "reviewing_lit" if citations_growing else "revising"
        if velocity > 0:
            return "revising"
        return "idle"
