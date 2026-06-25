"""
Productivity Signal Adapter — macOS personal-productivity domain.

Reads only structural metadata:
  - Active foreground application (via osascript on macOS)
  - File modification timestamps in the watched directory (via os.scandir)
  - Time elapsed since the last detected file change (velocity proxy)

Privacy rule: this module never calls open() or reads any file content.
"""
import hashlib
import os
import subprocess
import time
from typing import Callable, Optional

from loguru import logger

from apex.adapters.base import SignalAdapter, SignalVector

# ── Activity classification ──────────────────────────────────────────────────

# Map substrings of app names to APEX activity types.
# Checked in order — first match wins.
_APP_ACTIVITY_MAP: list[tuple[tuple[str, ...], str]] = [
    (("xcode", "pycharm", "android studio", "intellij", "clion",
      "webstorm", "rider", "goland", "rubymine", "appcode",
      "cursor", "zed"), "debugging"),
    (("pages", "word", "typora", "obsidian", "ia writer", "ulysses",
      "notion", "bear", "craft", "drafts", "textedit"), "writing"),
    (("safari", "chrome", "firefox", "arc", "brave", "opera",
      "preview", "skim", "papers", "zotero", "readkit"), "reading"),
]

_VELOCITY_DECAY_SECONDS = 30.0  # velocity reaches 0 after this many idle seconds
# Minimum velocity to infer "writing" from file-change activity when the
# active app maps to "idle" (e.g. Terminal running an eval script).
_FILE_ACTIVITY_THRESHOLD = 0.3


def _classify_activity(app_name: str) -> str:
    """Return an activity type string from the active application name."""
    lower = app_name.lower()
    for keywords, activity in _APP_ACTIVITY_MAP:
        if any(kw in lower for kw in keywords):
            return activity
    return "idle"


def _compute_content_hash(watch_path: str) -> str:
    """
    Hash the (name, mtime) pairs of top-level entries in watch_path.
    Reads only filesystem metadata — never file content.
    """
    try:
        entries = sorted(
            (e.name, e.stat().st_mtime)
            for e in os.scandir(watch_path)
        )
    except (PermissionError, FileNotFoundError):
        entries = []
    digest_input = str(entries).encode()
    return hashlib.sha256(digest_input).hexdigest()[:16]


def _detect_active_app_macos() -> str:
    """
    Query the macOS System Events API for the frontmost application name.
    Falls back to "Unknown" on any error (e.g., accessibility not granted).
    """
    script = (
        'tell application "System Events" to '
        'get name of first application process whose frontmost is true'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=0.5,
        )
        name = result.stdout.strip()
        return name if name else "Unknown"
    except Exception as exc:
        logger.debug("App detection failed: {}", exc)
        return "Unknown"


# ── Adapter ──────────────────────────────────────────────────────────────────

class ProductivityAdapter(SignalAdapter):
    """
    Behavioral Signal Adapter for the personal-productivity domain.

    Parameters
    ----------
    watch_path
        Directory whose file metadata is hashed to produce content_hash.
        Typically the user's active project or home directory.
    app_detector
        Callable that returns the foreground application name.
        Defaults to the macOS osascript implementation.
        Inject a lambda in tests to avoid subprocess calls.
    """

    def __init__(
        self,
        watch_path: str = os.path.expanduser("~"),
        app_detector: Optional[Callable[[], str]] = None,
    ) -> None:
        self._watch_path = watch_path
        self._app_detector = app_detector or _detect_active_app_macos
        self._last_change_time: float = time.time()

    # Called by SignalMonitor on every watchfiles event to update velocity state.
    def notify_change(self) -> None:
        """Record the time of the most recent file change event."""
        self._last_change_time = time.time()

    def observe(self) -> SignalVector:
        """
        Return a SignalVector snapshot of the current productivity context.
        Reads only metadata — never file content.
        """
        app_name = self._app_detector()

        # source_id: deterministic hash of app identity
        source_id = hashlib.sha256(app_name.encode()).hexdigest()[:16]

        # content_hash: hash of filesystem metadata in watch_path
        content_hash = _compute_content_hash(self._watch_path)

        # velocity: decays linearly from 1.0 to 0.0 over DECAY window
        # Computed before activity classification so it can inform the override.
        elapsed = time.time() - self._last_change_time
        velocity = max(0.0, 1.0 - elapsed / _VELOCITY_DECAY_SECONDS)

        activity_type = _classify_activity(app_name)

        # Override: if the active app maps to "idle" but files in the vault are
        # actively changing (velocity above threshold), infer "writing" from the
        # file-change signal. This handles eval scenarios where a Terminal script
        # (e.g. vault_agent.py) is editing files — Terminal maps to "idle" but the
        # behavioral signal is clearly writing-domain activity.
        if activity_type == "idle" and velocity >= _FILE_ACTIVITY_THRESHOLD:
            activity_type = "writing"
            logger.debug(
                "ProductivityAdapter: app='{}' → idle, but velocity={:.2f} ≥ {:.2f} → override to 'writing'",
                app_name, velocity, _FILE_ACTIVITY_THRESHOLD,
            )

        # temporal_proximity: use velocity as proxy
        # (high activity = user is about to need context)
        temporal_proximity = velocity

        return SignalVector(
            source_id=source_id,
            content_hash=content_hash,
            activity_type=activity_type,
            velocity_metric=velocity,
            temporal_proximity=temporal_proximity,
            urgency_flag=False,  # productivity domain is never safety-critical
        )
