"""
Behavioral Signal Monitor (BSM).

Always-on async loop that:
  1. Watches a directory for file-change events via watchfiles.awatch().
  2. On each change, calls adapter.observe() to capture the current signal vector.
  3. Dispatches the signal to all registered callbacks.
  4. Also fires a periodic heartbeat tick so downstream components never
     stall in the absence of file events.

Privacy rule: this module never reads file content.
watchfiles delivers (ChangeType, path) pairs — only path metadata is used,
and even that is not forwarded to the adapter.
"""
import asyncio
import inspect
from typing import Callable

from loguru import logger
from watchfiles import awatch, Change

from apex.adapters.base import SignalAdapter, SignalVector

# Heartbeat: fire adapter.observe() even when no file events occur,
# so time-based signals (idle, temporal_proximity) stay fresh.
_HEARTBEAT_SECONDS = 5.0


class SignalMonitor:
    """
    Behavioral Signal Monitor.

    Parameters
    ----------
    adapter
        The domain-specific SignalAdapter to call on each event.
    watch_path
        Directory to watch for file-system change events.
    heartbeat_interval
        How often (seconds) to fire observe() even without a file event.
    """

    def __init__(
        self,
        adapter: SignalAdapter,
        watch_path: str,
        heartbeat_interval: float = _HEARTBEAT_SECONDS,
    ) -> None:
        self._adapter = adapter
        self._watch_path = watch_path
        self._heartbeat_interval = heartbeat_interval
        self._callbacks: list[Callable] = []
        self._running = False

    def register_callback(self, cb: Callable[[SignalVector], None]) -> None:
        """Register a callable that receives every emitted SignalVector."""
        self._callbacks.append(cb)

    def stop(self) -> None:
        """Signal the monitor to stop after the current event is processed."""
        self._running = False

    async def _dispatch(self, signal: SignalVector) -> None:
        """Invoke all registered callbacks with the signal."""
        for cb in self._callbacks:
            try:
                result = cb(signal)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                logger.error("Callback {} raised: {}", cb, exc)

    async def _heartbeat_loop(self) -> None:
        """Periodically emit a signal even without file-system activity."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            if not self._running:
                break
            signal = self._adapter.observe()
            logger.debug("BSM heartbeat: {}", signal.activity_type)
            await self._dispatch(signal)

    async def run(self) -> None:
        """
        Start the monitor. Blocks until stop() is called.

        Spawns a heartbeat task, then enters the watchfiles event loop.
        On each file-system change event:
          - Notifies the adapter (if it supports notify_change)
          - Calls adapter.observe() to get the current SignalVector
          - Dispatches the signal to all callbacks

        The file path from watchfiles is used only to trigger the observe()
        call — its content is never read.
        """
        self._running = True
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            async for changes in awatch(self._watch_path, stop_event=self._make_stop_event()):
                if not self._running:
                    break

                # changes is a set of (Change, path) — we use only the event
                # as a trigger. The path is structural metadata; content is
                # never read.
                logger.debug("BSM file event: {} change(s) detected", len(changes))

                # Let the adapter update its velocity state if it supports it
                if hasattr(self._adapter, "notify_change"):
                    self._adapter.notify_change()

                signal = self._adapter.observe()
                await self._dispatch(signal)

        except Exception as exc:
            logger.error("BSM run loop error: {}", exc)
        finally:
            self._running = False
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    def _make_stop_event(self) -> asyncio.Event:
        """
        Return an asyncio.Event that is set when stop() is called.
        watchfiles.awatch() uses this to terminate cleanly.
        """
        stop_event = asyncio.Event()

        async def _watcher() -> None:
            while self._running:
                await asyncio.sleep(0.05)
            stop_event.set()

        asyncio.create_task(_watcher())
        return stop_event
