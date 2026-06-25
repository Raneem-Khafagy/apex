"""
Tests for SignalMonitor.
Real ProductivityAdapter via constructor injection — no mocks, no patches.
app_detector is a lambda (real callable, not a mock) so tests are deterministic
regardless of which application is in focus on the host machine.
"""
import asyncio
import os
import tempfile

import pytest

from apex.adapters.base import SignalVector
from apex.adapters.productivity import ProductivityAdapter
from apex.monitor.signal_monitor import SignalMonitor


def _make_adapter(tmpdir: str, app_name: str = "Xcode") -> ProductivityAdapter:
    return ProductivityAdapter(
        watch_path=tmpdir,
        app_detector=lambda: app_name,
    )


class TestSignalMonitorCallbacks:
    async def test_callback_receives_signal_vector(self):
        """Monitor must deliver a real SignalVector to the callback on file change."""
        received: list[SignalVector] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = _make_adapter(tmpdir)
            monitor = SignalMonitor(adapter=adapter, watch_path=tmpdir)
            monitor.register_callback(lambda sv: received.append(sv))

            task = asyncio.create_task(monitor.run())
            await asyncio.sleep(0.1)

            with open(os.path.join(tmpdir, "trigger.txt"), "w") as f:
                f.write("go")

            await asyncio.sleep(0.5)
            monitor.stop()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        assert len(received) >= 1
        assert all(isinstance(sv, SignalVector) for sv in received)

    async def test_callback_signal_matches_adapter_domain(self):
        """The signal's activity_type must reflect the adapter's injected app."""
        received: list[SignalVector] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = _make_adapter(tmpdir, app_name="PyCharm")
            monitor = SignalMonitor(adapter=adapter, watch_path=tmpdir)
            monitor.register_callback(lambda sv: received.append(sv))

            task = asyncio.create_task(monitor.run())
            await asyncio.sleep(0.1)

            with open(os.path.join(tmpdir, "code.py"), "w") as f:
                f.write("x = 1")

            await asyncio.sleep(0.5)
            monitor.stop()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        assert any(sv.activity_type == "debugging" for sv in received)

    async def test_multiple_callbacks_all_invoked(self):
        counts = [0, 0]

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = _make_adapter(tmpdir)
            monitor = SignalMonitor(adapter=adapter, watch_path=tmpdir)
            monitor.register_callback(lambda sv: counts.__setitem__(0, counts[0] + 1))
            monitor.register_callback(lambda sv: counts.__setitem__(1, counts[1] + 1))

            task = asyncio.create_task(monitor.run())
            await asyncio.sleep(0.1)

            with open(os.path.join(tmpdir, "trigger.txt"), "w") as f:
                f.write("go")

            await asyncio.sleep(0.5)
            monitor.stop()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        assert counts[0] >= 1
        assert counts[1] >= 1

    async def test_stop_halts_monitor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = _make_adapter(tmpdir)
            monitor = SignalMonitor(adapter=adapter, watch_path=tmpdir)

            task = asyncio.create_task(monitor.run())
            await asyncio.sleep(0.05)
            monitor.stop()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()
                pytest.fail("Monitor did not stop after stop() was called")

    async def test_notify_change_updates_velocity(self):
        """File events should increase velocity (shorter time since last change)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = _make_adapter(tmpdir)
            # Start with stale state
            import time
            adapter._last_change_time = time.time() - 60.0

            velocities: list[float] = []
            monitor = SignalMonitor(adapter=adapter, watch_path=tmpdir)
            monitor.register_callback(lambda sv: velocities.append(sv.velocity_metric))

            task = asyncio.create_task(monitor.run())
            await asyncio.sleep(0.1)

            with open(os.path.join(tmpdir, "active.py"), "w") as f:
                f.write("active")

            await asyncio.sleep(0.5)
            monitor.stop()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        # After notify_change, velocity should be high
        assert any(v >= 0.9 for v in velocities), (
            f"Expected velocity >= 0.9 after file event, got: {velocities}"
        )


class TestSignalMonitorPrivacy:
    def test_signal_vector_fields_are_metadata_types(self):
        """All SignalVector fields must be metadata primitives — no content carriers."""
        sv = SignalVector(
            source_id="abc", content_hash="deadbeef",
            activity_type="writing", velocity_metric=0.5,
            temporal_proximity=0.3, urgency_flag=False,
        )
        for field in ("source_id", "content_hash", "activity_type"):
            assert isinstance(getattr(sv, field), str)
        for field in ("velocity_metric", "temporal_proximity"):
            assert isinstance(getattr(sv, field), (int, float))
        assert isinstance(sv.urgency_flag, bool)

    def test_content_hash_is_fixed_length_hex(self):
        """content_hash must be a fixed-length hex digest — not embedded file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = ProductivityAdapter(watch_path=tmpdir, app_detector=lambda: "Xcode")
            sv = adapter.observe()
        assert len(sv.content_hash) == 16
        int(sv.content_hash, 16)  # raises ValueError if not valid hex
