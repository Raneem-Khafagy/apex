"""
APEX Live Terminal Display — Rich-powered pipeline state monitor.

Renders four panels in a live-updating terminal layout:

┌──────────────────────────────────────────────────────────────────┐
│ APEX — Proactive Context Pipeline                                │
├─────────────────────┬────────────────────┬───────────────────────┤
│  Signal Monitor     │  Pipeline State    │  Thesis Metrics       │
│  (latest signal)    │  (last decision)   │  (PRP / LtC / DPS)    │
├─────────────────────┴────────────────────┴───────────────────────┤
│  Context Buffer  (per-subscriber chunk summary)                  │
└──────────────────────────────────────────────────────────────────┘

Usage
-----
Run standalone via `just monitor` (or `uv run python -m apex.monitor.live`).
Or embed in the dev stack via `just dev`.

The display is a passive observer — it does not drive the pipeline.
It is updated via update_*() calls from PipelineCoordinator / SignalMonitor.

Privacy: no document text is displayed. Only labels, scores, and counts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── State dataclasses (plain Python — no Rich dependency) ─────────────────────

@dataclass
class SignalState:
    activity_type: str = "—"
    velocity: float = 0.0
    urgency: bool = False
    label: str = "—"
    confidence: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class PipelineState:
    last_action: str = "—"         # "RETRIEVE" or "WAIT"
    last_label: str = "—"
    last_tau: float = 0.65
    last_reason: str = "—"
    retrieve_count: int = 0
    wait_count: int = 0
    ts: float = field(default_factory=time.time)


@dataclass
class MetricsState:
    prp: Optional[float] = None      # Proactive Retrieval Precision
    mean_ltc: Optional[float] = None # Mean Latency-to-Context (ms)
    dps: Optional[float] = None      # Delivery Precision Score
    battery_mw: Optional[float] = None


@dataclass
class BufferState:
    # subscriber_id → chunk count
    partitions: dict[str, int] = field(default_factory=dict)
    total_chunks: int = 0


# ── LiveDisplay ───────────────────────────────────────────────────────────────

class LiveDisplay:
    """
    Rich live terminal display for the APEX pipeline.

    Designed to be driven by external updates — call update_*() from
    whichever pipeline component has new data.

    Parameters
    ----------
    refresh_rate
        Frames per second for the live display loop.
    """

    def __init__(self, refresh_rate: float = 4.0, screen: bool = True) -> None:
        self._console = Console()
        self._refresh_rate = refresh_rate
        self._screen = screen
        self._signal = SignalState()
        self._pipeline = PipelineState()
        self._metrics = MetricsState()
        self._buffer = BufferState()
        self._start_time = time.time()
        self._live: Optional[Live] = None

    # ── State update API ──────────────────────────────────────────────────────

    def update_signal(
        self,
        activity_type: str,
        velocity: float,
        urgency: bool,
        label: str,
        confidence: float,
    ) -> None:
        self._signal = SignalState(
            activity_type=activity_type,
            velocity=velocity,
            urgency=urgency,
            label=label,
            confidence=confidence,
        )

    def update_pipeline(
        self,
        action: str,
        label: str,
        tau: float,
        reason: str,
    ) -> None:
        self._pipeline.last_action = action
        self._pipeline.last_label = label
        self._pipeline.last_tau = tau
        self._pipeline.last_reason = reason
        self._pipeline.ts = time.time()
        if action == "RETRIEVE":
            self._pipeline.retrieve_count += 1
        else:
            self._pipeline.wait_count += 1

    def update_metrics(
        self,
        prp: Optional[float] = None,
        mean_ltc: Optional[float] = None,
        dps: Optional[float] = None,
        battery_mw: Optional[float] = None,
    ) -> None:
        if prp is not None:
            self._metrics.prp = prp
        if mean_ltc is not None:
            self._metrics.mean_ltc = mean_ltc
        if dps is not None:
            self._metrics.dps = dps
        if battery_mw is not None:
            self._metrics.battery_mw = battery_mw

    def update_buffer(self, partitions: dict[str, int]) -> None:
        self._buffer.partitions = partitions
        self._buffer.total_chunks = sum(partitions.values())

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_signal_panel(self) -> Panel:
        s = self._signal
        age = time.time() - s.ts

        label_color = {
            "debugging_python": "cyan",
            "writing_document": "green",
            "reading_reference": "blue",
        }.get(s.label, "white")

        urgency_text = Text("YES", style="bold red") if s.urgency else Text("no", style="dim")

        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", width=14)
        table.add_column()
        table.add_row("activity:", s.activity_type)
        table.add_row("velocity:", f"{s.velocity:.2f}")
        table.add_row("urgency:", urgency_text)
        table.add_row("label:", Text(s.label, style=label_color))
        table.add_row("confidence:", f"{s.confidence:.3f}")
        table.add_row("age:", f"{age:.1f}s ago")

        return Panel(table, title="[bold]Signal Monitor[/bold]", border_style="blue")

    def _render_pipeline_panel(self) -> Panel:
        p = self._pipeline
        age = time.time() - p.ts

        action_style = "bold green" if p.last_action == "RETRIEVE" else "bold yellow"

        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", width=10)
        table.add_column()
        table.add_row("action:", Text(p.last_action, style=action_style))
        table.add_row("label:", p.last_label)
        table.add_row("τ used:", f"{p.last_tau:.3f}")
        table.add_row("reason:", Text(p.last_reason[:40], style="italic"))
        table.add_row("retrieve:", str(p.retrieve_count))
        table.add_row("wait:", str(p.wait_count))
        table.add_row("age:", f"{age:.1f}s ago")

        return Panel(table, title="[bold]Pipeline State[/bold]", border_style="yellow")

    def _render_metrics_panel(self) -> Panel:
        m = self._metrics

        def _fmt(v: Optional[float], fmt: str = ".3f", suffix: str = "") -> str:
            return f"{v:{fmt}}{suffix}" if v is not None else "—"

        def _prp_style(v: Optional[float]) -> str:
            if v is None:
                return "white"
            return "green" if v > 0.65 else "red"

        def _dps_style(v: Optional[float]) -> str:
            if v is None:
                return "white"
            return "green" if v > 0.75 else "red"

        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", width=10)
        table.add_column()
        table.add_row("PRP:", Text(_fmt(m.prp), style=_prp_style(m.prp)))
        table.add_row("LtC:", Text(_fmt(m.mean_ltc, ".1f", "ms")))
        table.add_row("DPS:", Text(_fmt(m.dps), style=_dps_style(m.dps)))
        table.add_row("battery:", Text(_fmt(m.battery_mw, ".0f", "mW")))

        return Panel(table, title="[bold]Thesis Metrics[/bold]", border_style="magenta")

    def _render_buffer_panel(self) -> Panel:
        b = self._buffer

        if not b.partitions:
            content = Text("No subscribers yet", style="dim italic")
        else:
            table = Table("Subscriber", "Chunks", box=None, show_header=True)
            table.header_style = "bold dim"
            for sub_id, count in b.partitions.items():
                count_style = "green" if count > 0 else "dim"
                table.add_row(sub_id[:24], Text(str(count), style=count_style))
            content = table  # type: ignore[assignment]

        uptime = time.time() - self._start_time
        title = (
            f"[bold]Context Buffer[/bold] "
            f"[dim]total={b.total_chunks} | "
            f"uptime={uptime:.0f}s[/dim]"
        )
        return Panel(content, title=title, border_style="cyan")

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="buffer", size=10),
        )
        layout["main"].split_row(
            Layout(self._render_signal_panel(), name="signal"),
            Layout(self._render_pipeline_panel(), name="pipeline"),
            Layout(self._render_metrics_panel(), name="metrics"),
        )
        layout["buffer"].update(self._render_buffer_panel())
        layout["header"].update(
            Panel(
                Text(
                    "APEX — Application-agnostic Proactive Edge-native conteXt pushing",
                    style="bold white",
                    justify="center",
                ),
                style="bold blue",
            )
        )
        return layout

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "LiveDisplay":
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            screen=self._screen,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            self._live.__exit__(*args)

    def refresh(self) -> None:
        """Manually trigger a display refresh."""
        if self._live:
            self._live.update(self._render())


# ── Standalone entry point ────────────────────────────────────────────────────

def _demo() -> None:
    """
    Demo mode: display static placeholder data.
    Run via: uv run python -m apex.monitor.live
    """
    import math

    display = LiveDisplay(refresh_rate=4.0)

    display.update_signal(
        activity_type="debugging",
        velocity=0.75,
        urgency=False,
        label="debugging_python",
        confidence=0.90,
    )
    display.update_pipeline(
        action="RETRIEVE",
        label="debugging_python",
        tau=0.65,
        reason="c=0.900 >= τ=0.650, buffer_miss",
    )
    display.update_metrics(prp=0.72, mean_ltc=12.5, dps=0.81, battery_mw=320.0)
    display.update_buffer({"subscriber_vscode": 3, "subscriber_terminal": 1})

    try:
        with display:
            t = 0.0
            while True:
                time.sleep(0.25)
                t += 0.25
                # Animate velocity
                display.update_signal(
                    activity_type="debugging",
                    velocity=0.5 + 0.4 * abs(math.sin(t * 0.3)),
                    urgency=False,
                    label="debugging_python",
                    confidence=0.85 + 0.05 * math.sin(t * 0.5),
                )
                display.refresh()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    _demo()
