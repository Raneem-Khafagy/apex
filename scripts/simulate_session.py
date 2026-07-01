"""
Phase 0 — Automated session simulator for APEX.

Drives the Behavioral Signal Monitor by making controlled edits to files in
APEX_VAULT_PATH, producing realistic velocity patterns that the IIE and SRS
can act on.  Covers three scenarios back-to-back:

  1. writing   — Obsidian-style thesis editing (slow write → pause → write)
  2. debugging — VS Code-style code edits (fast saves, short pauses)
  3. reading   — document open with minimal writes (low velocity)

Each file edit triggers watchfiles → SignalMonitor → PipelineCoordinator →
ContextBuffer. If a push fires the WebSocket callback, the DuckDB store logs
a claimed prefetch event.

After the session, run:
    just metrics
to see PRP and LtC for this run.

Usage
-----
    # Prerequisites: daemon running with the experiment index
    APEX_VAULT_PATH=experiment_corpus APEX_INDEX_PATH=experiment_index/experiment just dev

    # In a separate terminal:
    uv run python scripts/simulate_session.py

    # Or with a custom vault path:
    APEX_VAULT_PATH=/path/to/vault uv run python scripts/simulate_session.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import websockets

# ── Configuration ─────────────────────────────────────────────────────────────

VAULT_PATH = Path(
    os.environ.get("APEX_VAULT_PATH", "experiment_corpus")
).expanduser()

APEX_PORT = int(os.environ.get("APEX_PORT", "8765"))

# File written by scripts/register_consumers.py
SUBSCRIBERS_FILE = Path(".phase0_subscribers.json")

# Total time per scenario (seconds). Override via --duration CLI arg.
# Default: 60s per scenario = ~3 min total. For full evaluation use 200s+ per scenario.
SCENARIO_DURATION = int(os.environ.get("APEX_SCENARIO_DURATION", "60"))

# ── Scenario definitions ───────────────────────────────────────────────────────

WRITING_FILE  = "writing/thesis_draft.md"
DEBUGGING_FILE = "debugging/async_patterns.py"
READING_FILE  = "reading/literature_notes.md"

# Content blocks appended on each edit cycle
_WRITING_BLOCKS = [
    """
## 3.1 System Architecture

APEX is composed of six domain-blind pipeline stages connected through
well-defined interfaces.  The Behavioral Signal Monitor (BSM) observes
OS-level metadata events and normalises them into a fixed-schema SignalVector.
""",
    """
The Intent Inference Engine receives each SignalVector and outputs a triple
(q̂, c, ℓ) where q̂ is a dense embedding, c is a confidence score, and ℓ is
a task-context label.  The IIE never produces a text query string.
""",
    """
The Speculative Retrieval Scheduler gates retrieval against a learnable
threshold τ.  When urgency_flag is True (e.g. factory anomaly), τ collapses
to zero and retrieval fires unconditionally.
""",
    """
Hybrid retrieval combines HNSW dense search on q̂ with BM25 sparse search on ℓ,
fusing results via Reciprocal Rank Fusion (k=60).  This gives recall benefits
of semantic search without sacrificing exact-match precision for domain labels.
""",
    """
The Context Buffer maintains per-subscriber TTL partitions (default 60 s).
A dirty-flag mechanism marks cached chunks stale when the source file changes,
suppressing outdated context from reaching any subscriber.
""",
    """
The LLM Adapter Layer (Phi-3.5 Mini, INT4, Ollama) reformats the same
retrieved chunks into subscriber-specific shapes.  It is a semantic translator,
not a generator — if the chunk set is empty, nothing is emitted.
""",
]

_DEBUGGING_BLOCKS = [
    """
# Pattern: Retry with exponential backoff
async def retry(fn, max_attempts=3, base_delay=0.5):
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))
""",
    """
# Pattern: Async timeout wrapper
async def with_timeout(coro, timeout_s=0.025):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        return None
""",
    """
# Pattern: Bounded task queue
class BoundedQueue:
    def __init__(self, max_size=10):
        self._q = asyncio.Queue(maxsize=max_size)
    async def put(self, item):
        await self._q.put(item)
    async def get(self):
        return await self._q.get()
""",
    """
# Pattern: Stale-while-revalidate buffer
async def get_or_refresh(buffer, key, fetch_fn, ttl_s=60.0):
    entry = buffer.get(key)
    if entry and time.time() - entry['ts'] < ttl_s:
        return entry['value']
    asyncio.create_task(fetch_fn())  # background refresh
    return entry['value'] if entry else None
""",
]

_READING_BLOCKS = [
    """
## Notes — ProactiveBench (Wang et al., ICLR 2025)

Key finding: fine-tuned smaller models outperform frontier models on proactive
tasks when given domain-specific training signal.  Motivates on-device
domain adaptation for APEX.
""",
    """
## Notes — ContextAgent (NeurIPS 2025)

Most comparable prior work.  Cloud-dependent — violates the on-device privacy
constraint that is central to APEX.  Latency numbers not directly comparable
due to network round-trips.  APEX's architectural advantage is clear here.
""",
]


# ── Subscriber WebSocket listeners ────────────────────────────────────────────

def _load_subscriber_ids() -> dict[str, str]:
    """
    Load label → subscriber_id mapping from .phase0_subscribers.json.

    Returns an empty dict if the file doesn't exist (simulation will still
    run, but no claims will be recorded since no WS connections are made).
    """
    if SUBSCRIBERS_FILE.exists():
        return json.loads(SUBSCRIBERS_FILE.read_text())
    return {}


async def _ws_listener(
    label: str,
    subscriber_id: str,
    stop_event: asyncio.Event,
    counters: dict[str, int],
) -> None:
    """
    Keep one WebSocket connection open for *subscriber_id*.

    Each received push constitutes a claim — the coordinator's
    _push_context returns True → log_claim fires.  We just receive
    and count; the actual claim bookkeeping happens server-side.
    """
    uri = f"ws://localhost:{APEX_PORT}/stream/{subscriber_id}"
    while not stop_event.is_set():
        try:
            async with websockets.connect(uri, ping_interval=20) as ws:
                while not stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        counters[label] = counters.get(label, 0) + 1
                        print(
                            f"  [push→{label}] {len(msg):>5} bytes  "
                            f"total={counters[label]}",
                        )
                    except asyncio.TimeoutError:
                        continue
        except (OSError, websockets.exceptions.WebSocketException):
            # Daemon not up yet or transient error — retry after a moment
            if not stop_event.is_set():
                await asyncio.sleep(2.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append(filepath: Path, content: str) -> None:
    """Append content to a file, creating it if necessary."""
    _ensure_dir(filepath.parent)
    with filepath.open("a", encoding="utf-8") as f:
        f.write(content)


# ── Scenario runners ─────────────────────────────────────────────────────────

async def run_writing_scenario(duration_s: int) -> None:
    """
    Simulate active thesis writing: bursts of edits separated by read pauses.
    Velocity pattern: high (0.9) → decay → spike → decay ...
    """
    filepath = VAULT_PATH / WRITING_FILE
    _ensure_dir(filepath.parent)

    print(f"  [writing] editing {filepath.name}  ({duration_s}s)")
    start = time.time()
    idx = 0

    while time.time() - start < duration_s:
        block = _WRITING_BLOCKS[idx % len(_WRITING_BLOCKS)]
        _append(filepath, block)
        print(f"    +edit {idx + 1}  vel≈high", end="\r")
        idx += 1
        # Writing burst: 3–5 edits in quick succession
        await asyncio.sleep(2.0)
        if idx % 3 == 0:
            # Simulate reading/thinking pause (velocity decay)
            await asyncio.sleep(20.0)

    print(f"\n  [writing] done — {idx} edits")


async def run_debugging_scenario(duration_s: int) -> None:
    """
    Simulate active code editing: rapid saves with short pauses.
    Velocity pattern: sustained high (0.8+) with brief dips.
    """
    filepath = VAULT_PATH / DEBUGGING_FILE
    _ensure_dir(filepath.parent)

    print(f"  [debugging] editing {filepath.name}  ({duration_s}s)")
    start = time.time()
    idx = 0

    while time.time() - start < duration_s:
        block = _DEBUGGING_BLOCKS[idx % len(_DEBUGGING_BLOCKS)]
        _append(filepath, block)
        print(f"    +edit {idx + 1}  vel≈high", end="\r")
        idx += 1
        # Fast save cycle: typical developer rhythm
        await asyncio.sleep(5.0)

    print(f"\n  [debugging] done — {idx} edits")


async def run_reading_scenario(duration_s: int) -> None:
    """
    Simulate literature review: occasional annotations, long reading pauses.
    Velocity pattern: near-zero → brief spike → near-zero.
    """
    filepath = VAULT_PATH / READING_FILE
    _ensure_dir(filepath.parent)

    print(f"  [reading] editing {filepath.name}  ({duration_s}s)")
    start = time.time()
    idx = 0

    while time.time() - start < duration_s:
        block = _READING_BLOCKS[idx % len(_READING_BLOCKS)]
        _append(filepath, block)
        print(f"    +note {idx + 1}  vel≈low", end="\r")
        idx += 1
        # Long reading pause between annotations
        await asyncio.sleep(25.0)

    print(f"\n  [reading] done — {idx} edits")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(scenario_duration: int = SCENARIO_DURATION) -> None:
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         APEX Phase 0 — Automated Session Simulator          ║
╠══════════════════════════════════════════════════════════════╣
║  Vault:    {str(VAULT_PATH):<50}║
║  Duration: ~{scenario_duration * 3 // 60} min  ({scenario_duration}s × 3 scenarios)              ║
╠══════════════════════════════════════════════════════════════╣
║  For full proactive evaluation use vault_agent.py instead:  ║
║    uv run python scripts/vault_agent.py                     ║
╚══════════════════════════════════════════════════════════════╝

  Prerequisites:
    Terminal 1:  just llm
    Terminal 2:  APEX_VAULT_PATH=experiment_corpus \\
                 APEX_INDEX_PATH=experiment_index/experiment just dev
    Terminal 3:  just register   (if not already done)

  After this run: just stop && just metrics
""")

    # ── Spawn WebSocket listeners for each registered subscriber ─────────────
    subscribers = _load_subscriber_ids()
    stop_event = asyncio.Event()
    push_counters: dict[str, int] = {}

    ws_tasks: list[asyncio.Task] = []
    if subscribers:
        print(f"  Connecting WS listeners for {len(subscribers)} subscriber(s)…")
        for lbl, sid in subscribers.items():
            task = asyncio.create_task(
                _ws_listener(lbl, sid, stop_event, push_counters),
                name=f"ws-{lbl}",
            )
            ws_tasks.append(task)
        # Give connections a moment to establish before file edits begin
        await asyncio.sleep(1.5)
    else:
        print(
            "  [warn] No .phase0_subscribers.json found — "
            "running without WS listeners (0 claims will be recorded).\n"
            "         Run `just register` first for full claim tracking."
        )

    input("  Press Enter to start (or Ctrl-C to abort) …\n")

    t0 = time.time()

    print("── Scenario 1/3: Writing ─────────────────────────────")
    await run_writing_scenario(scenario_duration)
    print(f"  Gap: switching scenario (15s) …")
    await asyncio.sleep(15)

    print("\n── Scenario 2/3: Debugging ───────────────────────────")
    await run_debugging_scenario(scenario_duration)
    print(f"  Gap: switching scenario (15s) …")
    await asyncio.sleep(15)

    print("\n── Scenario 3/3: Reading ─────────────────────────────")
    await run_reading_scenario(scenario_duration)

    # ── Shut down WS listeners ────────────────────────────────────────────────
    stop_event.set()
    if ws_tasks:
        await asyncio.gather(*ws_tasks, return_exceptions=True)

    elapsed = time.time() - t0
    total_pushes = sum(push_counters.values())
    push_summary = "  " + "  ".join(
        f"{lbl}: {n}" for lbl, n in push_counters.items()
    ) if push_counters else "  (no WS listeners)"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Session complete  ({elapsed:.0f}s elapsed)
║
║  Pushes received (= claims logged):
║{push_summary}
║  Total: {total_pushes}
║
║  To see metrics:   just stop && just metrics
║  To re-run:        uv run python scripts/simulate_session.py
╚══════════════════════════════════════════════════════════════╝
""")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="APEX Phase 0 session simulator (WebSocket delivery path). "
                    "For pull-mode supervision use vault_agent.py instead.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--duration", "-d", type=int, default=SCENARIO_DURATION,
        help="Duration per scenario in seconds (total ≈ 3×)",
    )
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    try:
        asyncio.run(main(scenario_duration=_args.duration))
    except KeyboardInterrupt:
        print("\nAborted.")
