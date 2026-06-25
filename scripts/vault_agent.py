"""
Vault Agent — Autonomous evaluation harness for APEX Phase 0.

This script is the primary evaluation driver. It is completely independent
of the APEX pipeline internals — it only interacts via the HTTP/WebSocket API.

What it does
------------
1. Edits files in experiment_corpus/ to generate realistic BSM events that
   drive the APEX pipeline (BSM → IIE → SRS → Retrieval → Buffer).

2. At scheduled intervals, makes GET /context/{subscriber_id} requests
   (pull-mode supervision). These are ground-truth "I would have searched here"
   events. Any pre-fetched content in the buffer is claimed with:
       LtC = t_available - t_need  (negative = APEX was proactive)

3. Keeps WebSocket connections open for all subscribers so push delivery
   is also recorded (parallel claim path).

4. Prints a real-time summary and final metrics link.

Evaluation interpretation
--------------------------
- PRP > 0.65: APEX correctly predicted what the user would need in >65% of cases
- LtC mean < 0 (e.g. -15000 ms): APEX was ready 15 seconds before the user pulled
- Both together prove proactivity: APEX was early AND accurate

Usage
-----
    # Prerequisites (all in separate terminals):
    #   Terminal 1:  just llm
    #   Terminal 2:  APEX_VAULT_PATH=experiment_corpus \\
    #                APEX_INDEX_PATH=experiment_index/experiment just dev
    #   Terminal 3:  just register  (only needed once per DB reset)

    # Run the full evaluation:
    uv run python scripts/vault_agent.py

    # Longer run with custom pull interval:
    uv run python scripts/vault_agent.py --duration 900 --pull-interval 30

    # Single scenario (for debugging):
    uv run python scripts/vault_agent.py --scenario writing --duration 120

    # After completion:
    just stop && just metrics

Options
-------
    --duration      Total session duration in seconds (default: 600 = 10 min)
    --pull-interval Seconds between pull-mode supervision requests (default: 35)
    --scenario      Run only one scenario: writing | debugging | reading | all (default: all)
    --apex-port     APEX server port (default: 8765)
    --vault         Path to experiment corpus (default: experiment_corpus)
    --no-ws         Disable WebSocket listeners (pure pull-mode evaluation)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import NamedTuple

import httpx
import websockets

# ── Configuration ─────────────────────────────────────────────────────────────

VAULT_PATH = Path(os.environ.get("APEX_VAULT_PATH", "experiment_corpus")).expanduser()
APEX_PORT  = int(os.environ.get("APEX_PORT", "8765"))
SUBSCRIBERS_FILE = Path(".phase0_subscribers.json")

# File paths (relative to vault)
_FILES = {
    "writing":   "writing/thesis_draft.md",
    "debugging": "debugging/async_patterns.py",
    "reading":   "reading/literature_notes.md",
}

# ── Content pools (realistic domain content per scenario) ─────────────────────

_WRITING_POOL = [
    """
## 4.1 Behavioral Signal Monitor Design

The BSM is architected as an always-on daemon with three sensing layers.
The macOS implementation uses NSWorkspace for active application detection,
FSEventStream for filesystem events, and IOKit for device state changes.
Each event is normalized into a SignalVector with a fixed schema.
""",
    """
The privacy constraint is fundamental: the BSM reads OS-level metadata only.
File names, timestamps, and application identifiers are observable;
document content is never read. This is enforced architecturally — the
BSM has no file reading code path whatsoever.
""",
    """
## 4.2 Intent Inference Engine — Hybrid Design

The IIE uses a two-path architecture to balance speed and accuracy.
The HeuristicGate handles known high-confidence patterns in <1ms using
a pre-computed dictionary lookup. Ambiguous signals fall through to
Phi-3.5 Mini (INT4, Ollama) which runs on the Neural Engine in <20ms.
""",
    """
The critical architectural invariant: the IIE never produces a text query string.
Its output q̂ is always a dense embedding vector. This eliminates the
natural language query formulation step entirely and is the primary
architectural departure from all existing RAG systems.
""",
    """
## 4.3 Speculative Retrieval Scheduler

The SRS implements the Goldilocks timing principle from Seo et al. (CHI 2025).
The threshold τ is not hardcoded — it is a learned, per-user, per-domain value
maintained by TauCalibrator which runs as a background async task every 120s.
""",
    """
## 4.4 Evaluation Protocol and Metrics

Proactive Retrieval Precision (PRP) measures the fraction of pre-fetched
context that the user actually needed. The ground truth comes from pull-mode
supervision: the vault agent makes deliberate GET /context/{id} requests at
known T_need times. Negative LtC confirms APEX was proactive.
""",
]

_DEBUGGING_POOL = [
    """
# asyncio.gather with error handling
async def safe_gather(*coros):
    results = await asyncio.gather(*coros, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise ExceptionGroup("gather errors", errors)
    return results
""",
    """
# Bounded semaphore for Ollama rate limiting
_ollama_sem = asyncio.Semaphore(3)  # max 3 concurrent Ollama calls

async def embed_with_limit(text: str) -> list[float]:
    async with _ollama_sem:
        response = await asyncio.to_thread(
            ollama.embed, model="all-minilm", input=text
        )
    return response.embeddings[0]
""",
    """
# WebSocket reconnection with exponential backoff
async def resilient_ws_connect(uri: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            async with websockets.connect(uri, ping_interval=20) as ws:
                async for message in ws:
                    yield message
        except (OSError, websockets.exceptions.WebSocketException) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
""",
    """
# DuckDB time-window query pattern
def claim_in_window(con, subscriber_id: str, t_need: float, window_s: float = 60.0):
    row = con.execute(
        \"\"\"SELECT id FROM prefetch_events
           WHERE subscriber_id = ? AND claimed = FALSE
             AND t_available BETWEEN ? AND ?
           ORDER BY t_available DESC LIMIT 1\"\"\",
        [subscriber_id, t_need - window_s, t_need]
    ).fetchone()
    if row:
        con.execute(
            \"UPDATE prefetch_events SET claimed=TRUE, "
            "latency_ms=(t_available-?) * 1000 WHERE id=?\",
            [t_need, row[0]]
        )
""",
    """
# FastAPI startup inject pattern (avoids circular imports)
_coordinator = None  # set by init_app()

def init_app(coordinator):
    global _coordinator
    _coordinator = coordinator
    coordinator._push_callback = _push_context

async def _push_context(subscriber_id: str) -> bool:
    sockets = _connections.get(subscriber_id, set())
    if not sockets:
        return False
    dead = set()
    delivered = False
    for ws in sockets:
        try:
            await ws.send_text(format_context(subscriber_id))
            delivered = True
        except Exception:
            dead.add(ws)
    for ws in dead:
        sockets.discard(ws)
    return delivered
""",
]

_READING_POOL = [
    """
## Notes — Goldilocks Timing (Seo et al., CHI 2025)

Peak utilisation at -10s before action (78%). Window: 5–30s.
APEX buffer TTL=60s, pull interval=35s targets the middle of this window.
LtC target: -5000ms to -30000ms for Phase 0 sessions.
""",
    """
## Notes — HNSW Algorithm

ef=50 gives 98% recall at ~1.5ms on Apple M1 for 13K vectors.
RRF k=60 from Cormack et al. (SIGIR 2009) — standard parameter.
Label-filtered BM25 critical for label coherence between IIE output and retrieval.
""",
    """
## Notes — ContextAgent (NeurIPS 2025)

Primary prior work. Cloud-dependent (violates on-device constraint).
800ms–2.1s delivery latency vs APEX <50ms.
Does not support multi-subscriber. No MCP integration.
APEX improves on all four dimensions.
""",
    """
## Notes — Phi-3.5 Mini Quantization

INT4 (Q4_K_M): 2.3 GB, 15–40 tok/s on ANE.
Semantic translator only — never generates new factual content.
Empty chunk set → empty output (no hallucination by construction).
""",
    """
## Notes — MCP Specification

Standard MCP: synchronous request-response. No push semantics.
APEX extension: WebSocket /stream/{id} for server-initiated push.
Backward compatible: /context/{id} works as standard MCP resource.
First MCP server with proactive push channel.
""",
]


# ── Scenario state ────────────────────────────────────────────────────────────

class ScenarioStats(NamedTuple):
    name: str
    edits: int
    pulls: int
    pull_hits: int   # pulls that found warm context (proactive hit)
    pull_misses: int  # pulls that found empty buffer (proactive miss)
    ws_pushes: int   # pushes received via WebSocket


# ── HTTP client ───────────────────────────────────────────────────────────────

def _base_url(port: int) -> str:
    return f"http://localhost:{port}"


async def _pull_context(
    client: httpx.AsyncClient,
    subscriber_id: str,
    port: int,
) -> bool:
    """
    Pull context via GET /context/{subscriber_id}.

    Returns True if the buffer had warm content (proactive hit),
    False if the buffer was empty (proactive miss).

    The APEX server calls claim_via_pull() on any non-empty response,
    recording LtC = t_available - t_now (negative = proactive).
    """
    try:
        resp = await client.get(
            f"{_base_url(port)}/context/{subscriber_id}",
            timeout=30.0,  # phi3.5 can take 10-15 s; must not time out before claim fires
        )
        if resp.status_code == 200:
            body = resp.json()
            return bool(body.get("context", "").strip())
        return False
    except (httpx.RequestError, httpx.HTTPStatusError):
        return False


# ── File edit helpers ─────────────────────────────────────────────────────────

def _append(filepath: Path, content: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as f:
        f.write(content)


# ── WebSocket listener ────────────────────────────────────────────────────────

async def _ws_listener(
    label: str,
    subscriber_id: str,
    port: int,
    stop_event: asyncio.Event,
    push_counter: dict[str, int],
) -> None:
    uri = f"ws://localhost:{port}/stream/{subscriber_id}"
    while not stop_event.is_set():
        try:
            async with websockets.connect(uri, ping_interval=20) as ws:
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=1.0)
                        push_counter[label] = push_counter.get(label, 0) + 1
                    except asyncio.TimeoutError:
                        continue
        except (OSError, websockets.exceptions.WebSocketException):
            if not stop_event.is_set():
                await asyncio.sleep(2.0)


# ── Scenario runners ──────────────────────────────────────────────────────────

async def run_scenario(
    name: str,
    content_pool: list[str],
    duration_s: int,
    pull_interval_s: int,
    subscribers: dict[str, str],
    port: int,
    edit_interval_s: float,
    label: str,
) -> ScenarioStats:
    """
    Run one scenario: interleave file edits with pull-mode supervision.

    Parameters
    ----------
    name            Scenario name for display.
    content_pool    Rotating list of content blocks to append.
    duration_s      Scenario duration in seconds.
    pull_interval_s Seconds between pull requests.
    subscribers     Dict of label → subscriber_id from .phase0_subscribers.json
    port            APEX server port.
    edit_interval_s Seconds between file edits.
    label           Subscriber label to pull for (e.g. "ide_plugin").
    """
    filepath = VAULT_PATH / _FILES[name]
    filepath.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n── Scenario: {name} ({duration_s}s, pull every {pull_interval_s}s) ──")

    edits = 0
    pulls = 0
    hits = 0
    misses = 0
    idx = 0
    last_pull = time.time()
    start = time.time()

    # Use the ide_plugin subscriber for writing/debugging; research_assistant for reading
    pull_label = "research_assistant" if name == "reading" else "ide_plugin"
    pull_sub_id = subscribers.get(pull_label, "")

    async with httpx.AsyncClient() as client:
        while time.time() - start < duration_s:
            now = time.time()

            # File edit
            block = content_pool[idx % len(content_pool)]
            _append(filepath, block)
            edits += 1
            elapsed = now - start
            remaining = duration_s - elapsed
            print(
                f"  [{name}] edit={edits:>3}  elapsed={elapsed:>5.0f}s  "
                f"remaining={remaining:>4.0f}s",
                end="\r",
            )
            idx += 1

            # Pull-mode supervision at scheduled intervals
            if pull_sub_id and (now - last_pull) >= pull_interval_s:
                hit = await _pull_context(client, pull_sub_id, port)
                pulls += 1
                if hit:
                    hits += 1
                    print(
                        f"\n  [pull ✓] {pull_label} → warm context found  "
                        f"(hit {hits}/{pulls})",
                    )
                else:
                    misses += 1
                    print(
                        f"\n  [pull ✗] {pull_label} → buffer empty  "
                        f"(miss {misses}/{pulls})",
                    )
                last_pull = now

            await asyncio.sleep(edit_interval_s)

    print(f"\n  {name} done — {edits} edits, {pulls} pulls ({hits} hits / {misses} misses)")
    return ScenarioStats(name, edits, pulls, hits, misses, 0)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    port = args.apex_port
    duration = args.duration
    pull_interval = args.pull_interval
    scenario_filter = args.scenario
    vault = Path(args.vault).expanduser()

    # Override global VAULT_PATH for file writes
    global VAULT_PATH
    VAULT_PATH = vault

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         APEX Vault Agent — Phase 0 Evaluation Harness       ║
╠══════════════════════════════════════════════════════════════╣
║  Vault:         {str(vault):<45}║
║  Duration:      {duration}s total                                    ║
║  Pull interval: {pull_interval}s                                             ║
║  Port:          {port}                                           ║
╠══════════════════════════════════════════════════════════════╣
║  What this proves:                                           ║
║  • PRP > 0.65 → APEX accurately predicted user needs        ║
║  • LtC mean < 0 → APEX was ready BEFORE the user pulled     ║
║  Together: proactive AND accurate                            ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Pre-flight: verify server is reachable and subscribers are registered ──
    async with httpx.AsyncClient() as probe:
        try:
            r = await probe.get(f"{_base_url(port)}/state", timeout=5.0)
            if r.status_code != 200:
                print(f"\n  [abort] Server at port {port} returned {r.status_code}.")
                print("  Is the daemon running?  →  just dev\n")
                return
        except httpx.ConnectError:
            print(f"\n  [abort] Cannot connect to APEX daemon on port {port}.")
            print("  Start it first:  APEX_VAULT_PATH=experiment_corpus "
                  "APEX_INDEX_PATH=experiment_index/experiment just dev\n")
            return

    # Load subscriber IDs
    subscribers: dict[str, str] = {}
    if SUBSCRIBERS_FILE.exists():
        subscribers = json.loads(SUBSCRIBERS_FILE.read_text())

        # Verify each subscriber ID is still registered with the running daemon.
        # A 404 means the daemon restarted and _profiles was cleared — stale IDs.
        async with httpx.AsyncClient() as probe:
            stale: list[str] = []
            for lbl, sid in subscribers.items():
                try:
                    r = await probe.get(f"{_base_url(port)}/context/{sid}", timeout=5.0)
                    if r.status_code == 404:
                        stale.append(lbl)
                except Exception:
                    pass

        if stale:
            print(
                f"\n  [abort] Subscriber IDs are stale (404) for: {', '.join(stale)}\n"
                "  The daemon was restarted but subscribers were not re-registered.\n"
                "\n"
                "  Fix:\n"
                "    just reset    ← wipes old DB + subscriber file\n"
                "    APEX_VAULT_PATH=experiment_corpus "
                "APEX_INDEX_PATH=experiment_index/experiment just dev\n"
                "    just register ← MANDATORY after every dev start\n"
                "    just eval\n"
            )
            return

        print(f"  Subscribers: {', '.join(subscribers.keys())}")
    else:
        print(
            "  [warn] .phase0_subscribers.json not found.\n"
            "  Run `just register` first. Continuing without pull claims."
        )

    # Start WebSocket listeners
    stop_event = asyncio.Event()
    push_counters: dict[str, int] = {}
    ws_tasks: list[asyncio.Task] = []

    if not args.no_ws and subscribers:
        print("\n  Connecting WebSocket listeners…")
        for lbl, sid in subscribers.items():
            task = asyncio.create_task(
                _ws_listener(lbl, sid, port, stop_event, push_counters),
                name=f"ws-{lbl}",
            )
            ws_tasks.append(task)
        await asyncio.sleep(2.0)  # allow WS connections to establish
        print(f"  {len(ws_tasks)} WebSocket connection(s) active.\n")

    if not args.auto_start:
        input("  Press Enter to start (or Ctrl-C to abort) …\n")
    else:
        print("  Auto-starting in 1 s …")
        await asyncio.sleep(1.0)

    t0 = time.time()
    all_stats: list[ScenarioStats] = []

    # Scenario definitions: (name, pool, duration_fraction, edit_interval, pull_label)
    scenario_defs = [
        ("writing",   _WRITING_POOL,   0.40, 5.0),
        ("debugging", _DEBUGGING_POOL, 0.35, 4.0),
        ("reading",   _READING_POOL,   0.25, 20.0),
    ]

    for name, pool, frac, edit_interval in scenario_defs:
        if scenario_filter != "all" and scenario_filter != name:
            continue
        scen_duration = int(duration * frac) if scenario_filter == "all" else duration
        stats = await run_scenario(
            name=name,
            content_pool=pool,
            duration_s=scen_duration,
            pull_interval_s=pull_interval,
            subscribers=subscribers,
            port=port,
            edit_interval_s=edit_interval,
            label=name,
        )
        all_stats.append(stats)

        if scenario_filter == "all" and name != "reading":
            print("  Scenario gap (10s)…")
            await asyncio.sleep(10)

    # Shut down WebSocket listeners
    stop_event.set()
    if ws_tasks:
        await asyncio.gather(*ws_tasks, return_exceptions=True)

    elapsed = time.time() - t0
    total_edits   = sum(s.edits for s in all_stats)
    total_pulls   = sum(s.pulls for s in all_stats)
    total_hits    = sum(s.pull_hits for s in all_stats)
    total_misses  = sum(s.pull_misses for s in all_stats)
    total_ws      = sum(push_counters.values())

    pull_prp = (total_hits / total_pulls * 100) if total_pulls > 0 else float("nan")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Session complete  ({elapsed:.0f}s elapsed)
╠══════════════════════════════════════════════════════════════╣
║  File edits       : {total_edits}
║  Pull requests    : {total_pulls}  ({total_hits} hits / {total_misses} misses)
║  Pull hit rate    : {pull_prp:.1f}%  (preliminary PRP estimate)
║  WS pushes recv'd : {total_ws}
╠══════════════════════════════════════════════════════════════╣
║  Per-scenario summary:
""")
    for s in all_stats:
        sr = (s.pull_hits / s.pulls * 100) if s.pulls else float("nan")
        print(f"║    {s.name:<12} edits={s.edits:<4} pulls={s.pulls:<3} "
              f"hits={s.pull_hits:<3} ({sr:.0f}%)")

    print(f"""║
║  WS pushes by subscriber:
""")
    for lbl, cnt in push_counters.items():
        print(f"║    {lbl:<24}: {cnt}")

    print(f"""╠══════════════════════════════════════════════════════════════╣
║  Next steps:                                                 ║
║    just stop && just metrics                                 ║
║                                                              ║
║  For detailed LtC distribution:                              ║
║    just metrics --all                                        ║
╚══════════════════════════════════════════════════════════════╝
""")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="APEX Vault Agent — Phase 0 evaluation harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--duration", "-d", type=int, default=600,
        help="Total session duration in seconds",
    )
    p.add_argument(
        "--pull-interval", "-p", type=int, default=35,
        help="Seconds between pull-mode supervision requests",
    )
    p.add_argument(
        "--scenario", "-s", default="all",
        choices=["all", "writing", "debugging", "reading"],
        help="Run all scenarios or a single one",
    )
    p.add_argument(
        "--apex-port", type=int, default=APEX_PORT,
        help="APEX server port",
    )
    p.add_argument(
        "--vault", default=str(VAULT_PATH),
        help="Path to experiment corpus directory",
    )
    p.add_argument(
        "--no-ws", action="store_true",
        help="Disable WebSocket listeners (pure pull-mode evaluation)",
    )
    p.add_argument(
        "--auto-start", action="store_true",
        help="Start immediately without the Enter prompt (for unattended eval runs)",
    )
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("\nAborted.")
