# ADR-002: Phase 0 — Local macOS Integration Environment

**Status:** Accepted
**Date:** 2026-04-07
**Deciders:** Raneem Khafagy (thesis author)

---

## Context

The previous hardware evaluation plan began at the Android Emulator (AVD ARM64),
calling it Phase 1. This created a gap: there was no defined environment for
validating the full end-to-end pipeline *before* introducing the complexity of
cross-compilation, ARM emulation, or hardware deployment.

The consequences of that gap are:
- The first time the full daemon runs is inside an emulator, making it hard to
  distinguish APEX bugs from emulator artifacts.
- Adding new adapters or consumers has no tested workflow — the API seams exist
  in code but there is no "do this to add one" runbook.
- Preliminary PRP and LtC data cannot be collected until the emulator is set up.
- The `main.py` stub has never been exercised as a daemon.

This ADR defines **Phase 0** as a local macOS developer-machine phase that closes
this gap. It runs entirely on the thesis author's development machine, uses the
same Ollama instance already used for testing, and requires no new hardware or
cross-compilation toolchain.

---

## Decision

**Insert Phase 0 (Local macOS Integration) before the current Phase 1 (Android Emulator).**

The full phase numbering becomes:

| Phase | Platform | Primary Purpose | Metrics Produced |
|---|---|---|---|
| **0** | **macOS dev machine** | **End-to-end integration, adapter/consumer authoring** | **PRP (preliminary), LtC (preliminary)** |
| 1 | Android Emulator (AVD ARM64) | Functional correctness on ARM | None — no NPU or power management |
| 2 | NVIDIA Jetson Orin | Real embedded edge hardware | Latency, battery (tegrastats) |
| 3 | iPhone 11/13/17 Pro Max + MacBook/Mac Mini M1 | **Primary thesis results** | PRP, LtC, BI, DPS |

Phase 3 remains the primary result set. Phase 0 metrics are explicitly labelled
"development environment" in the thesis and are not used for hardware claims.

---

## Phase 0 Definition

### Platform

Development MacBook running macOS. Same machine where Ollama is installed.
No cross-compilation. No hardware constraints. Ollama running at `localhost:11434`.

### What Phase 0 validates

| Concern | Validated in Phase 0? | Notes |
|---|---|---|
| `main.py` daemon wires up correctly | ✅ | First real execution of the runtime |
| BSM → Coordinator → MCP push path | ✅ | End-to-end on real file changes |
| Knowledge base populated and queryable | ✅ | `just ingest` from ApexVault |
| τ calibration feedback loop updates τ | ✅ | Claim rate observed over sessions |
| Multi-adapter pipeline (3 domains) | ✅ | factory.py + research.py authored here |
| Multi-consumer pipeline (N profiles) | ✅ | Multiple ConsumerProfiles registered |
| PRP > 0.65 achievable at all | ✅ (preliminary) | Not the thesis number, but validates the signal |
| LtC negative mean achievable at all | ✅ (preliminary) | Same caveat |
| Battery Impact (BI) | ❌ | macOS power APIs needed for Phase 3; not claimed here |
| Latency at NPU level | ❌ | No NPU on dev machine; not claimed here |

### Entry criteria (Phase 0 starts when)

- [ ] `main.py` runs the full daemon (BSM + Coordinator + MCP server)
- [ ] `just ingest` successfully populates the HNSW and BM25 indexes from ApexVault
- [ ] `just serve` starts the FastAPI server and accepts WebSocket connections
- [ ] τ calibration feedback loop is implemented and calls `scheduler.update_tau()`
- [ ] At least `productivity.py` adapter is connected to the daemon

### Exit criteria (Phase 0 complete when)

- [ ] All three domain adapters (productivity, factory, research) are running
- [ ] At least three distinct ConsumerProfiles are registered and receiving pushes
- [ ] One full 30-minute session produces PRP and mean LtC via `analytics/store.py`
- [ ] τ has been updated at least once by the calibration loop (logged in DuckDB)
- [ ] `just test` passes (`pytest -v` — all unit + integration tiers)

---

## Why Not Merge Phase 0 Into Phase 1

Phase 1 (Android Emulator) is ARM64 — a different architecture from the dev machine.
APEX's inference pipeline (Ollama + hnswlib) behaves differently on ARM because:

- `hnswlib` SIMD paths differ (x86 AVX vs ARM NEON)
- Ollama model quantization interacts with the instruction set
- `watchfiles` uses different OS event backends (kqueue on macOS vs inotify on Android)

Conflating "does the pipeline work at all" with "does it work on ARM" makes failures
ambiguous. Phase 0 separates integration correctness from architecture portability.

---

## Adapter Authoring Contract

Every domain adapter is exactly one class. Here is the complete contract, derived
from `ProductivityAdapter` as the reference implementation.

### The minimal contract

```python
# apex/adapters/<domain>.py
from apex.adapters.base import SignalAdapter, SignalVector

class MyDomainAdapter(SignalAdapter):
    def __init__(self, ...):
        # Any domain-specific initialization.
        # Never call open() here. Never start threads here.
        pass

    def observe(self) -> SignalVector:
        # Return a SignalVector snapshot of the current environment.
        # MUST: read only structural metadata (no file content)
        # MUST: return within a few ms (called in a hot loop)
        # MUST: be idempotent
        return SignalVector(
            source_id=...,          # str: hash of identity context
            content_hash=...,       # str: hex digest of structural state
            activity_type=...,      # str: semantic label
            velocity_metric=...,    # float [0,1]: intensity of current activity
            temporal_proximity=..., # float [0,1]: urgency from time pressure
            urgency_flag=...,       # bool: True only for safety-critical events
        )
```

### Field semantics by domain

| Field | Productivity | Factory | Research |
|---|---|---|---|
| `source_id` | SHA256 of app name | SHA256 of machine_id + sensor_id | SHA256 of doc_path + coauthor_set |
| `content_hash` | SHA256 of (filename, mtime) pairs | SHA256 of latest sensor readings | SHA256 of citation list + draft_hash |
| `activity_type` | "writing" \| "debugging" \| "reading" \| "idle" | "normal_operation" \| "anomaly_event" \| "maintenance_window" | "drafting" \| "reviewing_lit" \| "revising" \| "idle" |
| `velocity_metric` | Decaying recency of last file change | Normalized deviation from baseline sensor reading | Words-per-minute / typing burst intensity |
| `temporal_proximity` | Same as velocity (proxy) | Time-to-maintenance-window (normalized) | Time-to-deadline / review-due (normalized) |
| `urgency_flag` | Always `False` | `True` when anomaly exceeds σ threshold | Always `False` |

### Wiring an adapter into the daemon (Phase 0 checklist)

```python
# In main.py (or a domain-specific launcher):

# 1. Instantiate the adapter
adapter = FactoryAdapter(sensor_config=...)

# 2. Pass it to SignalMonitor
monitor = SignalMonitor(adapter=adapter, watch_path=APEX_WATCH_PATH)

# 3. Wire callback to the coordinator
monitor.register_callback(
    lambda sv: asyncio.create_task(coordinator.process_signal_all(sv))
)

# 4. Add all subscriber IDs the daemon will serve
coordinator.add_subscriber("factory_dashboard_app")
coordinator.add_subscriber("maintenance_alert_system")

# That's it. Zero changes to the core pipeline.
```

### FactoryAdapter sketch (Phase 0 target)

```python
# apex/adapters/factory.py
"""
Factory / industrial signal adapter.
Reads sensor telemetry from a local state file (MQTT bridge, OPC-UA proxy,
or simulated CSV). Never reads raw process data content — only deviation metrics.
urgency_flag=True when anomaly_delta exceeds safety threshold.
"""
import hashlib
import json
import time
from pathlib import Path
from apex.adapters.base import SignalAdapter, SignalVector

ANOMALY_THRESHOLD = 3.0   # standard deviations from baseline

class FactoryAdapter(SignalAdapter):
    def __init__(
        self,
        sensor_state_path: str,   # JSON file written by MQTT/OPC-UA bridge
        machine_id: str = "machine_01",
        baseline_sigma: float = 1.0,
    ):
        self._state_path = Path(sensor_state_path)
        self._machine_id = machine_id
        self._baseline_sigma = baseline_sigma

    def observe(self) -> SignalVector:
        # Read the last sensor snapshot (structural metadata, not process content)
        try:
            state = json.loads(self._state_path.read_text())
        except Exception:
            state = {"deviation": 0.0, "time_to_maintenance": 1.0, "sensor_id": "unknown"}

        deviation = float(state.get("deviation", 0.0))
        maintenance_proximity = float(state.get("time_to_maintenance", 1.0))
        sensor_id = str(state.get("sensor_id", "unknown"))

        # velocity = normalized anomaly deviation
        velocity = min(1.0, abs(deviation) / max(ANOMALY_THRESHOLD, 0.01))

        # urgency_flag fires when deviation exceeds the safety threshold
        urgency = abs(deviation) >= ANOMALY_THRESHOLD * self._baseline_sigma

        source_id = hashlib.sha256(
            f"{self._machine_id}:{sensor_id}".encode()
        ).hexdigest()[:16]

        content_hash = hashlib.sha256(
            f"{deviation:.3f}:{maintenance_proximity:.3f}".encode()
        ).hexdigest()[:16]

        activity_type = (
            "anomaly_event" if urgency
            else "maintenance_window" if maintenance_proximity > 0.8
            else "normal_operation"
        )

        return SignalVector(
            source_id=source_id,
            content_hash=content_hash,
            activity_type=activity_type,
            velocity_metric=velocity,
            temporal_proximity=maintenance_proximity,
            urgency_flag=urgency,
        )
```

### ResearchAdapter sketch (Phase 0 target)

```python
# apex/adapters/research.py
"""
PureSearch / academic research signal adapter.
Reads draft state (word count delta, citation list, co-author presence)
from filesystem metadata only. Never reads document content.
"""
import hashlib
import os
import time
from pathlib import Path
from apex.adapters.base import SignalAdapter, SignalVector

class ResearchAdapter(SignalAdapter):
    def __init__(
        self,
        draft_dir: str,            # directory containing .tex / .md draft files
        citations_file: str,       # .bib or .json citations list (metadata only)
        typing_window_sec: float = 30.0,
    ):
        self._draft_dir = Path(draft_dir)
        self._citations_file = Path(citations_file)
        self._typing_window = typing_window_sec
        self._last_change: float = time.time()
        self._last_word_count: int = 0

    def notify_change(self) -> None:
        self._last_change = time.time()

    def observe(self) -> SignalVector:
        # Word count proxy: total bytes in draft dir (metadata, not content)
        try:
            total_bytes = sum(
                e.stat().st_size for e in os.scandir(self._draft_dir)
                if e.name.endswith((".tex", ".md", ".txt"))
            )
        except Exception:
            total_bytes = 0

        # Citation count from file size proxy (never reading content)
        try:
            cit_size = self._citations_file.stat().st_size
        except Exception:
            cit_size = 0

        # Velocity: typing burst intensity (recency of last change)
        elapsed = time.time() - self._last_change
        velocity = max(0.0, 1.0 - elapsed / self._typing_window)

        # source_id from draft directory identity
        source_id = hashlib.sha256(str(self._draft_dir).encode()).hexdigest()[:16]

        # content_hash: hash of (filename, mtime, size) — no content
        try:
            entries = sorted(
                (e.name, e.stat().st_mtime, e.stat().st_size)
                for e in os.scandir(self._draft_dir)
            )
        except Exception:
            entries = []
        content_hash = hashlib.sha256(str(entries).encode()).hexdigest()[:16]

        # Activity type: classify by typing velocity
        if velocity >= 0.7:
            activity_type = "drafting"
        elif cit_size > 0 and velocity < 0.2:
            activity_type = "reviewing_lit"
        elif velocity > 0:
            activity_type = "revising"
        else:
            activity_type = "idle"

        return SignalVector(
            source_id=source_id,
            content_hash=content_hash,
            activity_type=activity_type,
            velocity_metric=velocity,
            temporal_proximity=velocity,  # proxy: high activity = deadline near
            urgency_flag=False,  # research domain is not safety-critical
        )
```

---

## Consumer Authoring Contract

A consumer is any application that registers a `ConsumerProfile` with the APEX
MCP server. Adding a consumer requires **zero code changes** to APEX — only a
POST request and a WebSocket connection.

### Registration (one-time setup)

```python
# Any language. This example uses httpx (Python).
import httpx

resp = httpx.post("http://localhost:8765/subscribe", json={
    # Output shape
    "output_format":     "markdown",          # json | markdown | plain-text | voice | structured-alert
    "vocabulary_level":  "technical",         # technical | domain-expert | layman
    "verbosity":         "concise",           # concise | standard | detailed
    "citation_style":    "inline",            # inline | footnote | none
    "max_context_tokens": 512,

    # Behavioral preferences
    "autonomy_level":    "assistive",         # suggestive | assistive | autonomous
    "goal_horizon":      "short",             # short | mid | long
    "interaction_style": "soft-interrupt",    # ambient | soft-interrupt | hard-interrupt | conversational

    # Optional: constrain output to a JSON Schema (structured-alert format)
    "domain_schema": None,
})

subscriber_id = resp.json()["subscriber_id"]
# Store this ID — it identifies your buffer partition and push channel
```

### Receiving pushes (primary path)

```python
import asyncio, websockets

async def receive_context(subscriber_id: str):
    uri = f"ws://localhost:8765/stream/{subscriber_id}"
    async with websockets.connect(uri) as ws:
        while True:
            context = await ws.recv()
            # context is the LLM-formatted string matching your ConsumerProfile
            handle_context(context)

asyncio.run(receive_context(subscriber_id))
```

### Pulling on demand (fallback path)

```python
import httpx

resp = httpx.get(f"http://localhost:8765/context/{subscriber_id}")
context = resp.json()["context"]   # empty string if buffer has nothing yet
```

### Unregistering

```python
httpx.delete(f"http://localhost:8765/subscribe/{subscriber_id}")
# Buffer partition is cleared. WebSocket connections are closed.
```

### Consumer profiles for Phase 0 evaluation

Register these three profiles to exercise the full multi-consumer pipeline:

```python
PROFILES = [
    # Consumer A: IDE plugin — technical markdown, ambient
    {
        "output_format": "markdown", "vocabulary_level": "technical",
        "verbosity": "concise", "citation_style": "inline",
        "interaction_style": "ambient", "max_context_tokens": 512,
    },
    # Consumer B: Factory dashboard — structured alert, hard interrupt
    {
        "output_format": "structured-alert", "vocabulary_level": "domain-expert",
        "verbosity": "concise", "citation_style": "none",
        "interaction_style": "hard-interrupt", "max_context_tokens": 256,
        "domain_schema": {
            "type": "object",
            "properties": {
                "severity":  {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                "action":    {"type": "string"},
                "context":   {"type": "string"},
            },
            "required": ["severity", "action"],
        },
    },
    # Consumer C: Research writing assistant — detailed markdown, conversational
    {
        "output_format": "markdown", "vocabulary_level": "domain-expert",
        "verbosity": "detailed", "citation_style": "footnote",
        "interaction_style": "conversational", "max_context_tokens": 1024,
    },
]
```

---

## Phase 0 Session Runbook

```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: Populate knowledge base (once)
just ingest

# Terminal 3: Run the full APEX daemon
just dev          # = ollama serve + uvicorn apex.server:app + monitor

# Terminal 4: Register consumers and open push channels
uv run python scripts/register_consumers.py   # registers the three profiles above

# Terminal 5: Trigger file changes in ApexVault to generate signals
touch ~/Documents/ApexVault/debug_session.py   # triggers debugging signal
touch ~/Documents/ApexVault/draft_intro.md     # triggers writing signal

# Terminal 6: Check metrics after session
uv run python scripts/print_metrics.py         # queries DuckDB for PRP + LtC
```

---

## Consequences

### What becomes easier
- The daemon has been exercised before any hardware phase, so Phase 1 failures
  are hardware/architecture issues, not pipeline bugs.
- `factory.py` and `research.py` adapters are written and tested in a tight
  feedback loop on the dev machine before targeting the emulator.
- Multi-consumer DPS evaluation can be prototyped (annotate relevance in Phase 0
  sessions) before committing to a full Phase 3 evaluation protocol.
- τ calibration is validated as converging before the hardware phases, which
  removes a confound from the hardware evaluation.

### What becomes harder
- One more phase to document in the thesis. Mitigated by keeping Phase 0 clearly
  labelled as "development environment" with no hardware performance claims.
- Phase 0 metrics (PRP, LtC) are preliminary. They cannot be compared to Phase 3
  results directly because macOS on dev hardware ≠ target evaluation hardware.

### Phase 0 must NOT produce
- Any latency claims (macOS dev machine is not evaluation hardware)
- Any battery impact claims (no power budget constraints)
- Any Neural Engine or Core ML metrics

---

## Updated Phase Rule in CLAUDE.md

```
Phase 0 results are NEVER used for latency or battery claims.
Phase 0 PRP/LtC are labelled "development environment — preliminary only".
Phase 3 is the primary result set. A13→A18 Pro sweep = Neural Engine generation curve.
```
