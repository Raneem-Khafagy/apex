# APEX — Technical Overview

**Application-agnostic Proactive Edge-native conteXt pushing**
Version 0.1.0 — Phase 0 (Local Integration)

---

## 1. What APEX Is

APEX is an OS-level daemon that watches what a user is doing, infers what information they will need next, and pushes it to any subscribing application — before the user asks. No cloud. No explicit query. Everything runs locally on the device.

### The Focused Research Problem

APEX addresses one problem from three angles that must all hold simultaneously:

> **Can proactive context delivery run entirely on-device, triggered by OS-level behavioral
> patterns, and serve any application domain without bespoke integration?**

| Research Angle | The Question | Primary Metric |
|---|---|---|
| **Proactive delivery** | Can behavioral signals alone trigger high-precision retrieval before the user asks — with no text query at any point? | PRP > 0.65, LtC negative mean |
| **On-device** | Can the full pipeline run within the power and latency budget of edge hardware? | BI < 15%, IIE < 20ms |
| **Application-agnostic** | Can one context engine serve heterogeneous consumers simultaneously using a standardized protocol? | DPS > 0.75, zero core changes per new consumer |

All three must be true at the same time. A system that is proactive but cloud-dependent is not APEX. A system that is on-device but application-specific is not APEX. The intersection is the contribution.

### The Core Inversion

Every existing retrieval system waits to be asked. APEX never waits.

```
Traditional RAG (Pull)
  User types query
    → Application sends request
      → RAG retrieves
        → Application displays result

APEX (Push)
  OS detects behavioral signal
    → APEX infers intent
      → Pre-fetches from local knowledge base
        → Pushes to all subscribers simultaneously
```

No text query enters the pipeline at any stage. The triggering signal is always a behavioral observation from the OS. The IIE output is always a dense vector — never a text string.

---

## 2. Research Claims

These are the three claims the system must demonstrate, mapped to evaluation phases and metrics.

### Claim 1 — Proactive Delivery (behavioral signal triggering)

**Claim:** OS-level behavioral signals (file events, app state, typing velocity, temporal proximity) are sufficient to trigger proactive information retrieval with PRP > 0.65 and negative mean LtC, without any text query at any pipeline stage.

**Sub-claim:** A continuously self-calibrating confidence threshold τ (per user, per domain) achieves higher PRP than a static threshold, because optimal delivery timing is user- and domain-specific.

**Evidence required:** Pull-mode supervision sessions across all three domains. PRP measured as `claimed / total prefetches`. LtC measured as `t_available − t_claimed` in milliseconds (negative = proactive).

**Evaluation phase:** Phase 0 (preliminary), Phase 3 (primary thesis result).

---

### Claim 2 — On-Device (edge-native execution)

**Claim:** The full APEX pipeline — always-on BSM monitoring, per-event IIE inference, per-event retrieval, per-push LLM formatting — runs within 15% battery overhead and delivers IIE inference in under 20ms on edge hardware ranging from Jetson Orin NX to iPhone A13–A18 Pro.

**Sub-claim:** INT4 quantization of Phi-3.5 Mini does not produce a statistically significant degradation in DPS for the constrained semantic formatting task.

**Evidence required:** BI measured via `tegrastats` (Jetson) and `IOKit` / iOS Energy Log (iPhone/Mac). Latency profiled at four pipeline timestamps per event.

**Evaluation phase:** Phase 2 (Jetson Orin), Phase 3 (iPhone A13/A15/A18 Pro, Mac M1).

---

### Claim 3 — Application-Agnostic (MCP pub-sub)

**Claim:** The MCP subscription extension enables any MCP-capable application to receive proactive context via a single `POST /subscribe` registration — with per-subscriber output formatting via `ConsumerProfile` — with zero changes to the APEX core pipeline.

**Sub-claim:** ConsumerProfile-driven formatting (runtime profile, no per-consumer fine-tuning) achieves higher DPS than generic context delivery without profiling.

**Evidence required:** Three ConsumerProfile types active simultaneously. DPS measured as `(relevance + format_compliance) / 2`, human-annotated. Overhead measured as latency from retrieval-complete to last subscriber push.

**Evaluation phase:** Phase 0 (multi-subscriber demo), Phase 3 (primary DPS results).

---

## 3. Pipeline Architecture

The pipeline is fixed and domain-blind. All domain logic lives exclusively in the signal adapter layer (Section 4). Adding a new domain requires zero changes to any of the six core components.

```
                    ┌─────────────────────────┐
                    │  Behavioral Signal Monitor│  ← OS metadata only, never content
                    │  monitor/signal_monitor.py│    (watchfiles + NSWorkspace + IOKit)
                    └────────────┬────────────┘
                                 │  SignalVector (6 fields)
                                 ▼
                    ┌─────────────────────────┐
                    │  Intent Inference Engine  │  ← Heuristic gate (<1ms)
                    │  inference/intent_engine  │    + Small LLM fallback (<20ms)
                    └────────────┬────────────┘
                                 │  (q̂, c, ℓ) — dense vector, confidence, label
                                 ▼
                    ┌─────────────────────────┐
                    │  Speculative Scheduler    │  ← Goldilocks timing, τ calibration
                    │  scheduler/speculative.py │
                    └────────────┬────────────┘
                                 │  RETRIEVE or WAIT
                                 ▼
                    ┌─────────────────────────┐
                    │  Hybrid Retrieval Engine  │  ← HNSW dense + BM25 sparse + RRF
                    │  retrieval/ (hnsw+bm25+  │    (sub-10ms, content-blind)
                    │            rrf)           │
                    └────────────┬────────────┘
                                 │  top-k ranked chunks
                                 ▼
                    ┌─────────────────────────┐
                    │  Context Buffer           │  ← TTL queue, per-subscriber partitions
                    │  buffer/context_buffer.py │    (60s TTL, ~200MB budget)
                    └────────────┬────────────┘
                                 │  WebSocket push
                                 ▼
                    ┌─────────────────────────┐
                    │  LLM Adapter Layer        │  ← Phi-3.5 Mini INT4 — semantic translator
                    │  adapter/llm_adapter.py   │    formats chunks per ConsumerProfile
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Subscribing Applications │  ← Any MCP consumer, any domain
                    └─────────────────────────┘
```

**Which component serves which claim:**

| Component | Claim 1 (Proactive) | Claim 2 (On-Device) | Claim 3 (Agnostic) |
|---|:---:|:---:|:---:|
| Behavioral Signal Monitor | ✅ signal source | ✅ always-on power | — |
| Intent Inference Engine | ✅ behavioral → vector | ✅ <20ms target | — |
| Speculative Scheduler | ✅ τ calibration | — | — |
| Hybrid Retrieval Engine | ✅ precision | ✅ sub-10ms | — |
| Context Buffer | ✅ LtC measurement | ✅ memory budget | ✅ per-subscriber isolation |
| LLM Adapter Layer | — | ✅ INT4 on NPU | ✅ ConsumerProfile formatting |
| MCP Server | — | — | ✅ pub-sub protocol |

---

### 3.1 Behavioral Signal Monitor

**File:** `apex/monitor/signal_monitor.py`

Always-on OS daemon. Reads OS-level metadata only — file names, sizes, modification times. Never reads document content. Emits a normalized `SignalVector` at every timestep.

```python
SignalVector(
    source_id          = str,   # hash of active app + context identifier
    content_hash       = str,   # hash of open file names / context type
    activity_type      = str,   # e.g. "drafting", "anomaly_event", "debugging"
    velocity_metric    = float, # normalized activity intensity [0.0–1.0]
    temporal_proximity = float, # proximity to next event/deadline [0.0–1.0]
    urgency_flag       = bool,  # True forces immediate retrieval (τ → 0)
)
```

**Privacy rule:** `observe()` never calls `open()` on any monitored file. Only `os.stat()` metadata is accessed. Enforced by automated privacy sentinel tests.

**Why this serves Claim 1:** The signal vector is the only input to the entire downstream pipeline. No text ever enters. If PRP > 0.65 is achieved, it is achieved solely on the basis of these six fields.

Implementation: `watchfiles` for file-change events; macOS `NSWorkspace` / `FSEventStream` / `IOKit` for app-state metadata.

---

### 3.2 Intent Inference Engine

**File:** `apex/inference/intent_engine.py`

Hybrid design. Fast heuristic gate handles known high-confidence patterns in under 1ms. Ambiguous signals fall through to a small on-device LLM reasoner.

```
Signal arrives
    │
    ▼
Heuristic Gate  (<1ms)
    ├── Known pattern  →  emit (q̂, c=0.9, ℓ) immediately
    └── Ambiguous      →  Small LLM (Phi-3.5 Mini, INT4, <20ms)
                              └── emit (q̂, c, ℓ)
```

**Output triple:**

| Field | Type | Description |
|---|---|---|
| `q̂` | `ndarray` | Dense intent vector in embedding space — never a text string |
| `c` | `float ∈ [0,1]` | Confidence score — input to the τ threshold decision |
| `ℓ` | `str` | Task context label e.g. `"drafting_research"`, `"debugging_python"` |

**Why this serves Claims 1 and 2:** `q̂` is always a dense vector — the behavioral-signal-only constraint is enforced here. The <20ms latency target for the LLM path is the on-device feasibility gate for Claim 2.

Embeddings: `all-MiniLM-L6-v2` (INT8) via Ollama.

---

### 3.3 Speculative Retrieval Scheduler

**File:** `apex/scheduler/speculative.py`

Implements the Goldilocks timing principle (Seo et al., CHI 2025). Controls exactly when retrieval fires — not too early (wasted prefetch), not too late (user already searching).

**Decision policy at each tick:**

```python
if urgency_flag:               → RETRIEVE immediately  (τ forced to 0)
elif c >= τ and buffer_miss:   → RETRIEVE
elif c < τ or buffer_hit:      → WAIT
elif buffer_TTL_expired:       → EVICT
```

**Threshold τ:**

| Condition | τ value | Notes |
|---|---|---|
| Normal operation (initial) | 0.65 | Overridden by calibration after first session |
| Battery < 20% | 0.80 | Hard override — reduces speculative misses on low power |
| `urgency_flag = True` | 0.00 | Hard override — safety-critical events, unconditional |
| Per-user calibrated | learned | Replaces 0.65 after sufficient claim history |

τ is never hardcoded. It is updated continuously at runtime by `TauCalibrator`
(`apex/inference/tau_calibrator.py`), running as an async background task every 120 seconds.
Algorithm: bucket-based claim rate — find the lowest confidence bucket where
`claim_rate ≥ 0.65`, set τ to that bucket's midpoint, clamped to `[0.30, 0.90]`.

**Why this serves Claim 1:** τ calibration is the mechanism that makes PRP measurably better than a fixed threshold. The sub-claim about calibration is proven or disproven by comparing PRP under fixed vs calibrated τ.

---

### 3.4 Hybrid Retrieval Engine

**Files:** `apex/retrieval/hnsw_index.py`, `apex/retrieval/bm25.py`, `apex/retrieval/rrf.py`

Combines dense and sparse retrieval via Reciprocal Rank Fusion.

```
HNSW dense search on q̂   (hnswlib, all-MiniLM-L6-v2 INT8, sub-10ms)
        +
BM25 sparse search on ℓ  (SQLite FTS5, label as query text)
        ↓
RRF fusion:  score(d) = Σ  1 / (k + rank_r(d)),   k = 60
        ↓
Top-k ranked chunks  (default k = 5)
```

The retrieval engine is content-blind. It does not know what domain it serves. The same engine handles productivity, factory, and research queries without modification. The knowledge base is populated at deployment time from the user's local documents via `just ingest`.

**Why this serves Claims 1 and 2:** Sub-10ms retrieval is part of the latency budget for Claim 2. High-precision top-k results (driven by q̂ quality) are the upstream prerequisite for PRP in Claim 1.

---

### 3.5 Context Buffer

**File:** `apex/buffer/context_buffer.py`

TTL-managed priority queue with per-subscriber isolation.

| Property | Value |
|---|---|
| Default TTL | 60 seconds |
| Memory budget | ~200 MB |
| Isolation | Each subscriber gets a completely separate partition |
| Staleness | File-change event → affected chunks marked stale → suppressed from push |

Subscriber A never sees content prefetched for Subscriber B. This is enforced structurally — there is no shared partition.

**Why this serves all three claims:** LtC is measured at buffer push time (Claim 1). Memory budget is an edge resource constraint (Claim 2). Per-subscriber partitions are the isolation mechanism that makes multi-consumer delivery correct (Claim 3).

---

### 3.6 LLM Adapter Layer

**File:** `apex/adapter/llm_adapter.py`

Single shared model: **Phi-3.5 Mini (3.8B params, INT4)** via Ollama.

The LLM is a semantic translator, not a generator. It takes the same retrieved chunks and reformats them into a different output shape for each consumer based on their registered `ConsumerProfile`. It does not generate new factual content. If retrieved chunks are empty, the adapter returns nothing — it never hallucinates fill content.

```python
# Same chunks → different output per subscriber
adapted = llm_adapter.format(chunks, subscriber.profile)
```

Backend: `ollama.generate()` with the subscriber's `ConsumerProfile` as the system prompt. Hardware backends: Metal (macOS), Core ML (iOS), CUDA (Jetson Orin).

**Why this serves Claims 2 and 3:** INT4 on device NPU is the Claim 2 efficiency mechanism. ConsumerProfile-driven formatting is the Claim 3 mechanism — the same model, same chunks, different output per subscriber.

---

## 4. Signal Adapters — Domain Boundary

The adapter layer is the **only** place domain-specific logic lives. Every adapter implements one method:

```python
class MyDomainAdapter(SignalAdapter):
    def observe(self) -> SignalVector:
        # collect domain-specific observations
        # normalize into the fixed SignalVector schema
        return SignalVector(...)
```

**Adding a new domain = one new adapter class. Zero pipeline changes.**

This is the proof of application-agnosticism on the input side: the core pipeline sees only `SignalVector` — it has no knowledge of what domain produced it.

All three adapters are tested against the privacy contract: `observe()` is never permitted to call `open()` on monitored files.

---

### 4.1 ProductivityAdapter

**File:** `apex/adapters/productivity.py`
**Domain:** Personal productivity — writing, debugging, research workflows on a developer machine.

**Signal sources:**
- File system events via `watchfiles` (modification, creation, deletion)
- OS metadata APIs: active application name, open file count, calendar file mtime

**Activity classification:**

| `activity_type` | Condition |
|---|---|
| `writing` | Recent file modifications in watched directory, velocity high |
| `debugging` | Python/code files modified, error patterns in filenames |
| `reading` | File access events without modification |
| `idle` | No events within `idle_threshold` seconds |

**Velocity metric:** Event frequency over a sliding 60-second window, normalized to `[0.0, 1.0]`.

**Temporal proximity:** Derived from calendar file mtime — how recently the user's calendar was updated (proxy for upcoming meeting or deadline pressure).

**`urgency_flag`:** Always `False`. Productivity domain has no safety-critical events.

---

### 4.2 FactoryAdapter

**File:** `apex/adapters/factory.py`
**Domain:** Smart factory / industrial IoT — machine monitoring, anomaly detection, maintenance scheduling.

**Signal source:** A JSON sensor state file written by an MQTT bridge or simulator.

```json
{
  "deviation":           2.7,
  "time_to_maintenance": 0.85,
  "sensor_id":           "pressure_sensor_01",
  "machine_id":          "cnc_lathe_03"
}
```

**Activity classification (priority order):**

| `activity_type` | Condition |
|---|---|
| `anomaly_event` | `abs(deviation) >= ANOMALY_THRESHOLD × baseline_sigma` |
| `maintenance_window` | `time_to_maintenance >= 0.75` |
| `normal_operation` | All other cases |

`ANOMALY_THRESHOLD = 3.0` (3σ). `baseline_sigma` is configurable per machine (default `1.0`).

**Velocity metric:** `min(abs(deviation) / (ANOMALY_THRESHOLD × baseline_sigma), 1.0)` — normalized deviation magnitude.

**`urgency_flag`:** `True` when `abs(deviation) >= ANOMALY_THRESHOLD × baseline_sigma`. This overrides τ completely and forces immediate retrieval. This is the primary safety-critical mechanism for the factory domain — and the most direct test of Claim 1's urgency path.

**Resilience:** Missing or malformed state file falls back to last parsed state, then to safe zero-deviation default. Never raises.

---

### 4.3 ResearchAdapter

**File:** `apex/adapters/research.py`
**Domain:** Academic research writing — draft authoring, literature review, revision, deadline-driven sessions.

**Signal sources:**
- Draft directory: `os.stat()` on all files (mtime, size, count)
- Citations file: mtime of `refs.bib` or equivalent
- Deadline file: mtime of a designated deadline marker file
- Internal `notify_change()` call from the BSM on file modification events

**Velocity metric:** Time-decaying function from the last `notify_change()` call:

```python
elapsed = time.time() - self._last_change_time
velocity = max(0.0, 1.0 - elapsed / typing_window_sec)
```

Immediately after a file change, velocity ≈ 1.0. Decays linearly to 0.0 over `typing_window_sec` (default 60s).

**Activity classification:**

| `activity_type` | Condition |
|---|---|
| `drafting` | `velocity >= 0.7` |
| `reviewing_lit` | `velocity ∈ [0.3, 0.7)` AND citations file recently modified |
| `revising` | `velocity > 0` (below drafting threshold, no recent citation activity) |
| `idle` | `velocity == 0` |

**Temporal proximity:** Derived from deadline file mtime — proximity within a 7-day window. Falls back to `velocity_metric` if file is missing.

**`urgency_flag`:** Always `False`. Research domain has no safety-critical events.

---

## 5. Consumer Profiles — Application-Agnostic Output

Every subscribing application registers a `ConsumerProfile` at subscription time via `POST /subscribe`. The profile persists for the session and tells the LLM Adapter how to reformat retrieved chunks for that specific subscriber.

```python
@dataclass
class ConsumerProfile:
    subscriber_id:      str
    autonomy_level:     str   # "suggestive" | "assistive" | "autonomous"
    goal_horizon:       str   # "short" | "mid" | "long"
    interaction_style:  str   # "ambient" | "soft-interrupt" | "hard-interrupt" | "conversational"
    output_format:      str   # "json" | "markdown" | "plain-text" | "voice" | "structured-alert"
    vocabulary_level:   str   # "technical" | "domain-expert" | "layman"
    verbosity:          str   # "concise" | "standard" | "detailed"
    citation_style:     str   # "inline" | "footnote" | "none"
    max_context_tokens: int
    domain_schema:      dict  # optional JSON Schema — constrains output structure
```

**Adding a new consumer = registering one new profile. Zero core changes.**

The same retrieved chunks are reformatted differently for each subscriber. This is Claim 3 in action: one retrieval event, heterogeneous outputs.

---

### 5.1 Consumer A — IDE Plugin

**Use case:** Developer in an IDE receives relevant debugging references, API docs, or code patterns passively in a sidebar.

| Profile field | Value |
|---|---|
| `autonomy_level` | `assistive` |
| `interaction_style` | `ambient` |
| `output_format` | `markdown` |
| `vocabulary_level` | `technical` |
| `verbosity` | `concise` |
| `citation_style` | `inline` |
| `max_context_tokens` | `512` |

**Example output:**
```markdown
### TypeError: 'NoneType' object is not subscriptable
Check that `config["key"]` is not returning `None` before subscripting.
Use `config.get("key", default)` or assert non-None before use.
See: [python_debugging_guide.md §3.2]
```

---

### 5.2 Consumer B — Factory Dashboard

**Use case:** Operations dashboard for a smart factory. Anomaly detected → structured alert fires immediately with severity, recommended action, and retrieved context.

| Profile field | Value |
|---|---|
| `autonomy_level` | `autonomous` |
| `interaction_style` | `hard-interrupt` |
| `output_format` | `structured-alert` |
| `vocabulary_level` | `domain-expert` |
| `verbosity` | `concise` |
| `domain_schema` | `{severity, action, context}` |

**Example output:**
```json
{
  "severity": "HIGH",
  "action": "Inspect pressure seal on CNC Lathe 03. Deviation at +3.2σ.",
  "context": "Pressure sensor 01 exceeded 3σ threshold. Last maintenance: 14 days ago."
}
```

---

### 5.3 Consumer C — Research Writing Assistant

**Use case:** Academic writing assistant. When the researcher pauses writing (velocity drop), APEX pushes relevant literature, methodology notes, or related prior work into a conversational panel.

| Profile field | Value |
|---|---|
| `autonomy_level` | `suggestive` |
| `interaction_style` | `conversational` |
| `output_format` | `markdown` |
| `vocabulary_level` | `domain-expert` |
| `verbosity` | `detailed` |
| `citation_style` | `footnote` |
| `max_context_tokens` | `1024` |

**Example output:**
```markdown
While drafting this section on proactive retrieval, the closest prior system to compare
against is ContextAgent (NeurIPS 2025)¹. The key distinction is that ContextAgent requires
cloud inference for its intent model, while APEX runs entirely on-device with Phi-3.5
Mini INT4. The Goldilocks timing formulation² provides the theoretical basis for the τ
threshold used in the scheduler.

---
¹ ContextAgent — reading/proactive_ai_survey.md §2.3
² Seo et al., CHI 2025 — reading/proactive_ai_survey.md §4.1
```

---

## 6. MCP Subscription Interface

APEX extends MCP from request-response to publish-subscribe. This is the protocol contribution of Claim 3.

| Endpoint | Method | Description |
|---|---|---|
| `/subscribe` | `POST` | Register a `ConsumerProfile` → returns `subscriber_id` |
| `/context/{id}` | `GET` | Pull-mode fallback — returns current buffer contents |
| `/stream/{id}` | `WebSocket` | Primary push channel — APEX fires when context is warm |
| `/subscribe/{id}` | `DELETE` | Unregister, clean up buffer partition |

Under normal operation, subscribers do not poll. They open the WebSocket and wait. APEX pushes when the pipeline fires. The GET fallback exists for applications that cannot maintain a persistent WebSocket.

**Protocol correctness contract:**
- A new `ConsumerProfile` registration must not affect any existing subscriber's state
- Unregistering must immediately stop pushes and release the buffer partition
- Re-registering with a different `ConsumerProfile` must adapt output to the new profile within one push cycle

---

## 7. Evaluation Design

All evaluation events are logged to DuckDB (`apex_eval.db`) via `apex/analytics/store.py`. Metrics are organized by the research claim they answer.

### Claim 1 Metrics — Proactive Delivery

| Metric | Formula | Target | Phase |
|---|---|---|---|
| **PRP** — Proactive Retrieval Precision | `claimed_prefetches / total_prefetches` | > 0.65 | Phase 0+ |
| **LtC** — Latency-to-Context | `t_available − t_claimed` (ms) | Negative mean | Phase 0+ |

**Claim rate feedback → τ calibration:**
Every session generates labeled data automatically. `TauCalibrator` reads DuckDB, computes per-bucket claim rates, updates τ. No manual annotation. The feedback loop is the mechanism behind the Claim 1 sub-claim.

**Condition comparison:** Fixed τ = 0.65 vs calibrated τ. Wilcoxon signed-rank test on PRP distributions.

### Claim 2 Metrics — On-Device

| Metric | Formula | Target | Phase |
|---|---|---|---|
| **BI** — Battery Impact | `(mW_apex − mW_baseline) / mW_baseline × 100%` | < 15% | Phase 2/3 only |
| **IIE latency** | `t_iie − t_signal` (ms) | < 20ms | Phase 2/3 |
| **Retrieval latency** | `t_retrieval − t_iie` (ms) | < 10ms | Phase 0+ |

**Hardware range:** Jetson Orin NX (Phase 2); iPhone A13/A15/A18 Pro, Mac M1 (Phase 3).
**Power measurement:** `tegrastats` on Jetson; IOKit and iOS Energy Log on Apple devices.

> **Phase 0 note:** Latency and memory are measured in Phase 0, but BI cannot be measured
> on a development machine. Phase 0 latency numbers are diagnostic only — not thesis claims.

### Claim 3 Metrics — Application-Agnostic

| Metric | Formula | Target | Phase |
|---|---|---|---|
| **DPS** — Delivery Precision per Subscriber | `(relevance_score + format_compliance) / 2` | > 0.75 | Phase 0+ |
| **Multi-subscriber overhead** | `max(t_push_N) − t_retrieval_complete` (ms) | Sublinear scaling | Phase 0+ |

**DPS annotation:** Two annotators. Relevance (0/0.5/1) and format compliance (0/0.5/1) scored independently. Cohen's κ > 0.6 required for both dimensions.

**Comparison:** Profiled (ConsumerProfile) vs unprofiled (generic markdown) — Mann-Whitney U test per subscriber type.

---

## 8. Implementation Status

| Component | File | Status |
|---|---|---|
| `SignalAdapter` base class | `apex/adapters/base.py` | ✅ Complete |
| `ProductivityAdapter` | `apex/adapters/productivity.py` | ✅ Complete |
| `FactoryAdapter` | `apex/adapters/factory.py` | ✅ Complete |
| `ResearchAdapter` | `apex/adapters/research.py` | ✅ Complete |
| `SignalMonitor` | `apex/monitor/signal_monitor.py` | ✅ Complete |
| `HeuristicGate` | `apex/inference/heuristic_gate.py` | ✅ Complete |
| `IntentEngine` | `apex/inference/intent_engine.py` | ✅ Complete |
| `TauCalibrator` | `apex/inference/tau_calibrator.py` | ✅ Complete |
| `SpeculativeScheduler` | `apex/scheduler/speculative.py` | ✅ Complete |
| `HNSWIndex` | `apex/retrieval/hnsw_index.py` | ✅ Complete |
| `BM25Index` | `apex/retrieval/bm25.py` | ✅ Complete |
| `RetrievalEngine` (RRF) | `apex/retrieval/rrf.py` | ✅ Complete |
| `ContextBuffer` | `apex/buffer/context_buffer.py` | ✅ Complete |
| `LLMAdapter` | `apex/adapter/llm_adapter.py` | ✅ Complete |
| `PipelineCoordinator` | `apex/pipeline/coordinator.py` | ✅ Complete |
| `AnalyticsStore` | `apex/analytics/store.py` | ✅ Complete |
| `MCP Server` | `apex/server.py` | ✅ Complete |
| Runtime daemon | `main.py` | ✅ Complete |
| LoRA adapters | `apex/inference/lora/` | ⚠️ Infrastructure complete, adapters in training |

**Test suite:** 312 tests passing. No mocks — all tests run against real Ollama and real pipeline components.

**Current phase:** Phase 0 — local macOS integration. Full stack runs with `just dev`. PRP and LtC are being collected from live sessions.

---

## 9. What This Document Does Not Cover

| Topic | Where to find it |
|---|---|
| Architecture Decision Records | `docs/adr/ADR-001-no-mock-testing.md`, `ADR-002-phase0-local-integration.md` |
| User-facing setup and usage | `docs/user_guide.md` |

> Research-angle deep-dives, academic positioning, and thesis-scope notes are kept
> in a separate private repository and can be shared with reviewers on request.
