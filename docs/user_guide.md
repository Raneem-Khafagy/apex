# APEX User Guide — Running, Testing, and Interpreting Proactive Responses

## What APEX Does (30-second version)

APEX watches what you are doing on your device — which files are open, how fast they are changing, what application is in focus — and silently pre-fetches relevant context from your local knowledge base **before you ask for it**. When it decides the context is ready and relevant enough (confidence ≥ τ threshold), it pushes a card to your UI. You never type a query. The system infers your need from behavioral signals alone.

Think of it like the AI in *Her* — it understands context from your behaviour, not from what you say.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python ≥ 3.11 | via pyenv or system |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js ≥ 18 | via nvm or system |
| Ollama | https://ollama.ai — install and keep running |
| just | `brew install just` |

Pull the two required models (one-time):
```bash
ollama pull phi3.5        # LLM Adapter — reformats retrieved chunks per subscriber
ollama pull all-minilm    # Embedder — converts signals into vectors for HNSW search
```

Install project dependencies:
```bash
uv sync                    # Python deps
just ui-install            # Node deps (apex-ui/)
```

---

## Startup Sequence

Every session must follow this exact order. Skipping a step causes silent failures.

### Terminal 1 — Ollama
```bash
ollama serve
```
Leave this running. All inference goes through it.

### Terminal 2 — Start the APEX daemon
```bash
just reset     # wipes stale DB and subscriber file — always run before a fresh session
APEX_VAULT_PATH=experiment_corpus \
APEX_INDEX_PATH=experiment_index/experiment \
just dev
```
The daemon starts FastAPI on port 8765, launches the Behavioral Signal Monitor, loads LoRA domain adapters (if available), and starts the per-domain τ calibrator.

If no index exists yet, run this first (one-time):
```bash
APEX_VAULT_PATH=experiment_corpus \
APEX_INDEX_PATH=experiment_index/experiment \
just reindex
```

### Terminal 3 — Open the web UI (development mode, hot-reload)
```bash
just ui-dev
```
Open `http://localhost:5173/app` in your browser.

Or use the **production build** (no separate dev server needed):
```bash
just ui-build
# then visit http://localhost:8765/app
```

---

## First-Time: Create an Account

1. Go to `http://localhost:5173/app` (or `http://localhost:8765/app` for prod build).
2. Click **Register** and enter a username and password (≥ 6 chars).
3. Type your **domain** — this is your identity in the system. It can be anything:
   - `writing`, `research`, `factory` — built-in themes with matching visual style
   - `medical`, `legal`, `engineering`, `logistics`, `teaching` — or any other string
   - Quick-pick chips appear below the input as suggestions
4. On the onboarding screen, configure how APEX interacts with you (or accept the defaults).
5. You are redirected to the stream. APEX auto-registers your subscriber and begins watching.

Multiple users can register on the same machine — each gets an isolated buffer partition and card history. They never see each other's data.

---

## Choosing Your Knowledge Base

The knowledge base is the folder of documents APEX retrieves from. You set it in **Settings → Knowledge Base**.

### Option A: Use your own folder
1. Open **Settings** (gear icon in the sidebar)
2. Under **Knowledge Base**, type the absolute path to a folder on your machine:
   ```
   /Users/yourname/Documents/MyProject
   /Users/yourname/Desktop/ResearchNotes
   ```
3. Click **Set path** — APEX validates the folder exists
4. Click **↻ Re-index now** — APEX scans all `.md`, `.txt`, `.pdf` files in that folder and builds the vector + BM25 index
5. Watch the chunk count update as indexing completes

The folder is monitored live. When files change, APEX detects the change and adjusts its signals accordingly.

### Option B: Use the experiment corpus (demo)
The repo includes a ready-made corpus for testing:
```bash
APEX_VAULT_PATH=experiment_corpus \
APEX_INDEX_PATH=experiment_index/experiment \
just dev
```
This uses the built-in writing/reading/debugging documents — enough to see proactive pushes immediately.

### Option C: Start fresh
Click **✕ Start fresh** in Settings → Knowledge Base. APEX creates an empty temp directory. You can add your own documents and then click **Re-index now**.

---

## Understanding the UI

### Sidebar (always dark)
- Your avatar initials + username + domain badge
- Last 8 context cards (history)
- Navigation: Stream · Settings
- Logout

### Card Feed (domain-themed main pane)
- Cards appear newest-first as APEX pushes context
- The **Layer 2 indicator** at the top pulses when signal confidence ≥ τ (pipeline is about to fire or just fired)
- **Pull context** button manually claims context from the buffer (fallback pull mode)
- Hover a card to reveal **Dismiss** (hides it locally)

### Settings
- **Knowledge Base** — choose the folder APEX retrieves from; re-index after changes
- **Domain** — change your domain at any time (takes effect on next re-subscribe)
- **Consumer Profile** — configure autonomy level, verbosity, interaction style, etc.
- **Save & re-subscribe** — applies changes and creates a fresh subscriber registration

### Researcher Metrics Strip
Press **Cmd+Shift+R** to reveal a 28px bottom strip showing live:
- **PRP** — Proactive Retrieval Precision (claimed / total prefetches). Target > 0.65
- **LtC** — Latency-to-Context in ms. **Negative = APEX was ahead of you (proactive ✓)**
- **DPS** — Delivery Precision per Subscriber. Measured via human annotation post-session
- **τ** — current threshold; calibrates per-domain over time as APEX learns your patterns
- **Buffer** — chunks currently queued for your subscriber
- **Domains** — LoRA adapters available (productivity, factory, research)

---

## LoRA Domain Adapters (Advanced Feature)

APEX supports domain-specific LoRA (Low-Rank Adaptation) adapters that fine-tune the Intent Inference Engine for different domains without retraining the base model. Each domain can have specialized behavior while sharing the universal base weights.

### Available Domains with LoRA Support

- **productivity**: Personal productivity (coding, writing, debugging)
- **factory**: Smart factory / industrial IoT
- **research**: Research and academic analysis

### Creating Mock Adapters (Development)

```bash
# Create all domain adapters
python -m apex.inference.lora.create_mock_adapters

# Create specific domain
python -m apex.inference.lora.create_mock_adapters --domain productivity

# List available domains
python -m apex.inference.lora.create_mock_adapters --list-domains

# Check what adapters are loaded
uv run python -c "from apex.inference.intent_engine import IntentEngine; print(IntentEngine().list_available_domains())"
```

### Domain → Label Mapping

The Intent Inference Engine automatically maps task context labels to domains:

- **productivity**: `debugging_python`, `writing_document`, `coding_javascript`, `api_testing`
- **factory**: `factory_anomaly`, `sensor_monitoring`, `maintenance_alert`, `production_line`
- **research**: `research_paper`, `reading_reference`, `academic_analysis`, `literature_review`

Labels are extracted from behavioral signals and mapped heuristically. Domain-specific τ thresholds are calibrated independently per domain.

### File Structure

LoRA adapter files are stored in `apex/inference/lora/`:
```
apex/inference/lora/
├── lora_productivity.bin      # PyTorch format
├── lora_factory.safetensors   # HuggingFace format
├── lora_research.bin
└── create_mock_adapters.py    # Development utility
```

### Integration with Intent Engine

When a behavioral signal is processed:
1. **Heuristic Gate**: Fast pattern matching (no LoRA applied)
2. **LLM Path**: Label generated → domain extracted → LoRA adapter applied to intent vector
3. **Retrieval**: Domain-adapted vector used for semantic search
4. **Scheduler**: Domain-specific τ threshold applied

---

## Domain-by-Domain: What to Do and What to Expect

Your domain shapes the visual theme, default interaction style, LLM adapter prompt, and LoRA adaptation behavior — but it does not restrict the system. Any string works.

---

### Built-in Domain: Writing

**Visual theme:** Warm parchment background, serif font — calm reading environment

**Corpus:** `experiment_corpus/writing/` — thesis draft, architecture notes, related work, evaluation methodology

**What APEX watches:** file modification rate in your vault directory, which `.md` files are changing, typing velocity inferred from change frequency

**Trigger scenario:**
1. Open and start editing a markdown file in your vault (e.g. `thesis_draft.md`)
2. Make several edits over 1–2 minutes at a moderate pace
3. APEX detects `activity_type = "writing"` with rising velocity

**Expected proactive push:**
- Excerpts from related work notes, architecture decisions, or prior thesis sections — whatever is semantically nearest to what you are currently editing
- Format: markdown, concise, with inline citations (e.g. `[ProactiveBench, 2025]`)
- If you are writing about the retrieval pipeline, APEX surfaces `hnsw_retrieval_algorithm_notes.md` without you searching for it

**How to test:**
```bash
just eval-quick    # 2-minute writing scenario — watch feed for cards
```

---

### Built-in Domain: Factory

**Visual theme:** Dark terminal aesthetic, monospace font, amber accent

**Corpus:** `experiment_corpus/` — field manuals on logistics, vehicle repair, artillery gunnery, road maintenance

**What APEX watches:** factory sensor simulator — temperature, pressure, vibration, anomaly delta

**Trigger scenarios:**

```bash
just sensor normal       # baseline — no push (confidence < τ)
just sensor anomaly      # urgency_flag = True — IMMEDIATE push, τ bypassed
just sensor maintenance  # scheduled window — moderate urgency, push within 5–15s
```

Expected anomaly card:
```json
{
  "severity": "HIGH",
  "action": "Inspect hydraulic system — pressure deviation 2.3σ above baseline",
  "context": "Relevant procedure: TM 9-2320 §4.3 — Emergency pressure relief"
}
```

Run `just sensor anomaly` → watch a hard-interrupt card appear almost instantly. Switch to `just sensor normal` → no card. This contrast is the core demo.

---

### Built-in Domain: Research

**Visual theme:** Academic cream background, serif font — paper reading environment

**Corpus:** `experiment_corpus/reading/` — HNSW notes, edge computing hardware, proactive AI literature

**What APEX watches:** which files are open, reading velocity, transitions between reading and drafting

**Trigger scenario:**
1. Open files in your reading folder and browse between them
2. Pause on one for 30+ seconds (reading absorption)
3. Switch to editing a draft

**Expected push:** Background literature relevant to the draft section you are about to write — surfaces while you are still reading, so it is ready when you start writing.

**How to test:**
```bash
just eval          # full 10-minute evaluation — writing + reading + debugging
just stop && just metrics
```

---

### Custom Domains

Any domain string works. Examples:

| Domain | What to put in your vault | Expected behaviour |
|--------|--------------------------|-------------------|
| `medical` | Clinical guidelines, drug references, case notes | Surfaces protocol references when monitoring alerts change |
| `legal` | Case law, statutes, contract templates | Surfaces relevant clauses when you start drafting |
| `engineering` | Design specs, datasheets, failure logs | Surfaces relevant specs when anomaly signals fire |
| `teaching` | Lesson plans, syllabi, student notes | Surfaces related material when editing a new plan |

For any custom domain: put your documents in a folder, set the path in Settings, re-index, and APEX will watch that folder and retrieve from it. The LLM adapter formats output with the `domain-expert` profile by default — adjust in Settings.

---

## Running the Full Proactive Evaluation

This measures whether APEX was genuinely proactive (LtC < 0) versus reactive. The evaluation framework now includes comprehensive claim analysis, per-domain τ calibration, and multi-subscriber overhead measurement.

### Basic Evaluation (Single Configuration)

```bash
# Step 1 — fresh start
just reset

# Step 2 — start daemon with experiment corpus
APEX_VAULT_PATH=experiment_corpus \
APEX_INDEX_PATH=experiment_index/experiment \
just dev

# Step 3 — register Phase 0 consumers (required — do not skip)
just register

# Step 4 — run evaluation (10 min default)
just eval

# Step 5 — stop daemon and read metrics
just stop && just metrics
```

### Advanced Evaluation (Claim 1: Fixed vs Adaptive τ Comparison)

```bash
# Run baseline with FIXED τ = 0.65 (no calibration)
just eval-fixed 600 35    # 10 min, 35s pull interval

# Run comparison with ADAPTIVE τ (calibrated per-domain)
just eval-adaptive 600 35

# Compare results side-by-side
just compare-tau
```

### Multi-Subscriber Overhead Analysis (Claim 3 Sub-experiment)

```bash
# Evaluation automatically measures overhead when N > 1 subscribers
just register   # Creates 3 subscribers: ide_plugin, factory_dashboard, research_assistant
just eval       # Records multi_sub_overhead_ms for scaling analysis

# View overhead scaling results
just stop && just metrics --verbose
```

### DPS Quality Annotation (Claim 3 Human Assessment)

```bash
# After any evaluation session, annotate delivery quality
just stop && just annotate-dps

# Follow interactive prompts for relevance and format compliance scores
# Scores: 0 (poor) / 0.5 (adequate) / 1.0 (excellent)

# View final DPS results
just metrics
```

**Interpreting the output:**

| What you see | What it means |
|---|---|
| `PRP > 0` | Pipeline is retrieving and at least some context was claimed |
| `PRP = 0, prefetches > 0` | Retrieval fires but nothing is claimed — check WS connection, buffer TTL |
| `PRP = 0, prefetches = 0` | Pipeline is not retrieving at all — check index, τ threshold, signal flow |
| `LtC mean = -15000 ms` | APEX was 15 seconds ahead of the user's need ✓ (proactive) |
| `LtC mean = +5000 ms` | APEX was late — system is reactive, not proactive |
| `WS pushes = 0` | Frontend never received a push — check WebSocket connection in browser DevTools |
| `DPS = (no data)` | No delivery events annotated — run `just annotate-dps` after evaluation |
| `Multi-sub overhead: 25ms` | Processing 3 subscribers adds 25ms vs single subscriber |
| `τ (productivity) = 0.55` | Domain-specific threshold calibrated down from 0.65 baseline |

**Target (Phase 0 development):** PRP > 0, LtC negative, DPS > 0.75 (post-annotation). Absolute values are diagnostic — sign and trend are what matter.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Login page shows "Loading…" forever | APEX daemon not running or Ollama not started | Start daemon (`just dev`) and Ollama (`ollama serve`) |
| Auth call fails in dev mode | `/auth` not proxied | Already fixed — update to latest; make sure `just ui-dev` is running |
| Cards never appear | Index not built, or wrong vault path | Run re-index from Settings, or `just reindex` |
| "0 chunks indexed" in Settings | Index path mismatch or ingest not run | Set vault path in Settings → click Re-index now |
| Domain registration fails with 422 | Old validator still running | Hard-restart the daemon (`just reset && just dev`) |
| Re-index takes very long | Large folder with many files | Normal — HNSW embedding is ~1s per document; start with a small folder |
| Daemon is slow to respond | Ollama / phi3.5 cold start | First request after start takes 5–30s while models load into memory |
| `DPS = (no data)` in metrics | No delivery events annotated | Run `just annotate-dps` after eval session to score delivery quality |
| Metrics show "No data in fixed DB" | eval-fixed not run yet | Run `just eval-fixed` before `just compare-tau` |
| LoRA adapters not loading | Missing adapter files | Run `python -m apex.inference.lora.create_mock_adapters` |
| τ not calibrating per domain | Fixed τ mode enabled | Remove `--fixed-tau` flag or unset `APEX_TAU_FIXED` env var |

---

## Quick-Start Cheat Sheet

```bash
# Every session (terminals 1 and 2)
ollama serve
just reset && APEX_VAULT_PATH=experiment_corpus APEX_INDEX_PATH=experiment_index/experiment just dev

# Terminal 3 — UI dev server
just ui-dev
# → http://localhost:5173/app

# Set your own knowledge base (UI)
# Settings → Knowledge Base → type path → Set path → Re-index now

# Test writing domain
just eval-quick

# Test factory domain
just sensor anomaly

# Basic evaluation
just register && just eval
just stop && just metrics

# Advanced evaluation (all claims)
# Claim 1: Fixed vs Adaptive τ comparison
just eval-fixed && just eval-adaptive && just compare-tau

# Claim 2: Latency profiling (automatically included in all evals)
just stop && just metrics  # Shows IIE, retrieval, push timing breakdown

# Claim 3: DPS annotation + multi-subscriber overhead
just register && just eval  # Multi-subscriber overhead measured automatically
just stop && just annotate-dps  # Interactive quality scoring
just metrics  # Final results with DPS

# LoRA domain adapters (development)
python -m apex.inference.lora.create_mock_adapters
uv run python -c "from apex.inference.intent_engine import IntentEngine; print(IntentEngine().list_available_domains())"

# Live metrics in browser
# Cmd+Shift+R
```
