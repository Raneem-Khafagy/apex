# APEX development commands
# Usage: just <recipe>

default:
    just --list

# Start the APEX MCP server (FastAPI on port 8765)
serve:
    uv run uvicorn apex.server:app --reload --port 8765

# Start the Ollama LLM runtime (must be installed separately)
llm:
    ollama serve

# Live rich terminal display of signal monitor output
monitor:
    uv run python -m apex.monitor.live

# Run the full pytest suite
test:
    uv run python -m pytest tests/ -v

# Re-index the knowledge base from ApexVault (set APEX_VAULT_PATH to override)
ingest:
    uv run python -m apex.ingest.ingestor

# Start full dev stack: Ollama (separate) + APEX daemon (server + BSM + calibrator)
# In one terminal:  just llm
# In another:       just dev
# IMPORTANT: run `just register` in a third terminal after this starts —
#            every daemon restart clears subscriber profiles; eval will show 0 hits without it.
dev:
    uv run python main.py

# Start APEX daemon with FIXED τ = 0.65 (no calibration) — for baseline comparison
# This is the control condition for RQ1.2: fixed τ vs adaptive τ comparison.
dev-fixed:
    uv run python main.py --fixed-tau

# ── Phase 0: local integration ───────────────────────────────────────────────

# Register the three Phase 0 consumer profiles with the running server
register:
    uv run python scripts/register_consumers.py

# Open a push stream for a named consumer (ide_plugin | factory_dashboard | research_assistant)
watch label:
    uv run python scripts/watch_stream.py --label {{label}}

# Print PRP + LtC metrics from the DuckDB evaluation store
# NOTE: the APEX daemon must be stopped first (Ctrl+C in the `just dev` terminal).
#       DuckDB's write lock prevents concurrent reads while the daemon is running.
metrics *args:
    uv run python scripts/print_metrics.py {{args}}

# Annotate delivery events for DPS measurement (Claim 3)
# Interactive tool to score relevance and format compliance of push deliveries
annotate-dps *args:
    uv run python scripts/annotate_dps.py {{args}}

# Stop the APEX daemon gracefully and wait for it to fully exit.
# Sends SIGTERM, then polls until the process is gone (up to 10s).
# This ensures DuckDB releases its write lock before `just metrics` runs.
stop:
    #!/usr/bin/env bash
    if pkill -SIGTERM -f "python main.py" 2>/dev/null; then
        echo "Daemon stopping…"
        for i in $(seq 1 20); do
            sleep 0.5
            pgrep -f "python main.py" > /dev/null || { echo "Daemon stopped."; exit 0; }
        done
        echo "Daemon did not exit after 10s — sending SIGKILL"
        pkill -SIGKILL -f "python main.py" || true
    else
        echo "No daemon process found."
    fi

# Full reset: stop daemon, wipe DuckDB + stale subscriber file.
# Run this whenever you restart the daemon to avoid stale session/subscriber mismatches.
# After reset: just dev → just register → just eval
reset:
    just stop || true
    rm -f apex_eval.db .phase0_subscribers.json
    @echo "DB + subscriber file cleared."
    @echo "Next: APEX_VAULT_PATH=experiment_corpus APEX_INDEX_PATH=experiment_index/experiment just dev"
    @echo "Then: just register  (MANDATORY)"
    @echo "Then: just eval"

# ── Phase 0 evaluation (full proactive harness) ───────────────────────────────

# Full evaluation run: vault agent with pull-mode supervision (proves proactivity).
# ── Required sequence (MUST follow this order every time) ──────────────────────
#   just reset    ← wipes stale DB + subscriber file; stops old daemon
#   APEX_VAULT_PATH=experiment_corpus APEX_INDEX_PATH=experiment_index/experiment just dev
#   just register ← MANDATORY after every dev start; eval aborts if skipped
#   just eval     ← will abort with a clear message if register was skipped
#   just stop && just annotate-dps ← OPTIONAL: annotate for DPS (Claim 3)
#   just metrics  ← view PRP, LtC, and DPS results
# ───────────────────────────────────────────────────────────────────────────────
# Usage: just eval              (10 min, 35s pull interval)
#        just eval 900 30       (15 min, 30s pull interval)
eval duration="600" pull_interval="35":
    uv run python scripts/vault_agent.py --duration {{duration}} --pull-interval {{pull_interval}} --auto-start

# Quick smoke-test: 2-minute writing scenario only (checks pipeline is wired)
# Uses --no-ws to avoid push/pull claim conflicts during the short window.
eval-quick:
    uv run python scripts/vault_agent.py --scenario writing --duration 120 --pull-interval 20 --auto-start --no-ws

# ── Claim 1 comparison: fixed τ vs calibrated τ ───────────────────────────────

# Run baseline evaluation with FIXED τ = 0.65 (no calibration)
# For RQ1.2: compare this PRP against adaptive τ
eval-fixed duration="600" pull_interval="35":
    @echo "Starting FIXED τ baseline evaluation..."
    @echo "1. Stop any running daemon..."
    just stop || true
    @echo "2. Clear old data..."
    rm -f apex_eval_fixed.db .phase0_subscribers.json
    @echo "3. Start daemon with fixed τ..."
    APEX_DB_PATH=apex_eval_fixed.db APEX_VAULT_PATH=experiment_corpus APEX_INDEX_PATH=experiment_index/experiment uv run python main.py --fixed-tau &
    @echo "4. Wait for startup..."
    sleep 3
    @echo "5. Register consumers..."
    uv run python scripts/register_consumers.py
    @echo "6. Run evaluation..."
    uv run python scripts/vault_agent.py --duration {{duration}} --pull-interval {{pull_interval}} --auto-start
    @echo "7. Stop daemon..."
    just stop
    @echo "Fixed τ baseline complete. Results in apex_eval_fixed.db"

# Run adaptive evaluation with CALIBRATED τ (normal mode)
# For RQ1.2: compare this PRP against fixed τ
eval-adaptive duration="600" pull_interval="35":
    @echo "Starting ADAPTIVE τ calibrated evaluation..."
    @echo "1. Stop any running daemon..."
    just stop || true
    @echo "2. Clear old data..."
    rm -f apex_eval_adaptive.db .phase0_subscribers.json
    @echo "3. Start daemon with adaptive τ..."
    APEX_DB_PATH=apex_eval_adaptive.db APEX_VAULT_PATH=experiment_corpus APEX_INDEX_PATH=experiment_index/experiment uv run python main.py &
    @echo "4. Wait for startup..."
    sleep 3
    @echo "5. Register consumers..."
    uv run python scripts/register_consumers.py
    @echo "6. Run evaluation..."
    uv run python scripts/vault_agent.py --duration {{duration}} --pull-interval {{pull_interval}} --auto-start
    @echo "7. Stop daemon..."
    just stop
    @echo "Adaptive τ evaluation complete. Results in apex_eval_adaptive.db"

# Compare fixed vs adaptive τ results
# Prints PRP for both conditions side by side
compare-tau:
    @echo "Claim 1 sub-claim comparison: Fixed τ vs Adaptive τ"
    @echo "=============================================="
    @echo "Fixed τ = 0.65 (baseline):"
    @if [ -f apex_eval_fixed.db ]; then \
        uv run python scripts/print_metrics.py --db apex_eval_fixed.db --session-id-pattern '%' || echo "No data in fixed DB"; \
    else \
        echo "  ❌ apex_eval_fixed.db not found. Run: just eval-fixed"; \
    fi
    @echo ""
    @echo "Adaptive τ (calibrated):"
    @if [ -f apex_eval_adaptive.db ]; then \
        uv run python scripts/print_metrics.py --db apex_eval_adaptive.db --session-id-pattern '%' || echo "No data in adaptive DB"; \
    else \
        echo "  ❌ apex_eval_adaptive.db not found. Run: just eval-adaptive"; \
    fi

# Rebuild the experiment index after adding new corpus documents
reindex:
    APEX_VAULT_PATH=experiment_corpus APEX_INDEX_PATH=experiment_index/experiment \
    uv run python -m apex.ingest.ingestor

# Simulate factory sensor state (mode: normal | anomaly | maintenance)
sensor mode="normal":
    uv run python scripts/simulate_factory_sensor.py --mode {{mode}}

# Run unit tests only (no Ollama required)
test-unit:
    uv run python -m pytest tests/ -m "not integration" -v

# Run integration tests (requires: ollama serve + ollama pull all-minilm + ollama pull phi3.5)
test-integration:
    uv run python -m pytest tests/ -m integration -v

# ── React SPA (apex-ui/) ──────────────────────────────────────────────────────

# Install SPA dependencies (run once after checkout)
ui-install:
    cd apex-ui && npm install

# Start Vite dev server (hot-reload) — proxies API to localhost:8765
# Requires `just dev` running in another terminal
ui-dev:
    cd apex-ui && npm run dev

# Build SPA and write output to apex/static/app/ (served at /app)
ui-build:
    cd apex-ui && npm run build

# Run Vitest pure-logic unit tests (no server required)
ui-test:
    cd apex-ui && npm test

# Remove build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.db" -delete 2>/dev/null || true
    find . -name "*.index" -delete 2>/dev/null || true
    find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
