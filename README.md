# APEX

**A**pplication-agnostic **P**roactive **E**dge-native conte**X**t pushing

APEX is an on-device daemon that watches what a user is doing, infers what
information they will need next, and pushes it to any subscribing application —
**before the user asks**. No cloud, no explicit query. Everything runs locally.

> **Research question:** Can proactive context delivery run entirely on-device,
> triggered by OS-level behavioral patterns, and serve any application domain
> without bespoke integration?

## The core inversion

```
Traditional RAG (Pull)            APEX (Push)
  User types a query                OS detects a behavioral signal
    → app sends a request             → APEX infers intent
      → RAG retrieves                   → APEX retrieves speculatively
        → app shows result              → context is already buffered when needed
```

## Architecture

The pipeline runs as a sequence of stages (C1–C4):

| Stage | Module | Role |
|---|---|---|
| C1 — Behavioral signals | `apex/monitor` | Capture OS-level activity signals |
| C2 — Intent inference   | `apex/inference` | Map signals to predicted information need |
| C3 — Speculative retrieval | `apex/retrieval`, `apex/scheduler` | Pre-fetch candidate context (HNSW + BM25) |
| C4 — Context delivery   | `apex/buffer`, `apex/pipeline` | Buffer and push to subscribers over a pub/sub protocol |

Supporting modules: `apex/ingest` (knowledge-base indexing), `apex/adapters`
(domain consumers), `apex/analytics` (metrics store), `apex/auth` (JWT auth),
`apex/server.py` (FastAPI app). A React UI lives in `apex-ui/`.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) running locally (tests run against a real model — no mocks)

## Quick start

```bash
# Install dependencies
uv sync

# Build the knowledge-base index from your vault
#   (defaults to ~/Documents/ApexVault; override with APEX_VAULT_PATH)
uv run python main.py ingest

# Run the daemon / API
uv run python main.py

# Run the test suite (requires a running Ollama)
uv run pytest
```

See `justfile` for the full set of development commands, and
[`docs/apex_technical_overview.md`](docs/apex_technical_overview.md) for the
detailed design.

## Documentation

- `docs/apex_technical_overview.md` — full architecture and design
- `docs/user_guide.md` — setup and usage
- `docs/adr/` — architecture decision records

## Status

Research prototype (Phase 0, local integration). This repository contains the
core system. Thesis-specific material — academic positioning, research-angle
deep-dives, and scope notes — is kept separately and can be shared with
reviewers on request.

## License

[MIT](LICENSE) © 2026 Raneem Khafagy
