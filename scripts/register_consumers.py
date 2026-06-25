"""
Phase 0 — Register three ConsumerProfiles with the APEX MCP server.

Registers:
  A: IDE plugin       — technical markdown, ambient, concise
  B: Factory dashboard — structured-alert JSON, hard-interrupt, concise
  C: Research assistant — domain-expert markdown, conversational, detailed

Usage (server must be running on port 8765):
    uv run python scripts/register_consumers.py

Saves subscriber IDs to .phase0_subscribers.json so that
print_metrics.py and simulate_signals.py can reuse them across terminal
sessions without re-registering.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

SERVER = "http://localhost:8765"
STATE_FILE = Path(".phase0_subscribers.json")

# ── Consumer profiles ─────────────────────────────────────────────────────────

PROFILES: list[dict] = [
    # ── A: IDE plugin ──────────────────────────────────────────────────────
    # Ambient — surfaces context in a sidebar without interrupting the user.
    # Technical markdown with inline citations for developer readability.
    {
        "_label": "ide_plugin",
        "autonomy_level":    "assistive",
        "goal_horizon":      "short",
        "interaction_style": "ambient",
        "output_format":     "markdown",
        "vocabulary_level":  "technical",
        "verbosity":         "concise",
        "citation_style":    "inline",
        "max_context_tokens": 512,
        "domain_schema":     None,
    },

    # ── B: Factory dashboard ───────────────────────────────────────────────
    # Hard-interrupt — fires an alert immediately when anomaly context arrives.
    # Structured JSON matching the dashboard's alert schema.
    {
        "_label": "factory_dashboard",
        "autonomy_level":    "autonomous",
        "goal_horizon":      "short",
        "interaction_style": "hard-interrupt",
        "output_format":     "structured-alert",
        "vocabulary_level":  "domain-expert",
        "verbosity":         "concise",
        "citation_style":    "none",
        "max_context_tokens": 256,
        "domain_schema": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                },
                "action":  {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["severity", "action"],
        },
    },

    # ── C: Research writing assistant ─────────────────────────────────────
    # Conversational — appears as a chat panel when the user pauses writing.
    # Detailed domain-expert markdown with footnote citations.
    {
        "_label": "research_assistant",
        "autonomy_level":    "suggestive",
        "goal_horizon":      "long",
        "interaction_style": "conversational",
        "output_format":     "markdown",
        "vocabulary_level":  "domain-expert",
        "verbosity":         "detailed",
        "citation_style":    "footnote",
        "max_context_tokens": 1024,
        "domain_schema":     None,
    },
]


def register_all() -> dict[str, str]:
    """POST each profile to /subscribe and return label → subscriber_id map."""
    ids: dict[str, str] = {}

    for profile in PROFILES:
        label = profile.pop("_label")
        try:
            resp = httpx.post(f"{SERVER}/subscribe", json=profile, timeout=5.0)
            resp.raise_for_status()
            subscriber_id = resp.json()["subscriber_id"]
            ids[label] = subscriber_id
            print(f"  ✓  {label:<25} → {subscriber_id}")
        except httpx.ConnectError:
            print(
                f"\n[ERROR] Cannot reach {SERVER}.\n"
                "Make sure the server is running:\n"
                "  just serve\n",
                file=sys.stderr,
            )
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            print(f"  ✗  {label}: HTTP {exc.response.status_code}", file=sys.stderr)

    return ids


def main() -> None:
    print(f"\nRegistering consumers with APEX MCP server at {SERVER} …\n")
    ids = register_all()

    STATE_FILE.write_text(json.dumps(ids, indent=2))
    print(f"\nSubscriber IDs saved to {STATE_FILE}")
    print("\nTo open push streams, run:")
    for label, sid in ids.items():
        print(f"  uv run python scripts/watch_stream.py {sid}  # {label}")


if __name__ == "__main__":
    main()
