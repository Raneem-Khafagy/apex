"""
Phase 0 — Open a WebSocket push stream for a registered subscriber.

Usage:
    uv run python scripts/watch_stream.py <subscriber_id>

    # Or use the saved IDs from register_consumers.py:
    uv run python scripts/watch_stream.py --label ide_plugin
    uv run python scripts/watch_stream.py --label factory_dashboard
    uv run python scripts/watch_stream.py --label research_assistant

Prints each pushed context block to stdout as it arrives.
Send "pull" to request the current buffer contents immediately.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

SERVER_WS = "ws://localhost:8765"
STATE_FILE = Path(".phase0_subscribers.json")


async def stream(subscriber_id: str) -> None:
    uri = f"{SERVER_WS}/stream/{subscriber_id}"
    print(f"\nConnecting to {uri} …")

    try:
        async with websockets.connect(uri) as ws:
            print("Connected. Waiting for context pushes. (Type 'pull' + Enter to request immediately)\n")

            async def read_stdin() -> None:
                loop = asyncio.get_event_loop()
                while True:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if line.strip().lower() in ("pull", "ping"):
                        await ws.send("pull")

            stdin_task = asyncio.create_task(read_stdin())

            try:
                async for message in ws:
                    if message:
                        print(f"\n{'─' * 60}")
                        print(message)
                        print(f"{'─' * 60}\n")
            finally:
                stdin_task.cancel()

    except websockets.exceptions.ConnectionClosedError as e:
        if e.code == 4004:
            print(
                f"\n[ERROR] Subscriber '{subscriber_id}' not found on the server.\n"
                "Register consumers first with:\n"
                "  uv run python scripts/register_consumers.py\n",
                file=sys.stderr,
            )
        else:
            print(f"\n[ERROR] Connection closed: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError:
        print(
            "\n[ERROR] Cannot connect to the APEX server.\n"
            "Start it with:  just serve\n",
            file=sys.stderr,
        )
        sys.exit(1)


def load_id(label: str) -> str:
    if not STATE_FILE.exists():
        print(
            f"\n[ERROR] {STATE_FILE} not found.\n"
            "Register consumers first with:\n"
            "  uv run python scripts/register_consumers.py\n",
            file=sys.stderr,
        )
        sys.exit(1)
    ids: dict[str, str] = json.loads(STATE_FILE.read_text())
    if label not in ids:
        print(
            f"\n[ERROR] Label '{label}' not in {STATE_FILE}.\n"
            f"Available: {list(ids.keys())}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return ids[label]


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch an APEX push stream")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("subscriber_id", nargs="?", help="Direct subscriber UUID")
    group.add_argument(
        "--label", "-l",
        choices=["ide_plugin", "factory_dashboard", "research_assistant"],
        help="Symbolic label (reads from .phase0_subscribers.json)",
    )
    args = parser.parse_args()

    sid = args.subscriber_id or load_id(args.label)
    asyncio.run(stream(sid))


if __name__ == "__main__":
    main()
