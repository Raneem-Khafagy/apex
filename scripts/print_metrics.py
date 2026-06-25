"""
Phase 0 — Query DuckDB and print the four thesis evaluation metrics.

Reads from the DuckDB file at $APEX_DB_PATH (default: apex_eval.db).
Run this after a Phase 0 evaluation session to see preliminary PRP and LtC.

Usage:
    uv run python scripts/print_metrics.py
    uv run python scripts/print_metrics.py --session <session_id>
    uv run python scripts/print_metrics.py --all
    uv run python scripts/print_metrics.py --db apex_eval_fixed.db
    uv run python scripts/print_metrics.py --db apex_eval_fixed.db --session-id-pattern '%'

Options:
    --db PATH               Path to DuckDB file (overrides $APEX_DB_PATH)
    --session / -s ID       Session ID to query (default: most recent)
    --all / -a              Print metrics for all sessions
    --session-id-pattern P  SQL LIKE pattern to select sessions (e.g. '%' for all)
    --verbose / -v          Show latency breakdown table

Output (example):
    ┌─────────────────────────────────────────────┐
    │  APEX Phase 0 — Evaluation Metrics          │
    ├───────────┬────────────┬────────────────────┤
    │  Metric   │  Value     │  Target            │
    ├───────────┼────────────┼────────────────────┤
    │  PRP      │  0.71      │  > 0.65  ✓         │
    │  LtC mean │  -342 ms   │  negative mean  ✓  │
    │  DPS      │  (no data) │  > 0.75            │
    │  BI       │  (Phase 3) │  < 15%             │
    └───────────┴────────────┴────────────────────┘
"""
from __future__ import annotations

import argparse
import os
import sys

import duckdb

_DEFAULT_DB = os.environ.get("APEX_DB_PATH", "apex_eval.db")

# Thesis targets
PRP_TARGET  = 0.65
LTC_TARGET  = 0.0    # negative mean = good
DPS_TARGET  = 0.75


def _fmt(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "(no data)"
    return f"{value:.{precision}f}"


def _check(value: float | None, target: float, higher_is_better: bool = True) -> str:
    if value is None:
        return ""
    met = value > target if higher_is_better else value < target
    return "✓" if met else "✗"


def print_metrics(session_id: str | None = None, db_path: str | None = None, verbose: bool = False) -> None:
    db = db_path or _DEFAULT_DB
    try:
        con = duckdb.connect(db, read_only=True)
    except duckdb.IOException:
        db_exists = os.path.exists(db)
        if db_exists:
            print(
                f"\n[ERROR] Cannot open '{db}' — the APEX daemon is holding it open.\n"
                "\n"
                "DuckDB uses an exclusive write lock, so metrics cannot be read while\n"
                "the daemon is running.  Stop the daemon first, then re-run:\n"
                "\n"
                "    just stop        # graceful SIGTERM\n"
                "    just metrics     # now reads cleanly\n"
                "\n"
                "Or press Ctrl+C in the `just dev` terminal, wait for the process to exit,\n"
                "then run `just metrics`.\n",
                file=sys.stderr,
            )
        else:
            print(
                f"\n[ERROR] Database not found at '{db}'.\n"
                "Make sure the APEX daemon has run at least one session,\n"
                "and that APEX_DB_PATH points to the correct file.\n",
                file=sys.stderr,
            )
        sys.exit(1)

    # ── Session list ──────────────────────────────────────────────────────────
    sessions: list[str] = [
        row[0] for row in con.execute(
            "SELECT DISTINCT session_id FROM prefetch_events ORDER BY session_id"
        ).fetchall()
    ]

    if not sessions:
        print("\nNo prefetch events found. Run a Phase 0 session first.\n")
        return

    if session_id is None:
        # Default: most recent session (last alphabetically / chronologically)
        session_id = sessions[-1]

    if session_id not in sessions:
        print(f"\n[ERROR] Session '{session_id}' not found.")
        print(f"Available sessions: {sessions}\n", file=sys.stderr)
        sys.exit(1)

    # ── Metric queries ────────────────────────────────────────────────────────
    prp_row = con.execute(
        """
        SELECT COUNT(*) FILTER (WHERE claimed) * 1.0 / NULLIF(COUNT(*), 0)
        FROM prefetch_events WHERE session_id = ?
        """,
        [session_id],
    ).fetchone()
    prp = float(prp_row[0]) if prp_row and prp_row[0] is not None else None

    ltc_row = con.execute(
        """
        SELECT AVG(latency_ms),
               MIN(latency_ms),
               MAX(latency_ms),
               PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms),
               PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY latency_ms),
               PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms)
        FROM prefetch_events
        WHERE session_id = ? AND claimed = TRUE
        """,
        [session_id],
    ).fetchone()
    ltc       = float(ltc_row[0]) if ltc_row and ltc_row[0] is not None else None
    ltc_min   = float(ltc_row[1]) if ltc_row and ltc_row[1] is not None else None
    ltc_max   = float(ltc_row[2]) if ltc_row and ltc_row[2] is not None else None
    ltc_p50   = float(ltc_row[3]) if ltc_row and ltc_row[3] is not None else None
    ltc_p10   = float(ltc_row[4]) if ltc_row and ltc_row[4] is not None else None
    ltc_p90   = float(ltc_row[5]) if ltc_row and ltc_row[5] is not None else None

    # Breakdown: how many claims came from pull supervision vs push delivery
    claim_breakdown = con.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE claimed AND latency_ms < -100)  AS pull_claims,
            COUNT(*) FILTER (WHERE claimed AND latency_ms >= -100) AS push_claims
        FROM prefetch_events WHERE session_id = ?
        """,
        [session_id],
    ).fetchone()
    pull_claims = int(claim_breakdown[0]) if claim_breakdown else 0
    push_claims = int(claim_breakdown[1]) if claim_breakdown else 0

    dps_row = con.execute(
        """
        SELECT AVG((relevance_score + format_score) / 2.0)
        FROM delivery_events WHERE session_id = ?
        """,
        [session_id],
    ).fetchone()
    dps = float(dps_row[0]) if dps_row and dps_row[0] is not None else None

    total_row = con.execute(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE claimed) FROM prefetch_events WHERE session_id = ?",
        [session_id],
    ).fetchone()
    total_prefetches  = int(total_row[0]) if total_row else 0
    claimed_prefetches = int(total_row[1]) if total_row else 0

    # ── Display ───────────────────────────────────────────────────────────────
    width = 62
    print(f"\n{'─' * width}")
    print(f"  APEX Evaluation Metrics  —  Session: {session_id[:24]}…")
    print(f"  Phase 0 (development environment — preliminary only)")
    print(f"{'─' * width}")
    print(f"  Prefetch events : {total_prefetches} total, {claimed_prefetches} claimed")
    print(f"  Claim breakdown : {pull_claims} pull-supervision  /  {push_claims} push-delivery")
    print(f"{'─' * width}")
    print(f"  {'Metric':<12}  {'Value':<16}  {'Target':<12}  {'Met?'}")
    print(f"  {'──────':<12}  {'─────':<16}  {'──────':<12}  {'────'}")

    prp_str = _fmt(prp)
    prp_check = _check(prp, PRP_TARGET, higher_is_better=True)
    print(f"  {'PRP':<12}  {prp_str:<16}  {f'> {PRP_TARGET}':<12}  {prp_check}")

    ltc_ms = f"{ltc:.0f} ms" if ltc is not None else "(no data)"
    ltc_check = _check(ltc, LTC_TARGET, higher_is_better=False)
    print(f"  {'LtC mean':<12}  {ltc_ms:<16}  {'< 0 ms':<12}  {ltc_check}")

    dps_str = _fmt(dps)
    dps_check = _check(dps, DPS_TARGET, higher_is_better=True)
    print(f"  {'DPS':<12}  {dps_str:<16}  {f'> {DPS_TARGET}':<12}  {dps_check}")

    print(f"  {'BI':<12}  {'(Phase 3)':<16}  {'< 15%':<12}  —")
    print(f"{'─' * width}")

    # LtC distribution (only if we have claimed events)
    if ltc is not None:
        print(f"\n  LtC distribution (negative = APEX was proactive):")
        print(f"    p10  : {ltc_p10:.0f} ms" if ltc_p10 is not None else "    p10  : (no data)")
        print(f"    p50  : {ltc_p50:.0f} ms" if ltc_p50 is not None else "    p50  : (no data)")
        print(f"    mean : {ltc:.0f} ms")
        print(f"    p90  : {ltc_p90:.0f} ms" if ltc_p90 is not None else "    p90  : (no data)")
        print(f"    min  : {ltc_min:.0f} ms" if ltc_min is not None else "    min  : (no data)")
        print(f"    max  : {ltc_max:.0f} ms" if ltc_max is not None else "    max  : (no data)")
        if ltc < -5000:
            lead_s = abs(ltc) / 1000
            print(f"\n  → APEX was on average {lead_s:.1f}s ahead of user need  ✓")
        elif ltc < 0:
            print(f"\n  → APEX was proactive but lead time is < 5s; tune τ or pull interval")
        else:
            print(f"\n  → LtC ≥ 0: check claim path — are pulls hitting the buffer?")

    # ── Verbose: pipeline latency breakdown ──────────────────────────────────
    if verbose:
        breakdown = con.execute(
            """
            SELECT
                AVG(iie_ms)              AS avg_iie_ms,
                AVG(retrieval_ms)        AS avg_retrieval_ms,
                AVG(push_ms)             AS avg_push_ms,
                AVG(multi_sub_overhead_ms) AS avg_overhead_ms,
                COUNT(*) FILTER (WHERE iie_ms IS NOT NULL) AS iie_n
            FROM prefetch_events WHERE session_id = ?
            """,
            [session_id],
        ).fetchone()
        if breakdown and breakdown[4]:
            print(f"\n  Pipeline latency breakdown (avg, N={breakdown[4]}):")
            print(f"    IIE inference    : {breakdown[0]:.1f} ms" if breakdown[0] is not None else "    IIE inference    : (no data)")
            print(f"    Retrieval (HNSW) : {breakdown[1]:.1f} ms" if breakdown[1] is not None else "    Retrieval (HNSW) : (no data)")
            print(f"    Push (WS)        : {breakdown[2]:.1f} ms" if breakdown[2] is not None else "    Push (WS)        : (no data)")
            if breakdown[3] is not None:
                print(f"    Multi-sub overhead: {breakdown[3]:.1f} ms")

    if sessions:
        print(f"\n  All sessions: {', '.join(s[:8] for s in sessions)} …")
    print()

    con.close()


def _open_db_sessions(db: str, pattern: str) -> list[str]:
    """Return session IDs from db matching SQL LIKE pattern."""
    try:
        con = duckdb.connect(db, read_only=True)
    except duckdb.IOException:
        print(f"\n[ERROR] Cannot open '{db}'.", file=sys.stderr)
        sys.exit(1)
    rows = con.execute(
        "SELECT DISTINCT session_id FROM prefetch_events WHERE session_id LIKE ? ORDER BY session_id",
        [pattern],
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print APEX evaluation metrics")
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="Path to DuckDB file (overrides $APEX_DB_PATH / apex_eval.db)",
    )
    parser.add_argument(
        "--session", "-s",
        default=None,
        help="Session ID to query (default: most recent)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Print metrics for all sessions",
    )
    parser.add_argument(
        "--session-id-pattern",
        default=None,
        metavar="PATTERN",
        help="SQL LIKE pattern to select sessions (e.g. '%%' for all). Implies --all.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show pipeline latency breakdown (IIE, retrieval, push)",
    )
    args = parser.parse_args()

    db = args.db or _DEFAULT_DB

    if args.session_id_pattern is not None:
        # --session-id-pattern implies iterating over matched sessions
        sessions = _open_db_sessions(db, args.session_id_pattern)
        if not sessions:
            print(f"\nNo sessions matching pattern '{args.session_id_pattern}' in '{db}'.\n")
            return
        for sid in sessions:
            print_metrics(sid, db_path=db, verbose=args.verbose)
    elif args.all:
        sessions = _open_db_sessions(db, "%")
        for sid in sessions:
            print_metrics(sid, db_path=db, verbose=args.verbose)
    else:
        print_metrics(args.session, db_path=db, verbose=args.verbose)


if __name__ == "__main__":
    main()
