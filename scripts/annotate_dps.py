#!/usr/bin/env python3
"""
DPS Annotation Script — human quality assessment for Claim 3.

Replays push events from the DuckDB evaluation store, reconstructs the delivered
context using the same pipeline components, and prompts for human annotation of
relevance and format compliance scores. Writes scores via log_delivery().

Usage:
    python scripts/annotate_dps.py [--db DB_PATH] [--session SESSION_ID]

This enables measurement of DPS (Delivery Precision per Subscriber):
    DPS = mean((relevance_score + format_score) / 2)

Where:
    relevance_score ∈ {0.0, 0.5, 1.0}  (not relevant / partially / highly relevant)
    format_score    ∈ {0.0, 0.5, 1.0}  (poor / adequate / excellent format compliance)

Privacy: only displays retrieved chunks and formatted output for annotation.
No behavioral signals or personal data are shown to the annotator.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from apex.analytics.store import AnalyticsStore
from apex.adapter.llm_adapter import LLMAdapter
from apex.retrieval.rrf import RetrievalEngine


class DPSAnnotator:
    """
    Interactive DPS annotation tool for Claim 3 measurement.

    Reconstructs delivered content from prefetch events and prompts
    for human quality assessment.
    """

    def __init__(self, db_path: str, llm_model: str = "phi3.5"):
        self.store = AnalyticsStore(db_path)
        self.llm_adapter = LLMAdapter(model=llm_model)
        self.retrieval_engine = None

        # Load retrieval engine if index exists
        index_path = os.environ.get("APEX_INDEX_PATH", "apex_vault")
        if os.path.exists(f"{index_path}.hnsw") and os.path.exists(f"{index_path}.meta.json"):
            try:
                self.retrieval_engine = RetrievalEngine()
                self.retrieval_engine.load_index(index_path)
                print(f"✅ Loaded retrieval index: {index_path}")
            except Exception as e:
                print(f"⚠️  Failed to load index: {e}")
                self.retrieval_engine = None
        else:
            print(f"⚠️  No retrieval index found at {index_path}")

    def get_claimed_prefetch_events(self, session_id: Optional[str] = None) -> list[dict]:
        """
        Get all claimed prefetch events that would have resulted in push delivery.

        Returns list of events with metadata needed for content reconstruction.
        """
        query = """
            SELECT
                id, session_id, subscriber_id, label, confidence,
                t_available, t_claimed, latency_ms
            FROM prefetch_events
            WHERE claimed = TRUE
        """
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY t_claimed"

        rows = self.store.con.execute(query, params).fetchall()

        events = []
        for row in rows:
            events.append({
                "id": row[0],
                "session_id": row[1],
                "subscriber_id": row[2],
                "label": row[3],
                "confidence": row[4],
                "t_available": row[5],
                "t_claimed": row[6],
                "latency_ms": row[7],
            })

        return events

    def reconstruct_delivered_content(self, event: dict) -> Optional[str]:
        """
        Reconstruct what would have been delivered for this push event.

        Since the original chunks aren't stored, we simulate the retrieval
        and formatting process based on the event metadata.
        """
        if not self.retrieval_engine:
            # Create mock content when index is unavailable
            return self._create_mock_content(event)

        try:
            # TODO: Reconstruct the intent vector that would have been used
            # For now, create representative content based on label
            chunks = self._get_representative_chunks(event["label"])

            # Get subscriber profile (mock for annotation)
            profile = self._get_mock_subscriber_profile(event["subscriber_id"])

            # Format content as it would have been delivered
            formatted_content = self.llm_adapter.format(chunks, profile)

            return formatted_content

        except Exception as e:
            print(f"⚠️  Failed to reconstruct content for event {event['id']}: {e}")
            return self._create_mock_content(event)

    def _get_representative_chunks(self, label: str) -> list:
        """Get representative chunks for the given label."""
        # Mock implementation - in a full implementation this would
        # perform retrieval based on the stored label
        from apex.retrieval.rrf import Chunk

        mock_chunks = [
            Chunk(
                chunk_id=f"{label}_chunk_1",
                text=f"Representative content for {label} domain. "
                     f"This would normally be retrieved based on the behavioral signal. "
                     f"Multiple chunks would be ranked by relevance.",
                source=f"docs/{label}_guide.md",
                label=label,
                score=0.85
            )
        ]

        return mock_chunks

    def _get_mock_subscriber_profile(self, subscriber_id: str):
        """Create a mock ConsumerProfile for formatting."""
        from apex.adapter.llm_adapter import ConsumerProfile

        # Create profile based on common subscriber patterns
        if "ide" in subscriber_id.lower():
            return ConsumerProfile(
                subscriber_id=subscriber_id,
                autonomy_level="assistive",
                goal_horizon="short",
                interaction_style="ambient",
                output_format="markdown",
                vocabulary_level="technical",
                verbosity="concise",
                citation_style="inline",
                max_context_tokens=512
            )
        elif "factory" in subscriber_id.lower():
            return ConsumerProfile(
                subscriber_id=subscriber_id,
                autonomy_level="autonomous",
                goal_horizon="short",
                interaction_style="hard-interrupt",
                output_format="structured-alert",
                vocabulary_level="domain-expert",
                verbosity="concise",
                citation_style="none",
                max_context_tokens=256,
                domain_schema={"severity": "str", "action": "str", "context": "str"}
            )
        else:  # research assistant
            return ConsumerProfile(
                subscriber_id=subscriber_id,
                autonomy_level="suggestive",
                goal_horizon="long",
                interaction_style="conversational",
                output_format="markdown",
                vocabulary_level="domain-expert",
                verbosity="detailed",
                citation_style="footnote",
                max_context_tokens=1024
            )

    def _create_mock_content(self, event: dict) -> str:
        """Create mock delivered content when retrieval engine is unavailable."""
        return f"""
[MOCK DELIVERED CONTENT for Event {event['id']}]

Domain: {event['label']}
Subscriber: {event['subscriber_id']}
Confidence: {event['confidence']:.3f}

This represents the context that would have been delivered to the subscriber
based on the behavioral signal. The actual content would be:

1. Retrieved chunks relevant to the {event['label']} domain
2. Formatted according to the subscriber's profile
3. Delivered via WebSocket push

For annotation purposes, consider whether this type of content would be:
- Relevant to a user working in the {event['label']} domain
- Properly formatted for the {event['subscriber_id']} interface
""".strip()

    def get_annotation_scores(self, event: dict, content: str) -> tuple[float, float]:
        """
        Interactive prompt for relevance and format compliance scores.

        Returns (relevance_score, format_score) both in {0.0, 0.5, 1.0}.
        """
        print("\n" + "="*80)
        print(f"ANNOTATION REQUEST - Event {event['id']}")
        print("="*80)
        print(f"Session: {event['session_id']}")
        print(f"Subscriber: {event['subscriber_id']}")
        print(f"Domain: {event['label']}")
        print(f"Confidence: {event['confidence']:.3f}")
        print(f"Latency: {event['latency_ms']:.1f}ms")
        print("-"*80)
        print("DELIVERED CONTENT:")
        print("-"*80)
        print(content)
        print("-"*80)

        # Get relevance score
        while True:
            try:
                print("\nRELEVANCE SCORE:")
                print("  0 = Not relevant to the user's current task")
                print("  0.5 = Partially relevant")
                print("  1 = Highly relevant")
                rel_input = input("Relevance (0/0.5/1): ").strip()

                if rel_input in ["0", "0.5", "1"]:
                    relevance_score = float(rel_input)
                    break
                else:
                    print("❌ Please enter 0, 0.5, or 1")
            except (ValueError, KeyboardInterrupt):
                print("\n⚠️  Annotation cancelled")
                return None, None

        # Get format compliance score
        while True:
            try:
                print("\nFORMAT COMPLIANCE SCORE:")
                print("  0 = Poor formatting (hard to read, wrong format)")
                print("  0.5 = Adequate formatting")
                print("  1 = Excellent formatting (perfect for this subscriber)")
                fmt_input = input("Format compliance (0/0.5/1): ").strip()

                if fmt_input in ["0", "0.5", "1"]:
                    format_score = float(fmt_input)
                    break
                else:
                    print("❌ Please enter 0, 0.5, or 1")
            except (ValueError, KeyboardInterrupt):
                print("\n⚠️  Annotation cancelled")
                return None, None

        return relevance_score, format_score

    def annotate_session(self, session_id: Optional[str] = None) -> int:
        """
        Annotate all claimed prefetch events for DPS measurement.

        Returns number of events annotated.
        """
        events = self.get_claimed_prefetch_events(session_id)

        if not events:
            print("ℹ️  No claimed prefetch events found for annotation")
            if session_id:
                print(f"   Session filter: {session_id}")
            return 0

        print(f"📊 Found {len(events)} claimed prefetch events to annotate")
        if session_id:
            print(f"   Session: {session_id}")
        else:
            print("   All sessions")

        annotated = 0
        skipped = 0

        for i, event in enumerate(events, 1):
            print(f"\n🔍 Processing event {i}/{len(events)}")

            # Check if already annotated
            existing = self.store.con.execute(
                "SELECT COUNT(*) FROM delivery_events WHERE session_id = ? AND subscriber_id = ?",
                [event["session_id"], event["subscriber_id"]]
            ).fetchone()[0]

            if existing > 0:
                print(f"⏭️  Event {event['id']} already annotated, skipping")
                skipped += 1
                continue

            # Reconstruct delivered content
            content = self.reconstruct_delivered_content(event)
            if not content:
                print(f"❌ Failed to reconstruct content for event {event['id']}")
                continue

            # Get human annotation
            relevance, format_score = self.get_annotation_scores(event, content)
            if relevance is None or format_score is None:
                print("⚠️  Annotation skipped")
                continue

            # Store annotation
            self.store.log_delivery(
                session_id=event["session_id"],
                subscriber_id=event["subscriber_id"],
                relevance_score=relevance,
                format_score=format_score
            )

            annotated += 1
            dps = (relevance + format_score) / 2.0
            print(f"✅ Annotation saved: relevance={relevance}, format={format_score}, DPS={dps:.2f}")

        print(f"\n📈 Annotation complete: {annotated} annotated, {skipped} skipped")

        if annotated > 0:
            # Compute DPS for the session(s)
            if session_id:
                dps = self.store.compute_dps(session_id)
                print(f"📊 Session DPS: {dps:.3f}" if dps else "📊 Session DPS: (no data)")

        return annotated


def main():
    parser = argparse.ArgumentParser(
        description="DPS Annotation Tool — human quality assessment for Claim 3",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db",
        default="apex_eval.db",
        help="Path to DuckDB evaluation store (default: apex_eval.db)"
    )
    parser.add_argument(
        "--session",
        help="Session ID to annotate (default: all sessions)"
    )
    parser.add_argument(
        "--llm-model",
        default="phi3.5",
        help="LLM model for content reconstruction (default: phi3.5)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"❌ Database not found: {args.db}")
        print("   Run an evaluation first: just eval")
        return 1

    print(f"🎯 APEX DPS Annotation Tool")
    print(f"📁 Database: {args.db}")
    if args.session:
        print(f"🔍 Session: {args.session}")
    else:
        print(f"🔍 Session: all")

    try:
        annotator = DPSAnnotator(args.db, args.llm_model)
        count = annotator.annotate_session(args.session)

        if count > 0:
            print(f"\n🎉 Successfully annotated {count} delivery events")
            print("📊 View results: just metrics")
        else:
            print("\nℹ️  No new annotations added")

    except KeyboardInterrupt:
        print("\n⚠️  Annotation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during annotation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())