"""
Grounded PRP / Precision–Recall Benchmark
==========================================
Replaces the fabricated modulo claim signal in bench_baselines.py with a REAL,
retrieval-grounded claim, and reports a full precision–recall curve over the
firing threshold τ instead of a single τ tuned to TARGET_PRP = 0.65.

WHAT CHANGED vs bench_baselines.py
----------------------------------
  Old claim:  store.log_claim() gated by `i % 2`, `i % 3 != 0`, `i % 3 != 2`.
              APEX was ASSIGNED the highest claim rate ("higher claim rate due to
              better precision") — i.e. the conclusion was hardcoded.
  New claim:  a prefetch is CLAIMED iff the top-k it actually retrieved contains a
              chunk from a topically-relevant source document for the query. This
              is computed from real retrieval output, never from the trial index.

  Old score:  PRP = claimed/total at the one τ the calibrator tuned to 0.65.
  New score:  precision AND recall AND coverage at every τ (a PR curve), plus AP
              and an operating point chosen by fixing a precision floor and
              reporting the recall achieved (the inverse of the circular logic).

HONEST SCOPE — read before citing any number this produces
----------------------------------------------------------
  * This validates the RETRIEVAL + THRESHOLD (C3/C4 delivery) layer only. It does
    NOT validate intent inference (C2), which is not yet built. Confidence here is
    a real retrieval score, not a measured intent-classifier confidence α.
  * Relevance is a DOMAIN-LEVEL PROXY: "the top-k contains a chunk from a source
    document in RELEVANT_SOURCES for this query." It is NOT human hit@k relevance.
    It is a stated, reproducible rule (see RELEVANT_SOURCES) that a reviewer can
    inspect and that is fixed independently of τ. Replace it with human relevance
    judgements when they exist — the harness is unchanged, only the label source.
  * Positives are 50 real MaintNet maintenance queries. Negatives are a documented
    set of off-corpus control queries where firing is wrong (no relevant source).
    The controls are synthetic and labelled as such; they are the negative class
    that lets precision fall — without them precision is trivially 1.0.

Output: JSON → ict_express/results/prp_grounded.json
Usage:  uv run python scripts/bench_prp_grounded.py
Requires: ollama serve + all-minilm (for embeddings). phi3.5 not required.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex.analytics.eval_metrics import (
    EvalEvent,
    average_precision,
    operating_point,
    precision_recall_curve,
)
from apex.inference.intent_engine import _embed, EMBED_MODEL
from apex.ingest.ingestor import Ingestor
from apex.retrieval.rrf import RetrievalEngine

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX_PATH = os.path.join(ROOT, "experiment_index", "experiment")
QUERIES_PATH = os.path.join(ROOT, "experiment_corpus", "queries.jsonl")
RESULTS_DIR = os.path.join(ROOT, "..", "ict_express", "results")
LABELS_OUT = os.path.join(ROOT, "experiment_corpus", "eval_labels.jsonl")

TOP_K = 5

# ── Ground-truth relevance rule (τ-INDEPENDENT, auditable) ────────────────────
# The corpus is Army doctrine + mechanical maintenance manuals. The MaintNet
# queries are aircraft/engine maintenance faults. A retrieved chunk counts as
# relevant iff its source document is in this maintenance/mechanical subset.
# Everything else in the corpus (law of land warfare, camouflage, survival,
# civil disturbances, tents, artillery gunnery ...) is topically unrelated.
RELEVANT_SOURCES = {
    "tm9_1803_engine_maintenance.txt",
    "tm9_2320_vehicle_repair_parts.txt",
    "dtic_road_maintenance.txt",
    "dtic_aviation_logistics_mgmt.txt",
    "tep_factory_operations_manual.txt",
}

# ── Off-corpus control queries (NEGATIVE class, synthetic, documented) ────────
# These have NO relevant source document. If a query like this crosses τ and is
# delivered, that is a false positive. Without a negative class, precision is
# trivially 1.0 and the metric is meaningless — this set is what makes the PR
# curve informative. Topics deliberately absent from a military-maintenance corpus.
CONTROL_QUERIES = [
    # food / home
    "best sourdough bread recipe for a home oven",
    "how to remove red wine stains from carpet",
    "how to compost food scraps in an apartment",
    "recommended houseplants for low light apartments",
    "how long to boil an egg for a soft yolk",
    "how to descale a coffee machine with vinegar",
    "easy vegetarian dinner ideas for weeknights",
    # finance / work / life admin
    "how to file quarterly taxes as a freelancer",
    "how to negotiate a software engineer salary offer",
    "difference between a Roth and traditional IRA",
    "how to write a two-week resignation letter",
    "steps to refinance a home mortgage",
    # software / tech (unrelated to the corpus domain)
    "python asyncio event loop already running error",
    "how to set up a Kubernetes ingress controller",
    "git how to undo the last commit but keep changes",
    "why is my React useEffect running twice",
    "how to center a div with flexbox",
    "postgres connection pool exhausted troubleshooting",
    # travel / geography
    "cheapest flights from Tokyo to Seoul in spring",
    "what is the capital of Portugal",
    "best time of year to visit Iceland",
    "do I need a visa to travel to Japan as a tourist",
    # health / fitness
    "how do I train for a first marathon",
    "symptoms of vitamin D deficiency in adults",
    "how much water should I drink per day",
    "beginner yoga poses for lower back pain",
    # hobbies / culture
    "how to knit a beginner scarf step by step",
    "explain the plot of Hamlet in simple terms",
    "beginner guitar chords for pop songs",
    "rules of chess for a complete beginner",
    "how to start a small vegetable garden",
    "what camera settings for night photography",
    "how to brew pour-over coffee at home",
    "recommended fantasy novels similar to Tolkien",
    "how to teach a dog to sit and stay",
]


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _source_basename(source: str) -> str:
    return os.path.basename(source) if source else source


def load_queries() -> list[dict]:
    """Load positives (real MaintNet) + negatives (controls) as a labelled set."""
    positives = []
    with open(QUERIES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            positives.append({
                "qid": row["id"],
                "text": row["text"],
                "class": "positive",
                "needed": True,          # a relevant maintenance manual exists
            })
    negatives = [
        {"qid": f"ctrl_{i:03d}", "text": t, "class": "control", "needed": False}
        for i, t in enumerate(CONTROL_QUERIES)
    ]
    return positives + negatives


def load_engine() -> RetrievalEngine:
    engine = RetrievalEngine()
    ingestor = Ingestor(engine)
    if os.path.exists(INDEX_PATH + ".hnsw") and os.path.exists(INDEX_PATH + ".meta.json"):
        ingestor.load_index(INDEX_PATH)
        print(f"  Index loaded: {len(ingestor._metadata)} chunks")
    else:
        raise SystemExit(f"Index not found at {INDEX_PATH} — run the ingest step first.")
    return engine


def run_trials(engine: RetrievalEngine, queries: list[dict]) -> list[dict]:
    """
    One retrieval trial per query. Confidence = real top-1 retrieval score.
    Claim (retrieved_relevant) = top-k contains a RELEVANT_SOURCES chunk.
    No modulo anywhere; the claim is a function of retrieval output, not trial index.
    """
    trials = []
    for i, q in enumerate(queries):
        q_hat = _norm(_embed(q["text"], embed_model=EMBED_MODEL))
        chunks = engine.search(q_hat, label="reading_reference", k=TOP_K)
        # Confidence = cosine similarity between the query and its best retrieved
        # chunk. This is a real, computed retrieval-confidence signal that actually
        # separates on-topic from off-topic queries — unlike the RRF fusion score,
        # which is rank-based and near-constant. Fully τ-independent.
        best_sim = 0.0
        for c in chunks:
            sim = float(np.dot(q_hat, _norm(_embed(c.text, embed_model=EMBED_MODEL))))
            best_sim = max(best_sim, sim)
        hit_sources = [_source_basename(c.source) for c in chunks]
        retrieved_relevant = any(s in RELEVANT_SOURCES for s in hit_sources)
        trials.append({
            "qid": q["qid"],
            "class": q["class"],
            "needed": q["needed"],
            "confidence": round(best_sim, 4),
            "retrieved_relevant": bool(retrieved_relevant and q["needed"]),
            "top_sources": hit_sources,
        })
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(queries)} trials")
    return trials


def main() -> None:
    print("Loading retrieval engine...")
    engine = load_engine()

    queries = load_queries()
    n_pos = sum(1 for q in queries if q["needed"])
    n_neg = len(queries) - n_pos
    print(f"Eval set: {n_pos} positives (MaintNet) + {n_neg} controls = {len(queries)} queries")

    # Persist the labelled eval set so the relevance basis is auditable.
    with open(LABELS_OUT, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote labelled eval set → {LABELS_OUT}")

    print("Running trials (real retrieval, grounded claim)...")
    trials = run_trials(engine, queries)

    events = [
        EvalEvent(
            confidence=t["confidence"],
            needed=t["needed"],
            retrieved_relevant=t["retrieved_relevant"],
        )
        for t in trials
    ]

    # Sweep a fixed grid of thresholds for a readable, evenly-spaced curve.
    grid = [round(x, 3) for x in np.linspace(0.0, 0.9, 19)]
    curve = precision_recall_curve(events, thresholds=grid)
    ap = average_precision(precision_recall_curve(events))  # AP on the exact curve
    op = operating_point(curve, min_precision=0.65)

    print("\nPrecision–Recall curve (τ = query↔best-chunk cosine similarity):")
    print(f"{'tau':>8}  {'fired':>6}  {'TP':>4}  {'precision':>9}  {'recall':>7}  {'coverage':>8}")
    print(f"{'-'*8}  {'-'*6}  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*8}")
    for p in curve:
        prec = f"{p.precision:.3f}" if p.precision is not None else "  n/a"
        print(f"{p.tau:>8.4f}  {p.fired:>6}  {p.tp:>4}  {prec:>9}  {p.recall:>7.3f}  {p.coverage:>8.3f}")

    print(f"\nAverage precision (area under PR curve): {ap}")
    if op is not None:
        print(f"Operating point @ precision≥0.65: τ={op.tau:.4f} → "
              f"precision={op.precision:.3f}, recall={op.recall:.3f}, coverage={op.coverage:.3f}")
    else:
        print("Operating point @ precision≥0.65: NONE — no threshold reaches 65% precision.")

    results = {
        "scope": "retrieval+threshold (C3/C4) only; NOT intent inference (C2)",
        "claim_definition": "top-k contains a chunk from a RELEVANT_SOURCES document",
        "relevance_basis": "domain-level source-document proxy (NOT human hit@k)",
        "relevant_sources": sorted(RELEVANT_SOURCES),
        "n_positive": n_pos,
        "n_control": n_neg,
        "top_k": TOP_K,
        "average_precision": ap,
        "operating_point_p65": (
            None if op is None else
            {"tau": op.tau, "precision": op.precision, "recall": op.recall, "coverage": op.coverage}
        ),
        "curve": [
            {"tau": p.tau, "fired": p.fired, "tp": p.tp,
             "precision": p.precision, "recall": p.recall, "coverage": p.coverage}
            for p in curve
        ],
        "trials": trials,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "prp_grounded.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
