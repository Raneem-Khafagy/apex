"""
Precision–Recall evaluation for proactive retrieval.
=====================================================

This module replaces the single-point, threshold-tuned PRP number with a full
precision–recall curve swept over the firing threshold τ. It exists to answer
two specific methodological objections to the original PRP metric:

  (a) Circularity — the old τ calibrator tuned τ to TARGET_PRP = 0.65 and then
      reported "PRP ≥ 0.65" as the result. Here τ is a free variable swept over
      its whole range; the ground-truth relevance labels are fixed independently
      of τ, so no tuning-to-the-reported-number can occur.

  (b) Precision-only — the old PRP had no recall counterpart, so a near-silent
      system scored ~0.9 by firing almost never. Here every threshold reports
      precision AND recall AND coverage, exposing the retrieve-less/score-higher
      tradeoff explicitly.

An *event* is one (behavioral-signal → retrieval) trial with three fields:

    confidence          float   real retrieval/inference confidence for this trial
    needed              bool    a topically-relevant document exists in the corpus
                                for this query (ground truth, τ-independent)
    retrieved_relevant  bool    the top-k actually returned a relevant-source chunk
                                (real claim signal — replaces the i % k modulo)

`retrieved_relevant` can only be True when `needed` is True.

Definitions at a given threshold τ (fire iff confidence ≥ τ):

    TP        = fired AND retrieved_relevant          (delivered the right context)
    fired     = confidence ≥ τ
    precision = TP / fired                             (PRP, now recall-anchored)
    recall    = TP / (total events with needed=True)  (coverage of real needs)
    coverage  = fired / N                              (how loud the system is)

Precision alone is gameable by lowering coverage; the curve is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class EvalEvent:
    """One retrieval trial with a τ-independent ground-truth relevance label."""
    confidence: float
    needed: bool
    retrieved_relevant: bool

    def __post_init__(self) -> None:
        if self.retrieved_relevant and not self.needed:
            raise ValueError(
                "retrieved_relevant=True requires needed=True "
                "(cannot retrieve a relevant chunk when none exists)"
            )


@dataclass(frozen=True)
class PRPoint:
    tau: float
    fired: int
    tp: int
    precision: Optional[float]   # None when nothing fired (undefined, not 0)
    recall: float
    coverage: float


def precision_recall_curve(
    events: Iterable[EvalEvent],
    thresholds: Optional[Iterable[float]] = None,
) -> list[PRPoint]:
    """
    Sweep the firing threshold and compute precision/recall/coverage at each.

    Parameters
    ----------
    events
        Retrieval trials. Confidence values need not be in [0, 1]; thresholds
        are compared directly against them.
    thresholds
        Threshold values to evaluate. If None, uses every distinct confidence
        present in the data (the only thresholds that change the outcome), which
        yields the exact curve with no interpolation artefacts.

    Returns
    -------
    list[PRPoint] sorted by ascending τ.
    """
    events = list(events)
    n = len(events)
    if n == 0:
        return []

    total_needed = sum(1 for e in events if e.needed)

    if thresholds is None:
        thresholds = sorted({e.confidence for e in events})

    points: list[PRPoint] = []
    for tau in sorted(thresholds):
        fired = [e for e in events if e.confidence >= tau]
        n_fired = len(fired)
        tp = sum(1 for e in fired if e.retrieved_relevant)
        precision = (tp / n_fired) if n_fired > 0 else None
        recall = (tp / total_needed) if total_needed > 0 else 0.0
        coverage = n_fired / n
        points.append(PRPoint(
            tau=round(float(tau), 6),
            fired=n_fired,
            tp=tp,
            precision=round(precision, 4) if precision is not None else None,
            recall=round(recall, 4),
            coverage=round(coverage, 4),
        ))
    return points


def average_precision(points: list[PRPoint]) -> Optional[float]:
    """
    Area under the precision–recall curve (AP), the standard single-number
    summary that — unlike a tuned PRP — cannot be inflated by lowering coverage.

    Computed as sum over the curve of precision * (Δrecall), walking from high
    recall to low recall. Points where nothing fired (precision undefined) are
    skipped.
    """
    usable = [p for p in points if p.precision is not None]
    if not usable:
        return None
    # Order by ascending recall for a clean Δrecall integration.
    usable = sorted(usable, key=lambda p: p.recall)
    ap = 0.0
    prev_recall = 0.0
    for p in usable:
        d_recall = p.recall - prev_recall
        if d_recall > 0:
            ap += p.precision * d_recall
            prev_recall = p.recall
    return round(ap, 4)


def operating_point(
    points: list[PRPoint],
    min_precision: float = 0.65,
) -> Optional[PRPoint]:
    """
    Pick the honest operating point: the threshold that MAXIMISES RECALL subject
    to precision ≥ min_precision. This is the inverse of the old circular logic —
    instead of tuning τ to hit a precision target and reporting that precision,
    we fix a precision floor and report the recall we can actually achieve at it.
    Returns None if no threshold meets the precision floor.
    """
    eligible = [p for p in points if p.precision is not None and p.precision >= min_precision]
    if not eligible:
        return None
    return max(eligible, key=lambda p: p.recall)
