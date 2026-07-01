# Grounded PRP / PR-curve methodology

Short note documenting the replacement for the fabricated PRP benchmark. Pairs with
`scripts/bench_prp_grounded.py` and `apex/analytics/eval_metrics.py`.

## Why the old PRP was invalid
- `bench_baselines.py` fabricated the "claim" (was-the-prefetch-used) signal with
  modulo rules (`i % 2`, `i % 3 != 0`, `i % 3 != 2`) and assigned APEX the highest rate.
- `tau_calibrator.py` tuned τ to `TARGET_PRP = 0.65`, then `compute_prp()` reported
  ≥0.65 — circular.
- `compute_prp()` = claimed/total has no recall term; a silent system scores high.
- Even the live pipeline's "claim" = push delivered, i.e. delivery success ≠ user need.

## What the grounded benchmark measures
Scope: **retrieval + firing threshold (C3/C4 delivery)**. NOT intent inference (C2).

| Element | Old (invalid) | New (grounded) |
|---|---|---|
| Claim signal | modulo on trial index | top-k contains a chunk from a relevant source doc |
| Confidence | hardcoded 0.65 | query↔best-chunk cosine similarity (real) |
| Threshold τ | tuned to hit 0.65 | swept over full range; labels τ-independent |
| Metric | single PRP point | precision + recall + coverage curve, AP |
| Negatives | none | documented off-corpus control queries |

### Eval set
- Positives: 50 real MaintNet aircraft-maintenance queries (`experiment_corpus/queries.jsonl`).
- Negatives: 35 documented off-corpus control queries (no relevant source exists);
  synthetic and labelled as such. Without a negative class precision is trivially 1.0.
- Written to `experiment_corpus/eval_labels.jsonl` on each run for auditability.

### Relevance rule (τ-independent, auditable)
A retrieved chunk is relevant iff its source document is in the maintenance/mechanical
subset (`RELEVANT_SOURCES` in the script): `tm9_1803_engine_maintenance`,
`tm9_2320_vehicle_repair_parts`, `dtic_road_maintenance`, `dtic_aviation_logistics_mgmt`,
`tep_factory_operations_manual`. Everything else in the corpus (law of land warfare,
camouflage, survival, tents, artillery, civil disturbances) is topically unrelated.

## Definitions (fire iff confidence ≥ τ)
- TP = fired AND retrieved a relevant-source chunk
- precision = TP / fired   (PRP, now recall-anchored)
- recall = TP / (#queries with a relevant doc available)
- coverage = fired / N
- AP = area under the precision–recall curve

## First result (this Jetson, 11,926-chunk index, top-k=5, 50 positives + 35 controls)
- Base rate (fire everything, τ=0): precision 0.565, recall 0.96.
- As τ rises 0.25 → 0.45: precision 0.585 → 0.905, recall 0.96 → 0.76.
- Average precision (AP) = 0.81.
- Operating point @ precision≥0.65: τ=0.35 → precision 0.71, recall 0.96, coverage 0.80.
- False positives are real (off-corpus controls whose top-5 still pulled a maintenance
  chunk) — this is the signal the old precision-only PRP erased.
- Full curve + per-query trials: `ict_express/results/prp_grounded.json`.

## Known limitations (state these when citing)
1. Relevance is a **domain-level proxy** (source-document membership), not human hit@k.
   Swap in human judgements when they exist; the harness is unchanged.
2. Confidence is a **retrieval** score, not a measured intent-classifier confidence α.
   This benchmark does not validate C2 (intent inference), which is still unbuilt.
3. Negatives are synthetic controls, not observed off-task user behavior.

## Next steps to make it end-to-end
- Ground the claim in actual usage (user opened/cited the pushed doc), not retrieval hit.
- Build C2 and report intent accuracy α; use α to gate, then re-run this curve.
- Replace domain proxy with human relevance labels for true hit@k / MRR.
