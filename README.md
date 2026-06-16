# CMO Sensor Fusion: calibrated combat identification

This repository supports a pipeline in which Command: Modern Operations (CMO) supplies observations and hidden ground truth, Graph DB queries retrieve supporting/contradicting evidence, and a probabilistic model produces a **combat identification**: a candidate identity plus a calibrated probability. The LLM may explain the result, but must not supply the probability.

```text
CMO Lua export -> Graph DB ingestion/query -> candidate logits -> calibration -> combat ID + probability
                                      CMO ground truth ----------^
```

## Input contract

Export one JSON object per contact/timepoint. `scores` are uncalibrated logits produced from graph-derived evidence (support, contradiction, source quality, recency, independent paths, sensor observations, etc.). `truth` is the identity revealed by CMO after the run. Preserve IDs so every estimate is auditable back to its scenario and evidence query.

```json
{"scenario_id":"raid-001","contact_id":"C-101","observation_time":"2026-06-15T10:00:00Z","scores":{"friendly_fighter":0.8,"hostile_fighter":2.8},"truth":"hostile_fighter","evidence_query_id":"q-101"}
```

Candidate keys must be identical across a dataset. Fit and evaluate on scenario-disjoint datasets to avoid learning scenario-specific contacts. Calibration data should represent intended operating conditions; report results by sensor mix, range, track quality, scenario family, and candidate class to expose distribution shift.

## Usage

The dependency-free package implements multiclass temperature scaling. This method preserves the graph model's candidate ranking while correcting global over/under-confidence by minimizing held-out negative log loss.

```bash
python -m combat_id_calibration fit examples/cmo_graph_scores.jsonl calibration-model.json
python -m combat_id_calibration evaluate calibration-model.json examples/cmo_graph_scores.jsonl --bins 10
python -m combat_id_calibration apply calibration-model.json new-scores.jsonl calibrated-identifications.jsonl
```

`apply` adds the full candidate distribution, the highest-probability combat ID, and its probability. `evaluate` reports accuracy, multiclass Brier score, log loss, expected calibration error (ECE), and reliability bins before and after calibration. A calibrated 0.7 prediction means that, among comparable predictions assigned 0.7, about 70% should be correct.

## Operational safeguards

- Fit only on labelled historical/simulated outcomes; never fit and assess on the same scenarios in operational evaluation.
- Retain the complete candidate distribution and evidence-query reference, not only the winning identity.
- Treat low sample counts, empty reliability bins, and shifted sensor/scenario conditions as uncertainty warnings.
- Refit/version the calibrator whenever the graph schema, retrieval, score model, CMO scenario population, or candidate taxonomy changes.
- Use reliability diagrams and class/condition-specific reports before trusting probabilities for decision support.
