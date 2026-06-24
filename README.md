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


## Recommended graph database package

Use **Kuzu** (`kuzu` Python package) as the default Graph DB for this project. It is a good fit because this repository currently expects an offline/simulation pipeline: CMO exports observations, Python transforms them, graph queries produce evidence-derived candidate logits, and the local calibrator turns those logits into calibrated probabilities. An embedded property-graph database keeps that workflow reproducible without requiring a separate graph server for each experiment.

Recommended requirements for using Kuzu here:

- **Runtime:** keep the existing Python 3.10+ baseline and install the optional graph extras in the environment that performs graph ingestion and evidence queries: `python -m pip install -e .[graph]`. Run Ollama locally with `qwen3.5:9b` available, for example `ollama pull qwen3.5:9b` and `ollama serve`.
- **Data model:** represent CMO entities as typed nodes, such as scenarios, contacts, observations, platforms, sensors, emitters, tracks, evidence sources, and candidate identities. Represent evidence as typed relationships, such as `OBSERVED_BY`, `EMITTED`, `CLASSIFIED_AS`, `SUPPORTS`, `CONTRADICTS`, `NEAR`, and `DERIVED_FROM`.
- **Auditability:** persist `scenario_id`, `contact_id`, `observation_time`, `evidence_query_id`, source reliability, recency, sensor mix, and graph/schema version on nodes or relationships so every calibrated estimate can be traced back to the CMO run and graph query that produced it.
- **Query output contract:** graph queries should not output final probabilities. They should output deterministic, auditable features or logits per candidate, including support score, contradiction score, independent path count, path lengths, source quality, recency, and sensor/track quality. Those values map into the `scores` contract consumed by this package.
- **Calibration discipline:** refit and version the calibration model whenever the graph schema, evidence extraction queries, feature/logit construction, scenario population, or candidate taxonomy changes. Fit and evaluate on scenario-disjoint datasets.
- **When to choose Neo4j instead:** use Neo4j plus the official Python driver if the work needs a long-running multi-user graph service, browser-based operational exploration, managed hosting, role-based access control, or integration with Neo4j Graph Data Science. In that case, keep the same feature/logit output contract so the calibration layer remains database-agnostic.


### Ollama-to-Kuzu ingestion pipeline

The `ingest-graph` command extracts source text from local PDFs and Wikipedia pages, chunks the text, asks the local Ollama model to return strict JSON facts, optionally writes those facts to JSONL for review, and populates a Kuzu graph with `Entity`, `Source`, `FACT`, and `MENTIONED_IN` records.

```bash
python -m combat_id_calibration ingest-graph \
  --model qwen3.5:9b \
  --ollama-url http://localhost:11434 \
  --pdf 34_A_Holistic_Approach_to_Combat_Identification_200701.pdf \
  --wikipedia https://en.wikipedia.org/wiki/Identification_friend_or_foe \
  --facts-jsonl extracted-facts.jsonl \
  --db cmo-evidence.kuzu
```

Pipeline stages and requirements:

1. **Source loading:** PDF ingestion uses `pypdf`; Wikipedia ingestion uses the standard-library HTTP and HTML parsers.
2. **LLM extraction:** the command calls Ollama's local `/api/generate` endpoint and defaults to `qwen3.5:9b`; extracted facts are constrained to JSON triples with evidence snippets and confidence values.
3. **Human/audit review:** use `--facts-jsonl` to inspect extracted facts before relying on the graph.
4. **Kuzu population:** the pipeline creates a minimal property-graph schema and writes entities, sources, fact edges, provenance, evidence text, and extraction confidence.
5. **Downstream calibration:** graph queries over this Kuzu database should produce candidate features or logits that conform to the `scores` input contract below; the graph extraction step is not a substitute for calibration against CMO truth labels.


### Wikipedia Kuzu knowledge graph notebook

A companion notebook, [`notebooks/wikipedia_airborne_radars_kuzu_kg.ipynb`](notebooks/wikipedia_airborne_radars_kuzu_kg.ipynb), builds a local Kuzu knowledge graph from Wikipedia articles for the MiG-29, the Ukrainian Air Force, the Russian Air Force, the N011M Bars radar, and representative Russian and Israeli airborne radars. It reuses the repository's `combat_id_calibration.graph_ingest` module to fetch Wikipedia pages, extract auditable facts with local Ollama, write a review JSONL file, and populate the standard `Entity`, `Source`, `FACT`, and `MENTIONED_IN` Kuzu schema.

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
