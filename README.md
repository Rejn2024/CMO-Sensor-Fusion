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

Use **Neo4j** with the official `neo4j` Python driver as the default Graph DB for this project. It is a good fit because this repository now targets a long-running, inspectable graph service for CMO evidence: CMO exports observations, Python transforms them, Neo4j Cypher queries produce evidence-derived candidate logits, and the local calibrator turns those logits into calibrated probabilities. Neo4j also supports browser-based operational exploration, managed hosting, role-based access control, and integration with Neo4j Graph Data Science while preserving the repository's database-agnostic calibration contract.

Recommended requirements for using Neo4j here:

- **Runtime:** keep the existing Python 3.10+ baseline and install the optional graph extras in the environment that performs graph ingestion and evidence queries: `python -m pip install -e .[graph]`. Run a Neo4j server reachable over Bolt, and run Ollama locally with `qwen3.5:9b` available, for example `ollama pull qwen3.5:9b` and `ollama serve`.
- **Data model:** represent CMO entities as typed nodes, such as scenarios, contacts, observations, platforms, sensors, emitters, tracks, evidence sources, and candidate identities. Represent evidence as typed relationships, such as `OBSERVED_BY`, `EMITTED`, `CLASSIFIED_AS`, `SUPPORTS`, `CONTRADICTS`, `NEAR`, and `DERIVED_FROM`.
- **Auditability:** persist `scenario_id`, `contact_id`, `observation_time`, `evidence_query_id`, source reliability, recency, sensor mix, and graph/schema version on nodes or relationships so every calibrated estimate can be traced back to the CMO run and graph query that produced it.
- **Query output contract:** graph queries should not output final probabilities. They should output deterministic, auditable features or logits per candidate, including support score, contradiction score, independent path count, path lengths, source quality, recency, and sensor/track quality. Those values map into the `scores` contract consumed by this package.
- **Calibration discipline:** refit and version the calibration model whenever the graph schema, evidence extraction queries, feature/logit construction, scenario population, or candidate taxonomy changes. Fit and evaluate on scenario-disjoint datasets.


### Ollama-to-Neo4j ingestion pipeline

The `ingest-graph` command extracts source text from local PDFs and Wikipedia pages, chunks the text, asks the local Ollama model to return strict JSON facts, optionally writes those facts to JSONL for review, and populates a Neo4j graph with `Entity`, `Source`, `FACT`, and `MENTIONED_IN` records.

```bash
python -m combat_id_calibration ingest-graph \
  --model qwen3.5:9b \
  --ollama-url http://localhost:11434 \
  --pdf 34_A_Holistic_Approach_to_Combat_Identification_200701.pdf \
  --wikipedia https://en.wikipedia.org/wiki/Identification_friend_or_foe \
  --facts-jsonl extracted-facts.jsonl \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password "$NEO4J_PASSWORD"
```

Pipeline stages and requirements:

1. **Source loading:** PDF ingestion uses `pypdf`; Wikipedia ingestion uses the standard-library HTTP and HTML parsers.
2. **LLM extraction:** the command calls Ollama's local `/api/generate` endpoint and defaults to `qwen3.5:9b`; extracted facts are constrained to JSON triples with evidence snippets and confidence values.
3. **Human/audit review:** use `--facts-jsonl` to inspect extracted facts before relying on the graph.
4. **Neo4j population:** the pipeline creates uniqueness constraints and writes entities, sources, fact edges, provenance, evidence text, and extraction confidence.
5. **Downstream calibration:** Cypher queries over this Neo4j database should produce candidate features or logits that conform to the `scores` input contract below; the graph extraction step is not a substitute for calibration against CMO truth labels.


### Wikipedia Neo4j knowledge graph notebook

A companion notebook, [`notebooks/wikipedia_airborne_radars_neo4j_kg.ipynb`](notebooks/wikipedia_airborne_radars_neo4j_kg.ipynb), builds a local Neo4j knowledge graph from Wikipedia articles for the MiG-29, the Ukrainian Air Force, the Russian Air Force, the N011M Bars radar, and representative Russian and Israeli airborne radars. It reuses the repository's `combat_id_calibration.graph_ingest` module to fetch Wikipedia pages, extract auditable facts with local Ollama, write a review JSONL file, and populate the standard `Entity`, `Source`, `FACT`, and `MENTIONED_IN` Neo4j schema.



### CMO emission-observation ingestion

`event_export_lua_02.lua` emits `PY_CONTACT_LOG` lines that are treated as Phase-2 graph observations. The `ingest-cmo-observations` command parses each line into a typed ontology and writes it into the same Neo4j database used by the Wikipedia airborne-radar notebook. This route keeps static reference facts from Wikipedia and dynamic CMO observations in one evidence graph: Wikipedia-derived `Entity`/`FACT` records can support candidate identity hypotheses, while CMO-derived `Observation`, `Contact`, `Sensor`, `Emission`, `Platform`, and `PlatformClass` records provide timestamped operational evidence.

The observation ontology maps the Lua fields as follows:

- `Time` -> `Observation.time`
- `Sensor_aircraft` -> `Platform` connected by `(:Observation)-[:OBSERVED_BY]->(:Platform)`
- `Emission_sensor_name` -> `Sensor` and `Emission` connected by `DETECTS`, `EMITTED`, and `DETECTED_BY`; dynamic observation ingest uses `DETECTS` from the observing `Sensor_aircraft` platform to the observed emitter/sensor instead of asserting platform ownership with `HAS_SENSOR`
- `Emission_age`, `Emission_solid`, latitude, longitude, heading, altitude, and speed -> `Observation` kinematic/evidence properties
- `Emission_type` and `Emission_role` -> `Emission` properties
- `Emission_target_type` and `Emission_classificationlevel` -> `PlatformClass` plus `CLASSIFIED_AS` evidence
- log provenance -> `Source` connected by `DERIVED_FROM`

```bash
python -m combat_id_calibration ingest-cmo-observations \
  --input LuaHistory.txt \
  --observations-jsonl observations.jsonl \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password "$NEO4J_PASSWORD"
```

This follows the architecture seed sequence: CMO scenario output -> observation parser -> graph writer -> feature extractor/probability model. Downstream Cypher feature extraction can join observation evidence to the Wikipedia radar graph through sensor names, aliases, emitted signatures, radar families, platform classes, and other reference facts.

### Graph neighbourhood feature extraction

The `extract-features` command materializes the feature-extraction layer between Neo4j evidence queries and calibration. Provide one JSON object per contact-hypothesis-time combination, and the command writes deterministic numerical features such as `supporting_path_count`, `contradicting_path_count`, `mean_source_reliability`, `recency`, `shortest_path_to_platform_class`, `emission_match_score`, `kinematic_match_score`, and `contradiction_score`. These records are auditable inputs for logit construction; they are not calibrated probabilities.

```bash
python -m combat_id_calibration extract-features feature-requests.jsonl feature-records.jsonl \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password "$NEO4J_PASSWORD" \
  --include-logit
```

Example request:

```json
{"scenario_id":"raid-001","contact_id":"C-101","observation_time":"2026-06-15T10:00:00Z","hypothesis":"hostile_fighter"}
```

### Probability model and LLM explainer phases

The `probability-model` command implements the architecture seed's Probability Model phase for platform-level emitter identification. It consumes feature rows grouped by `scenario_id`, `contact_id`, and `observation_time`, where each row is one competing platform hypothesis such as `Su-27SM` or `MiG-29MT`. Rows may include a precomputed `feature_logit`, or the command will derive a baseline logit from the graph-neighbourhood feature columns. Candidate rows should also include `country_of_origin` (or `country`/`origin_country`) so the model can report both platform probabilities and marginal country probabilities, for example Belarus versus Kazakhstan.

```bash
python -m combat_id_calibration probability-model feature-records.jsonl probability-assignments.jsonl \
  --model calibration-model.json
```

Each output record preserves the full candidate distribution and adds `top_platform`, `top_platform_probability`, `top_country_of_origin`, `top_country_probability`, `platform_probabilities`, `country_probabilities`, and per-candidate evidence query references. If no calibration model is supplied, the command emits a deterministic uncalibrated softmax baseline for development only; operational runs should use a fitted calibrator.

The `explain` command implements the LLM Explainer preparation phase. It turns calibrated probability records into auditable explanation payloads and prompts that explicitly instruct the LLM to explain, not change, model probabilities.

```bash
python -m combat_id_calibration explain probability-assignments.jsonl explanation-payloads.jsonl
```

Explanation payloads include the calibrated platform and country distributions, confidence-limit warnings, supporting evidence, contradicting evidence, missing evidence, and recommended collection prompts. This keeps the LLM downstream of the calibrated probability model as required by `GraphDB_Probability_Architecture_Seed.txt`.


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
