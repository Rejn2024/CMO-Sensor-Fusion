"""Feature extraction from graph neighbourhoods for combat-ID hypotheses.

This module converts the local evidence neighbourhood around each
(contact, hypothesis, observation_time) tuple into deterministic numerical
features.  The features are intended to be auditable inputs to downstream
logit construction and calibration; they are not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .graph_ingest import _neo4j_connection_error_message, _validate_neo4j_credentials, stable_id
from .io import write_jsonl

FEATURE_SCHEMA = "graph_neighbourhood_features_v1"
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "supporting_path_count": 0.65,
    "contradicting_path_count": -0.75,
    "mean_source_reliability": 0.80,
    "recency": 0.50,
    "shortest_path_to_platform_class": -0.20,
    "emission_match_score": 1.10,
    "kinematic_match_score": 0.90,
    "contradiction_score": -1.20,
}


@dataclass(frozen=True)
class FeatureRequest:
    """One feature-extraction target from a CMO contact and candidate hypothesis."""

    scenario_id: str
    contact_id: str
    observation_time: str
    hypothesis: str


@dataclass(frozen=True)
class ContactHypothesisFeatures:
    """Numerical evidence features for one contact-hypothesis-time tuple."""

    scenario_id: str
    contact_id: str
    observation_time: str
    hypothesis: str
    supporting_path_count: float = 0.0
    contradicting_path_count: float = 0.0
    mean_source_reliability: float = 0.0
    recency: float = 0.0
    shortest_path_to_platform_class: float = 0.0
    emission_match_score: float = 0.0
    kinematic_match_score: float = 0.0
    contradiction_score: float = 0.0
    evidence_query_id: str = ""
    schema: str = FEATURE_SCHEMA

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        if not record["evidence_query_id"]:
            record["evidence_query_id"] = evidence_query_id(
                self.scenario_id, self.contact_id, self.observation_time, self.hypothesis
            )
        return record


def evidence_query_id(scenario_id: str, contact_id: str, observation_time: str, hypothesis: str) -> str:
    """Return a deterministic audit identifier for one feature query."""

    return stable_id("feature-query", scenario_id, contact_id, observation_time, hypothesis, FEATURE_SCHEMA)


def _bounded(value: object, *, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(numeric):
        return low
    return max(low, min(high, numeric))


def _non_negative(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) and numeric > 0 else 0.0


def features_from_mapping(request: FeatureRequest, values: Mapping[str, object]) -> ContactHypothesisFeatures:
    """Normalize a raw graph-query row into the stable feature contract."""

    shortest_path = values.get("shortest_path_to_platform_class", 0.0)
    return ContactHypothesisFeatures(
        scenario_id=request.scenario_id,
        contact_id=request.contact_id,
        observation_time=request.observation_time,
        hypothesis=request.hypothesis,
        supporting_path_count=_non_negative(values.get("supporting_path_count")),
        contradicting_path_count=_non_negative(values.get("contradicting_path_count")),
        mean_source_reliability=_bounded(values.get("mean_source_reliability")),
        recency=_bounded(values.get("recency")),
        shortest_path_to_platform_class=_non_negative(shortest_path),
        emission_match_score=_bounded(values.get("emission_match_score")),
        kinematic_match_score=_bounded(values.get("kinematic_match_score")),
        contradiction_score=_bounded(values.get("contradiction_score")),
        evidence_query_id=str(values.get("evidence_query_id") or evidence_query_id(
            request.scenario_id, request.contact_id, request.observation_time, request.hypothesis
        )),
    )


def feature_logit(features: ContactHypothesisFeatures, weights: Mapping[str, float] | None = None) -> float:
    """Build a deterministic baseline logit from extracted features."""

    selected = DEFAULT_FEATURE_WEIGHTS if weights is None else weights
    return sum(float(selected.get(name, 0.0)) * float(getattr(features, name)) for name in DEFAULT_FEATURE_WEIGHTS)


FEATURE_EXTRACTION_CYPHER = """
MATCH (contact:Contact {id: $contact_id})
OPTIONAL MATCH support_path = (contact)-[*1..4]-(hypothesis)
WHERE any(label IN labels(hypothesis) WHERE label IN ['Entity', 'PlatformClass', 'CandidateIdentity'])
  AND coalesce(properties(hypothesis)['name'], properties(hypothesis)['id']) = $hypothesis
  AND any(rel IN relationships(support_path) WHERE type(rel) IN ['SUPPORTS', 'FACT', 'CLASSIFIED_AS', 'EMITTED', 'HAS_EMITTER', 'EMITTED_BY', 'DETECTED_BY_PLATFORM'])
OPTIONAL MATCH contradict_path = (contact)-[*1..4]-(contradiction)
WHERE any(label IN labels(contradiction) WHERE label IN ['Entity', 'PlatformClass', 'CandidateIdentity'])
  AND any(rel IN relationships(contradict_path) WHERE type(rel) IN ['CONTRADICTS'] OR coalesce(properties(rel)['predicate'], '') STARTS WITH 'CONTRADICT')
OPTIONAL MATCH (contact)-[:HAS_OBSERVATION]->(obs:Observation)
OPTIONAL MATCH (obs)-[:DERIVED_FROM]->(source:Source)
OPTIONAL MATCH (contact)-[:EMITTED]->(emission:Emission)
WITH contact, support_path, contradict_path,
     properties(obs) AS obs_props,
     properties(source) AS source_props,
     properties(emission) AS emission_props
WITH contact,
     count(DISTINCT support_path) AS supporting_path_count,
     count(DISTINCT contradict_path) AS contradicting_path_count,
     min(length(support_path)) AS shortest_path_to_platform_class,
     avg(coalesce(source_props['reliability'], source_props['confidence'], 0.5)) AS mean_source_reliability,
     avg(CASE WHEN obs_props['age_seconds'] IS NULL THEN 0.5 ELSE 1.0 / (1.0 + toFloat(obs_props['age_seconds'])) END) AS recency,
     max(CASE WHEN toLower(coalesce(emission_props['sensor_name'], '')) CONTAINS toLower($hypothesis) THEN 1.0 ELSE 0.0 END) AS emission_match_score,
     avg(CASE WHEN obs_props['heading'] IS NOT NULL OR obs_props['speed'] IS NOT NULL OR obs_props['altitude'] IS NOT NULL THEN 0.5 ELSE 0.0 END) AS kinematic_match_score
RETURN supporting_path_count,
       contradicting_path_count,
       coalesce(mean_source_reliability, 0.0) AS mean_source_reliability,
       coalesce(recency, 0.0) AS recency,
       coalesce(shortest_path_to_platform_class, 0) AS shortest_path_to_platform_class,
       coalesce(emission_match_score, 0.0) AS emission_match_score,
       coalesce(kinematic_match_score, 0.0) AS kinematic_match_score,
       CASE WHEN supporting_path_count + contradicting_path_count = 0 THEN 0.0
            ELSE toFloat(contradicting_path_count) / toFloat(supporting_path_count + contradicting_path_count)
       END AS contradiction_score
"""


def extract_features_with_session(session: object, request: FeatureRequest) -> ContactHypothesisFeatures:
    """Run the feature Cypher against an existing Neo4j session-like object."""

    result = session.run(
        FEATURE_EXTRACTION_CYPHER,
        contact_id=request.contact_id,
        observation_time=request.observation_time,
        hypothesis=request.hypothesis,
    )
    row = result.single() if hasattr(result, "single") else next(iter(result), {})
    return features_from_mapping(request, dict(row or {}))


def read_feature_requests(path: str | Path) -> list[FeatureRequest]:
    """Read feature requests from JSONL records."""

    requests: list[FeatureRequest] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        missing = [key for key in ("scenario_id", "contact_id", "observation_time", "hypothesis") if not payload.get(key)]
        if missing:
            raise ValueError(f"feature request line {line_number} missing: {', '.join(missing)}")
        requests.append(FeatureRequest(*(str(payload[key]) for key in ("scenario_id", "contact_id", "observation_time", "hypothesis"))))
    if not requests:
        raise ValueError("feature request input contains no records")
    return requests


def extract_features_neo4j(requests: Sequence[FeatureRequest], uri: str, user: str, password: str, database: str | None = None) -> list[ContactHypothesisFeatures]:
    """Extract features for multiple requests from Neo4j."""

    user, password = _validate_neo4j_credentials(user, password)
    neo4j = importlib.import_module("neo4j")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        try:
            driver.verify_connectivity()
            session_kwargs = {"database": database} if database else {}
            with driver.session(**session_kwargs) as session:
                return [extract_features_with_session(session, request) for request in requests]
        except neo4j.exceptions.ServiceUnavailable as error:
            raise RuntimeError(_neo4j_connection_error_message(uri, database)) from error
        except neo4j.exceptions.AuthError as error:
            raise RuntimeError(f"Neo4j rejected the credentials for {uri!r}.") from error
    finally:
        driver.close()


def add_feature_extraction_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser("extract-features", help="convert graph neighbourhoods into numerical contact-hypothesis features")
    parser.add_argument("input", help="JSONL feature requests with scenario_id, contact_id, observation_time, hypothesis")
    parser.add_argument("output", help="JSONL output with extracted feature records")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--neo4j-database", help="optional Neo4j database name")
    parser.add_argument("--include-logit", action="store_true", help="append a baseline uncalibrated feature_logit")
    parser.set_defaults(handler=run_feature_extraction_command)


def run_feature_extraction_command(args: argparse.Namespace) -> None:
    requests = read_feature_requests(args.input)
    features = extract_features_neo4j(requests, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
    records = [item.to_record() for item in features]
    if args.include_logit:
        for record, item in zip(records, features):
            record["feature_logit"] = feature_logit(item)
    write_jsonl(records, args.output)
    print(json.dumps({"input": args.input, "output": args.output, "features": len(records)}, indent=2))
