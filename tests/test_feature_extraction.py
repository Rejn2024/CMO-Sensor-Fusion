import json

import pytest

from combat_id_calibration.feature_extraction import (
    FEATURE_EXTRACTION_CYPHER,
    ContactHypothesisFeatures,
    FeatureRequest,
    evidence_query_id,
    extract_features_with_session,
    feature_logit,
    features_from_mapping,
    read_feature_requests,
)


def test_features_from_mapping_normalizes_expected_feature_contract():
    request = FeatureRequest("raid-001", "C-101", "2026-06-15T10:00:00Z", "hostile_fighter")

    features = features_from_mapping(
        request,
        {
            "supporting_path_count": 3,
            "contradicting_path_count": "1",
            "mean_source_reliability": 1.5,
            "recency": 0.25,
            "shortest_path_to_platform_class": 2,
            "emission_match_score": 0.75,
            "kinematic_match_score": None,
            "contradiction_score": -1,
        },
    )

    assert features.supporting_path_count == 3.0
    assert features.contradicting_path_count == 1.0
    assert features.mean_source_reliability == 1.0
    assert features.recency == 0.25
    assert features.shortest_path_to_platform_class == 2.0
    assert features.emission_match_score == 0.75
    assert features.kinematic_match_score == 0.0
    assert features.contradiction_score == 0.0
    assert features.evidence_query_id == evidence_query_id("raid-001", "C-101", "2026-06-15T10:00:00Z", "hostile_fighter")


def test_contact_hypothesis_features_to_record_adds_schema_and_query_id():
    features = ContactHypothesisFeatures(
        scenario_id="raid-001",
        contact_id="C-101",
        observation_time="2026-06-15T10:00:00Z",
        hypothesis="hostile_fighter",
        supporting_path_count=2,
    )

    record = features.to_record()

    assert record["schema"] == "graph_neighbourhood_features_v1"
    assert record["evidence_query_id"] == evidence_query_id("raid-001", "C-101", "2026-06-15T10:00:00Z", "hostile_fighter")
    assert record["supporting_path_count"] == 2


def test_feature_logit_uses_signed_feature_weights():
    features = ContactHypothesisFeatures(
        scenario_id="s",
        contact_id="c",
        observation_time="t",
        hypothesis="h",
        supporting_path_count=2,
        contradicting_path_count=1,
        emission_match_score=1,
        contradiction_score=0.5,
    )

    assert feature_logit(features) == pytest.approx(1.05)


class FakeResult:
    def single(self):
        return {
            "supporting_path_count": 2,
            "contradicting_path_count": 1,
            "mean_source_reliability": 0.7,
            "recency": 0.9,
            "shortest_path_to_platform_class": 3,
            "emission_match_score": 1.0,
            "kinematic_match_score": 0.5,
            "contradiction_score": 1 / 3,
        }


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, statement, **parameters):
        self.calls.append((statement, parameters))
        return FakeResult()


def test_extract_features_with_session_runs_neighbourhood_query():
    session = FakeSession()
    request = FeatureRequest("scenario", "contact-1", "2026-06-15T10:00:00Z", "MiG-29")

    features = extract_features_with_session(session, request)

    statement, parameters = session.calls[0]
    assert statement == FEATURE_EXTRACTION_CYPHER
    assert parameters == {
        "contact_id": "contact-1",
        "observation_time": "2026-06-15T10:00:00Z",
        "hypothesis": "MiG-29",
    }
    assert features.supporting_path_count == 2
    assert features.emission_match_score == 1.0


def test_feature_extraction_cypher_avoids_schema_specific_warning_patterns():
    assert ":CandidateIdentity" not in FEATURE_EXTRACTION_CYPHER
    assert "source.reliability" not in FEATURE_EXTRACTION_CYPHER
    assert "source.confidence" not in FEATURE_EXTRACTION_CYPHER
    assert "rel.predicate" not in FEATURE_EXTRACTION_CYPHER
    assert "properties(source) AS source_props" in FEATURE_EXTRACTION_CYPHER
    assert "WITH contact, support_path, contradict_path," in FEATURE_EXTRACTION_CYPHER
    assert "labels(hypothesis)" in FEATURE_EXTRACTION_CYPHER


def test_read_feature_requests_validates_jsonl(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps({"scenario_id": "s", "contact_id": "c", "observation_time": "t", "hypothesis": "h"}) + "\n")

    assert read_feature_requests(path) == [FeatureRequest("s", "c", "t", "h")]


def test_read_feature_requests_rejects_missing_required_field(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps({"scenario_id": "s", "contact_id": "c", "observation_time": "t"}) + "\n")

    with pytest.raises(ValueError, match="hypothesis"):
        read_feature_requests(path)
