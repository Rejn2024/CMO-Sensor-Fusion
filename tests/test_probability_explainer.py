import json
import subprocess
import sys

import pytest

from combat_id_calibration.graph_ingest import _COUNTRY_NAMES
from combat_id_calibration.llm_explainer import build_explanation_payload
from combat_id_calibration.probability_model import (
    _MIG29_SU27_OPERATOR_NATIONS,
    _KNOWN_OPERATOR_NATION_BORDER_POLYGONS,
    OPERATOR_NATION_CENTROIDS,
    PROBABILITY_SCHEMA,
    group_feature_rows,
    border_distance_km,
    haversine_distance_km,
    operator_nation_distance_evidence,
    platform_operator_nation_distribution,
)


def candidate_rows():
    return [
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "Su-27SM",
            "operator_nation": "Belarus",
            "supporting_path_count": 3,
            "emission_match_score": 1,
            "evidence_query_id": "q1",
        },
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "MiG-29MT",
            "operator_nation": "Kazakhstan",
            "supporting_path_count": 1,
            "contradicting_path_count": 1,
            "emission_match_score": 0.3,
            "evidence_query_id": "q2",
        },
    ]



def test_operator_nation_centroids_cover_country_names():
    missing = {country_name for country_name in _COUNTRY_NAMES if country_name.casefold() not in OPERATOR_NATION_CENTROIDS}

    assert missing == set()
    assert OPERATOR_NATION_CENTROIDS["belarus"] == pytest.approx((53.7098, 27.9534))


def test_mig29_su27_operator_nations_have_border_polygons():
    missing = _MIG29_SU27_OPERATOR_NATIONS - set(_KNOWN_OPERATOR_NATION_BORDER_POLYGONS)

    assert missing == set()
    assert "china" in _KNOWN_OPERATOR_NATION_BORDER_POLYGONS

def test_haversine_distance_km_measures_great_circle_distance():
    assert haversine_distance_km(53.7098, 27.9534, 53.7098, 27.9534) == pytest.approx(0.0)
    assert haversine_distance_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.195, abs=0.01)


def test_border_distance_km_measures_nearest_border_segment():
    square_border = ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0))

    assert border_distance_km(1.0, 1.0, square_border) == pytest.approx(111.18, abs=0.01)
    assert border_distance_km(0.0, 1.0, square_border) == pytest.approx(0.0)


def test_operator_nation_distance_evidence_uses_emitter_observation_coordinates():
    evidence = operator_nation_distance_evidence(
        {"emission_latitude": 53.7, "emission_longitude": 28.0},
        "Belarus",
    )

    assert evidence is not None
    assert evidence["operator_nation_distance_reference"] == "border"
    assert evidence["operator_nation_distance_km"] == pytest.approx(137.44, abs=0.01)
    assert evidence["operator_nation_distance_score"] == pytest.approx(0.9465, abs=0.001)


def test_operator_nation_distance_evidence_prefers_row_supplied_border_coordinates_over_centroid():
    evidence = operator_nation_distance_evidence(
        {
            "emission_latitude": 0.0,
            "emission_longitude": 1.0,
            "operator_nation_latitude": 30.0,
            "operator_nation_longitude": 30.0,
            "operator_nation_border_coordinates": [[0.0, 0.0], [0.0, 2.0]],
        },
        "Nation With No Built In Border",
    )

    assert evidence is not None
    assert evidence["operator_nation_distance_reference"] == "border"
    assert evidence["operator_nation_distance_km"] == pytest.approx(0.0)
    assert evidence["operator_nation_distance_score"] == pytest.approx(1.0)


def test_platform_operator_nation_distribution_adjusts_logits_with_distance_evidence():
    rows = [
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "near-platform",
            "operator_nation": "Belarus",
            "feature_logit": 0.0,
            "emission_latitude": 53.7,
            "emission_longitude": 28.0,
            "evidence_query_id": "near",
        },
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "far-platform",
            "operator_nation": "United States",
            "feature_logit": 0.0,
            "emission_latitude": 53.7,
            "emission_longitude": 28.0,
            "evidence_query_id": "far",
        },
    ]

    result = platform_operator_nation_distribution(rows)

    assert result["top_operator_nation"] == "Belarus"
    assert result["operator_nation_probabilities"]["Belarus"] > result["operator_nation_probabilities"]["United States"]
    near_candidate = next(candidate for candidate in result["candidates"] if candidate["evidence_query_id"] == "near")
    far_candidate = next(candidate for candidate in result["candidates"] if candidate["evidence_query_id"] == "far")
    assert near_candidate["logit"] > far_candidate["logit"]
    assert near_candidate["operator_nation_distance_km"] < far_candidate["operator_nation_distance_km"]



def test_platform_operator_nation_distribution_reads_coordinates_from_features_dictionary():
    rows = [
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "near-platform",
            "operator_nation": "Belarus",
            "feature_logit": 0.0,
            "features": {"latitude": 53.7, "longitude": 28.0},
            "evidence_query_id": "near",
        },
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "far-platform",
            "operator_nation": "United States",
            "feature_logit": 0.0,
            "features": {"lattitude": 53.7, "lon": 28.0},
            "evidence_query_id": "far",
        },
    ]

    result = platform_operator_nation_distribution(rows)

    near_candidate = next(candidate for candidate in result["candidates"] if candidate["evidence_query_id"] == "near")
    far_candidate = next(candidate for candidate in result["candidates"] if candidate["evidence_query_id"] == "far")
    assert near_candidate["emitter_latitude"] == pytest.approx(53.7)
    assert near_candidate["emitter_longitude"] == pytest.approx(28.0)
    assert far_candidate["emitter_latitude"] == pytest.approx(53.7)
    assert far_candidate["emitter_longitude"] == pytest.approx(28.0)
    assert near_candidate["operator_nation_distance_km"] < far_candidate["operator_nation_distance_km"]

def test_platform_operator_nation_distribution_assigns_platform_and_origin_probabilities():
    result = platform_operator_nation_distribution(candidate_rows())

    assert result["schema"] == PROBABILITY_SCHEMA
    assert result["top_platform"] == "Su-27SM"
    assert result["top_operator_nation"] == "Belarus"
    assert sum(result["platform_probabilities"].values()) == pytest.approx(1.0)
    assert result["operator_nation_probabilities"]["Belarus"] == result["platform_probabilities"]["Su-27SM"]
    assert result["candidates"][0]["evidence_query_id"] == "q1"


def test_platform_operator_nation_distribution_allows_repeated_platform_names():
    rows = candidate_rows() + [
        {
            **candidate_rows()[0],
            "operator_nation": "Russia",
            "supporting_path_count": 2,
            "evidence_query_id": "q3",
        }
    ]

    result = platform_operator_nation_distribution(rows)

    assert len(result["candidates"]) == 3
    assert {candidate["candidate_index"] for candidate in result["candidates"]} == {0, 1, 2}
    assert sum(candidate["probability"] for candidate in result["candidates"]) == pytest.approx(1.0)
    assert result["platform_probabilities"]["Su-27SM"] == pytest.approx(
        sum(candidate["probability"] for candidate in result["candidates"] if candidate["platform"] == "Su-27SM")
    )
    assert result["operator_nation_probabilities"]["Russia"] == pytest.approx(
        next(candidate["probability"] for candidate in result["candidates"] if candidate["evidence_query_id"] == "q3")
    )


def test_platform_operator_nation_distribution_top_platform_uses_marginal_distribution():
    rows = [
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "HighSingle",
            "operator_nation": "Nation A",
            "feature_logit": 3.0,
            "evidence_query_id": "single",
        },
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "Repeated",
            "operator_nation": "Nation B",
            "feature_logit": 2.9,
            "evidence_query_id": "repeat-1",
        },
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "Repeated",
            "operator_nation": "Nation C",
            "feature_logit": 2.9,
            "evidence_query_id": "repeat-2",
        },
    ]

    result = platform_operator_nation_distribution(rows)

    assert result["candidates"][0]["platform"] == "HighSingle"
    assert result["platform_probabilities"]["Repeated"] > result["platform_probabilities"]["HighSingle"]
    assert result["top_platform"] == "Repeated"
    assert result["top_platform_probability"] == pytest.approx(result["platform_probabilities"]["Repeated"])


def test_operator_nation_top_identification_skips_unknown_placeholder():
    rows = [
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "MiG-29",
            "operator_nation": "Unknown",
            "feature_logit": 3.0,
            "evidence_query_id": "unknown",
        },
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "MiG-29",
            "operator_nation": "Russia",
            "feature_logit": 2.9,
            "evidence_query_id": "russia",
        },
    ]

    result = platform_operator_nation_distribution(rows)
    payload = build_explanation_payload(result)

    assert result["operator_nation_probabilities"]["Unknown"] > result["operator_nation_probabilities"]["Russia"]
    assert result["top_operator_nation"] == "Russia"
    assert "Russia" in payload["summary"]
    assert "Unknown" not in payload["summary"]
    assert "Unknown" not in payload["llm_prompt"]


def test_group_feature_rows_groups_by_contact_time():
    rows = candidate_rows() + [{**candidate_rows()[0], "contact_id": "c2"}]

    groups = group_feature_rows(rows)

    assert [len(group) for group in groups] == [2, 1]


def test_llm_explainer_preserves_probabilities_and_builds_grounded_prompt():
    probability = platform_operator_nation_distribution(candidate_rows())
    payload = build_explanation_payload(
        probability,
        {
            "supporting_evidence": [{"text": "N001-family emission matched Su-27SM references", "source": "graph:q1"}],
            "contradicting_evidence": ["MiG-29MT path has a contradictory range feature"],
            "missing_evidence": ["Collect another emitter scan"],
        },
    )

    assert "Su-27SM" in payload["summary"]
    assert "Platform distribution" in payload["llm_prompt"]
    assert "without changing" in payload["llm_prompt"]
    assert "N001-family" in payload["llm_prompt"]


def test_probability_model_and_explainer_cli(tmp_path):
    features = tmp_path / "features.jsonl"
    probabilities = tmp_path / "probabilities.jsonl"
    explanations = tmp_path / "explanations.jsonl"
    features.write_text("".join(json.dumps(row) + "\n" for row in candidate_rows()))

    subprocess.run([sys.executable, "-m", "combat_id_calibration", "probability-model", str(features), str(probabilities)], check=True)
    subprocess.run([sys.executable, "-m", "combat_id_calibration", "explain", str(probabilities), str(explanations)], check=True)

    assert json.loads(probabilities.read_text().splitlines()[0])["top_platform"] == "Su-27SM"
    assert "llm_prompt" in json.loads(explanations.read_text().splitlines()[0])
