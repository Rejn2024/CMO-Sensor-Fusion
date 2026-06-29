import json
import subprocess
import sys

import pytest

from combat_id_calibration.llm_explainer import build_explanation_payload
from combat_id_calibration.probability_model import (
    PROBABILITY_SCHEMA,
    group_feature_rows,
    platform_country_distribution,
)


def candidate_rows():
    return [
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "Su-27SM",
            "country_of_origin": "Belarus",
            "supporting_path_count": 3,
            "emission_match_score": 1,
            "evidence_query_id": "q1",
        },
        {
            "scenario_id": "s",
            "contact_id": "c",
            "observation_time": "t",
            "hypothesis": "MiG-29MT",
            "country_of_origin": "Kazakhstan",
            "supporting_path_count": 1,
            "contradicting_path_count": 1,
            "emission_match_score": 0.3,
            "evidence_query_id": "q2",
        },
    ]


def test_platform_country_distribution_assigns_platform_and_origin_probabilities():
    result = platform_country_distribution(candidate_rows())

    assert result["schema"] == PROBABILITY_SCHEMA
    assert result["top_platform"] == "Su-27SM"
    assert result["top_country_of_origin"] == "Belarus"
    assert sum(result["platform_probabilities"].values()) == pytest.approx(1.0)
    assert result["country_probabilities"]["Belarus"] == result["platform_probabilities"]["Su-27SM"]
    assert result["candidates"][0]["evidence_query_id"] == "q1"


def test_group_feature_rows_groups_by_contact_time():
    rows = candidate_rows() + [{**candidate_rows()[0], "contact_id": "c2"}]

    groups = group_feature_rows(rows)

    assert [len(group) for group in groups] == [2, 1]


def test_llm_explainer_preserves_probabilities_and_builds_grounded_prompt():
    probability = platform_country_distribution(candidate_rows())
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
