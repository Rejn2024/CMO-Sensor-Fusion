import json

from combat_id_calibration.cmo_observation_ingest import parse_observation_line
from combat_id_calibration.hypothesis_generation import (
    build_llm_hypothesis_prompt,
    emitter_aliases,
    emitter_semantic_tokens,
    fetch_graph_hypotheses_with_session,
    graph_hypothesis_query,
    probe_knowledge_graph_with_session,
    select_offline_hypotheses,
)


SAMPLE = "PY_CONTACT_LOG Time : 1844772240 , Sensor_aircraft : Typhoon FGR.4 , Emission_sensor_name : Slot Back [N-010 Zhuk-M] , Emission_altitude : 10316.349609375 , Emission_speed : 479.64691162109 , Emission_target_type : Type: Multirole (Fighter/Attack)"


class RecordingSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return self.responses.pop(0)


def test_emitter_aliases_extracts_full_bracketed_prefix_and_semantic_aliases():
    assert emitter_aliases("Slot Back [N-010 Zhuk-M]") == [
        "Slot Back [N-010 Zhuk-M]",
        "N-010 Zhuk-M",
        "Slot Back",
        "N010 Zhuk M",
        "N010 Zhuk",
        "Zhuk M",
        "Slot Back N010 Zhuk M",
        "Slot Back N010 Zhuk",
    ]
    assert emitter_semantic_tokens(["Slot Back [N-010 Zhuk-M]"]) == ["slot", "back", "n010", "zhuk"]


def test_graph_hypothesis_query_is_parameterized_and_requests_platform_evidence():
    query, params = graph_hypothesis_query(["N-010 Zhuk-M"], 10)

    assert "$emitter_aliases" in query
    assert "$limit" in query
    assert "Platform" in query
    assert "OPERATED_BY" in query
    assert "semantic_match_score" in query
    assert "matched_tokens" in query
    assert params == {
        "emitter_aliases": ["N-010 Zhuk-M"],
        "emitter_semantic_tokens": ["n010", "zhuk"],
        "minimum_semantic_token_matches": 2,
        "limit": 10,
    }


def test_fetch_graph_hypotheses_with_session_normalizes_rows_for_candidate_contract():
    observation = parse_observation_line(SAMPLE, source_line=1)
    session = RecordingSession(
        [
            [
                {
                    "hypothesis": "MiG-29SMT",
                    "operator_nation": "Russia",
                    "matched_aliases": ["N-010 Zhuk-M"],
                    "support_count": 3,
                    "evidence_paths": [["HAS_SENSOR"]],
                    "semantic_match_score": 0.0,
                }
            ]
        ]
    )

    rows = fetch_graph_hypotheses_with_session(session, observation, 10)

    assert "N010 Zhuk" in session.calls[0][1]["emitter_aliases"]
    assert session.calls[0][1]["emitter_semantic_tokens"] == ["slot", "back", "n010", "zhuk"]
    assert session.calls[0][1]["limit"] == 40
    assert rows == [
        {
            "hypothesis": "MiG-29SMT",
            "operator_nation": "Russia",
            "emitter_aliases": ["N-010 Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [0.0, 2500.0],
            "typical_altitude_m": [0.0, 25000.0],
            "kg_support_count": 3,
            "evidence_paths": [["HAS_SENSOR"]],
            "semantic_match_score": 0.0,
        }
    ]


def test_probe_knowledge_graph_with_session_reports_alias_node_and_neighbour_rows():
    observation = parse_observation_line(SAMPLE, source_line=1)
    session = RecordingSession(
        [
            [{"labels": ["Sensor"], "name": "N-010 Zhuk-M"}],
            [{"alias_node": "N-010 Zhuk-M", "relationship": "INSTALLED_ON", "neighbour_labels": ["Platform"], "neighbour": "MiG-29"}],
        ]
    )

    report = probe_knowledge_graph_with_session(session, observation)

    assert [probe["name"] for probe in report["probes"]] == ["alias_nodes", "alias_neighbours"]
    assert report["probes"][0]["rows"][0]["name"] == "N-010 Zhuk-M"
    assert report["probes"][1]["rows"][0]["neighbour"] == "MiG-29"


def test_select_offline_hypotheses_prefers_alias_class_kinematics_and_graph_support():
    observation = parse_observation_line(SAMPLE, source_line=1)
    candidates = [
        {"hypothesis": "Weak", "operator_nation": "Unknown", "emitter_aliases": [], "platform_class": "Other", "typical_speed_kt": [0, 100], "typical_altitude_m": [0, 100], "kg_support_count": 0},
        {"hypothesis": "MiG-29SMT", "operator_nation": "Russia", "emitter_aliases": ["N-010 Zhuk-M"], "platform_class": "Type: Multirole (Fighter/Attack)", "typical_speed_kt": [300, 900], "typical_altitude_m": [5000, 15000], "kg_support_count": 2},
    ]

    assert select_offline_hypotheses(observation, candidates, 1)[0]["hypothesis"] == "MiG-29SMT"


def test_build_llm_hypothesis_prompt_contains_json_contract_and_graph_rows():
    observation = parse_observation_line(SAMPLE, source_line=1)
    prompt = build_llm_hypothesis_prompt(observation, [{"hypothesis": "MiG-29SMT"}], 10)

    assert "Generate exactly 10" in prompt
    assert '"hypotheses"' in prompt
    assert "MiG-29SMT" in prompt
    json.loads(prompt.split("Observation: ", 1)[1].split("\n", 1)[0])
