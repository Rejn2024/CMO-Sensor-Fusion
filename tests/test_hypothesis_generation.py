import json

from combat_id_calibration.cmo_observation_ingest import parse_observation_line
from combat_id_calibration.hypothesis_generation import (
    build_llm_hypothesis_prompt,
    emitter_aliases,
    evidence_paths_query_id,
    emitter_semantic_tokens,
    fetch_graph_hypotheses_with_session,
    fetch_relationship_type_counts_with_session,
    graph_hypothesis_query,
    relationship_type_counts_query,
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
    assert emitter_semantic_tokens(["Slot Back [N-010 Zhuk-M]"]) == [
        "slot",
        "back",
        "n010",
        "zhuk",
    ]


def test_graph_hypothesis_query_is_parameterized_and_requests_platform_evidence():
    query, params = graph_hypothesis_query(["N-010 Zhuk-M"], 10)

    assert "$emitter_alias_match_terms" in query
    assert "$limit" in query
    assert "Platform" in query
    assert "Entity', 'CandidateIdentity" not in query
    assert "none(label IN labels(platform) WHERE label IN ['Country', 'Sensor', 'Operator', 'Location'])" in query
    assert "OPERATED_BY" not in query
    assert "OPERATOR_COUNTRY" not in query
    assert "VARIANT_OF" not in query
    assert "ASSIGNED_TO" not in query
    assert "[:OPERATED_BY" not in query
    assert "[:OPERATOR_COUNTRY" not in query
    assert "[:VARIANT_OF" not in query
    assert "[kinematic_fact:FACT]" not in query
    assert "type(operator_relationship) IN $operator_relationship_types" in query
    assert "type(kinematic_fact) = $kinematic_fact_relationship_type" in query
    assert "semantic_match_score" in query
    assert "matched_tokens" in query
    assert ".title" not in query
    assert "properties(emitter)['title']" in query
    assert "MAX_SPEED_KT" in query
    assert "SERVICE_CEILING_M" in query
    assert params == {
        "emitter_aliases": ["N-010 Zhuk-M"],
        "emitter_alias_match_terms": ["n 010 zhuk m"],
        "emitter_semantic_tokens": ["n010", "zhuk"],
        "minimum_reverse_alias_chars": 3,
        "minimum_semantic_token_matches": 2,
        "platform_relationship_types": [],
        "aircraft_variant_relationship_types": [
            "VARIANT_OF",
            "HAS_VARIANT",
            "AIRCRAFT_FAMILY",
        ],
        "emitter_variant_relationship_types": [
            "VARIANT_OF",
            "HAS_VARIANT",
            "ALSO_KNOWN_AS",
        ],
        "operator_relationship_types": [
            "OPERATED_BY",
            "OPERATOR",
            "USED_BY",
            "SERVICE_WITH",
            "ASSIGNED_TO",
        ],
        "operator_country_relationship_types": [
            "OPERATOR_COUNTRY",
            "HOME_BASE_COUNTRY",
        ],
        "operator_country_via_operator_relationship_types": [
            "OPERATOR_COUNTRY",
            "HOME_BASE_COUNTRY",
            "LOCATED_IN",
        ],
        "kinematic_fact_relationship_type": "FACT",
        "operator_aircraft_model_fact_predicates": [
            "OPERATED_BY_AIRCRAFT_MODEL",
            "AIRCRAFT_MODEL_OPERATED_BY",
        ],
        "limit": 10,
    }


def test_graph_hypothesis_query_falls_back_to_aircraft_variant_for_operator_country():
    query, params = graph_hypothesis_query(["N-010 Zhuk-M"], 10)

    assert (
        "OPTIONAL MATCH (aircraft_variant)-[aircraft_variant_operator_relationship]"
        "-(aircraft_variant_operator)"
    ) in query
    assert (
        "OPTIONAL MATCH (aircraft_variant)-[aircraft_variant_operator_country_relationship]"
        "-(aircraft_variant_operator_country)"
    ) in query
    assert (
        "OPTIONAL MATCH (aircraft_variant_operator)-"
        "[aircraft_variant_country_via_operator_relationship]"
        "-(aircraft_variant_country_via_operator)"
    ) in query
    assert "WHERE operator_country IS NULL" in query
    assert "AND operator_country_via_operator IS NULL" in query
    assert "aircraft_variant_operator_country.name" in query
    assert "aircraft_variant_country_via_operator.name" in query
    assert params["operator_relationship_types"] == [
        "OPERATED_BY",
        "OPERATOR",
        "USED_BY",
        "SERVICE_WITH",
        "ASSIGNED_TO",
    ]


def test_graph_hypothesis_query_uses_fact_predicates_for_operator_aircraft_model():
    query, params = graph_hypothesis_query(["N-010 Zhuk-M"], 10)

    assert "operator_aircraft_model_fact.predicate IN $operator_aircraft_model_fact_predicates" in query
    assert "aircraft_variant_operator_aircraft_model_fact.predicate IN $operator_aircraft_model_fact_predicates" in query
    assert "operator_fact_country.name" in query
    assert "aircraft_variant_operator_fact_country.name" in query
    assert params["operator_aircraft_model_fact_predicates"] == [
        "OPERATED_BY_AIRCRAFT_MODEL",
        "AIRCRAFT_MODEL_OPERATED_BY",
    ]


def test_graph_hypothesis_query_rejects_tiny_reverse_substring_alias_matches():
    query, params = graph_hypothesis_query(
        emitter_aliases("Slot Back [N-010 Zhuk-M]"), 10
    )

    assert "alias CONTAINS normalized_emitter_name" in query
    assert (
        "size(replace(normalized_emitter_name, ' ', '')) >= $minimum_reverse_alias_chars"
        in query
    )
    assert params["minimum_reverse_alias_chars"] == 3
    assert "n 010 zhuk m" in params["emitter_alias_match_terms"]


def test_select_offline_hypotheses_dampens_support_for_tiny_alias_false_positive():
    observation = parse_observation_line(SAMPLE, source_line=1)
    candidates = [
        {
            "hypothesis": "CAESAR",
            "operator_nation": "Unknown",
            "aircraft_variant": "CAESAR",
            "emitter_variant": "10",
            "emitter_aliases": ["10"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [0.0, 2500.0],
            "typical_altitude_m": [0.0, 25000.0],
            "kg_support_count": 50,
        },
        {
            "hypothesis": "MiG-29K/KUB",
            "operator_nation": "Unknown",
            "aircraft_variant": "MiG-29K/KUB",
            "emitter_variant": "Zhuk",
            "emitter_aliases": ["Zhuk"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [0.0, 2500.0],
            "typical_altitude_m": [0.0, 25000.0],
            "kg_support_count": 7,
        },
    ]

    assert (
        select_offline_hypotheses(observation, candidates, 1)[0]["hypothesis"]
        == "MiG-29K/KUB"
    )


def test_graph_hypothesis_query_can_limit_platform_path_relationship_types():
    query, params = graph_hypothesis_query(
        ["N-010 Zhuk-M"], ["HAS_SENSOR", "HAS_PLATFORM"], 10
    )

    assert "OPTIONAL MATCH platform_path = (platform)-[*1..4]-(emitter)" in query
    assert "type(rel) IN $platform_relationship_types" in query
    assert "HAS_SENSOR" not in query
    assert params["platform_relationship_types"] == ["HAS_SENSOR", "HAS_PLATFORM"]
    assert params["limit"] == 10


def test_graph_hypothesis_query_rejects_invalid_relationship_types():
    try:
        graph_hypothesis_query(["N-010 Zhuk-M"], ["HAS_SENSOR`) MATCH (n) //"], 10)
    except ValueError as error:
        assert "Invalid Neo4j relationship type" in str(error)
    else:
        raise AssertionError("expected invalid relationship type to be rejected")


def test_fetch_relationship_type_counts_with_session_runs_count_query():
    session = RecordingSession([[{"relationship_type": "HAS_SENSOR", "count": 2}]])

    rows = fetch_relationship_type_counts_with_session(session)

    assert rows == [{"relationship_type": "HAS_SENSOR", "count": 2}]
    assert "MATCH ()-[rel]->()" in session.calls[0][0]
    assert "ORDER BY count DESC" in relationship_type_counts_query()


def test_fetch_graph_hypotheses_with_session_normalizes_rows_for_candidate_contract():
    observation = parse_observation_line(SAMPLE, source_line=1)
    session = RecordingSession(
        [
            [
                {
                    "hypothesis": "MiG-29SMT",
                    "operator_nation": "Russia",
                    "aircraft_variant": "MiG-29SMT",
                    "emitter_variant": "N-010 Zhuk-ME",
                    "matched_aliases": ["N-010 Zhuk-M"],
                    "support_count": 3,
                    "evidence_paths": [["HAS_SENSOR"]],
                    "platform_labels": ["Platform"],
                    "aircraft_variant_labels": ["Platform"],
                    "operator_country_labels": ["Country"],
                    "semantic_match_score": 0.0,
                    "max_speed_kt_values": ["Mach 2.25 (1,320 kt)"],
                    "service_ceiling_m_values": ["57,400 ft (17,500 m)"],
                }
            ]
        ]
    )

    rows = fetch_graph_hypotheses_with_session(session, observation, 10)

    assert "N010 Zhuk" in session.calls[0][1]["emitter_aliases"]
    assert session.calls[0][1]["emitter_semantic_tokens"] == [
        "slot",
        "back",
        "n010",
        "zhuk",
    ]
    assert session.calls[0][1]["limit"] == 40
    assert rows == [
        {
            "hypothesis": "MiG-29SMT",
            "operator_nation": "Russia",
            "aircraft_variant": "MiG-29SMT",
            "emitter_variant": "N-010 Zhuk-ME",
            "emitter_aliases": ["N-010 Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [0.0, 1320.0],
            "typical_altitude_m": [0.0, 17500.0],
            "kg_support_count": 3,
            "evidence_paths": [["HAS_SENSOR"]],
            "semantic_match_score": 0.0,
        }
    ]


def test_fetch_graph_hypotheses_strips_operator_phrases_from_aircraft_variant_and_infers_country():
    observation = parse_observation_line(SAMPLE, source_line=1)
    session = RecordingSession(
        [
            [
                {
                    "hypothesis": "MiG-29K",
                    "operator_nation": "Unknown",
                    "aircraft_variant": "carrier-based variants for Indian Navy",
                    "emitter_variant": "N-010 Zhuk-ME",
                    "matched_aliases": ["N-010 Zhuk-M"],
                    "support_count": 3,
                    "evidence_paths": [["HAS_SENSOR", "OPERATOR_COUNTRY"]],
                    "platform_labels": ["Platform"],
                    "aircraft_variant_labels": ["Aircraft"],
                    "operator_country_labels": [],
                    "semantic_match_score": 10.0,
                },
                {
                    "hypothesis": "F-16",
                    "operator_nation": "Unknown",
                    "aircraft_variant": "Romanian Air Force (RoAF)",
                    "emitter_variant": "AN/APG-68",
                    "matched_aliases": ["AN/APG-68"],
                    "support_count": 2,
                    "evidence_paths": [["HAS_SENSOR", "OPERATED_BY"]],
                    "platform_labels": ["Platform"],
                    "aircraft_variant_labels": ["Aircraft"],
                    "operator_country_labels": [],
                    "semantic_match_score": 9.0,
                },
            ]
        ]
    )

    rows = fetch_graph_hypotheses_with_session(session, observation, 10)

    assert rows[0]["aircraft_variant"] == "MiG-29K"
    assert rows[0]["operator_nation"] == "India"
    assert rows[1]["aircraft_variant"] == "F-16"
    assert rows[1]["operator_nation"] == "Romania"


def test_fetch_graph_hypotheses_rejects_country_platform_and_sensor_operator_rows():
    observation = parse_observation_line(SAMPLE, source_line=1)
    session = RecordingSession(
        [
            [
                {
                    "hypothesis": "India",
                    "operator_nation": "Zhuk-ME",
                    "aircraft_variant": "India",
                    "emitter_variant": "N010 Zhuk",
                    "matched_aliases": ["N010 Zhuk"],
                    "support_count": 1,
                    "evidence_paths": [["USES_RADAR", "VARIANT_OF", "OPERATOR_COUNTRY"]],
                    "platform_labels": ["Entity", "Country"],
                    "aircraft_variant_labels": ["Country"],
                    "operator_labels": ["Sensor"],
                    "operator_country_labels": ["Sensor"],
                    "semantic_match_score": 42.0,
                },
                {
                    "hypothesis": "MiG-29SMT",
                    "operator_nation": "Russia",
                    "aircraft_variant": "MiG-29SMT",
                    "emitter_variant": "N-010 Zhuk-ME",
                    "matched_aliases": ["N-010 Zhuk-M"],
                    "support_count": 3,
                    "evidence_paths": [["HAS_SENSOR"]],
                    "platform_labels": ["Platform"],
                    "aircraft_variant_labels": ["Aircraft"],
                    "operator_country_labels": ["Country"],
                    "semantic_match_score": 10.0,
                },
            ]
        ]
    )

    rows = fetch_graph_hypotheses_with_session(session, observation, 10)

    assert [row["hypothesis"] for row in rows] == ["MiG-29SMT"]
    assert rows[0]["operator_nation"] == "Russia"


def test_evidence_paths_query_id_handles_nested_graph_paths_and_seed_strings():
    assert (
        evidence_paths_query_id([["HAS_SENSOR"], ["DETECTED_BY", "OPERATED_BY"]])
        == "HAS_SENSOR|DETECTED_BY>OPERATED_BY"
    )
    assert evidence_paths_query_id(["offline_seed"]) == "offline_seed"
    assert evidence_paths_query_id("") == ""


def test_probe_knowledge_graph_with_session_reports_alias_node_and_neighbour_rows():
    observation = parse_observation_line(SAMPLE, source_line=1)
    session = RecordingSession(
        [
            [{"labels": ["Sensor"], "name": "N-010 Zhuk-M"}],
            [
                {
                    "alias_node": "N-010 Zhuk-M",
                    "relationship": "INSTALLED_ON",
                    "neighbour_labels": ["Platform"],
                    "neighbour": "MiG-29",
                }
            ],
        ]
    )

    report = probe_knowledge_graph_with_session(session, observation)

    assert [probe["name"] for probe in report["probes"]] == [
        "alias_nodes",
        "alias_neighbours",
    ]
    probe_queries = [call[0] for call in session.calls]
    assert all(".title" not in query for query in probe_queries)
    assert all("['title']" in query for query in probe_queries)
    assert report["probes"][0]["rows"][0]["name"] == "N-010 Zhuk-M"
    assert report["probes"][1]["rows"][0]["neighbour"] == "MiG-29"


def test_select_offline_hypotheses_prefers_alias_class_kinematics_and_graph_support():
    observation = parse_observation_line(SAMPLE, source_line=1)
    candidates = [
        {
            "hypothesis": "Weak",
            "operator_nation": "Unknown",
            "emitter_aliases": [],
            "platform_class": "Other",
            "typical_speed_kt": [0, 100],
            "typical_altitude_m": [0, 100],
            "kg_support_count": 0,
        },
        {
            "hypothesis": "MiG-29SMT",
            "operator_nation": "Russia",
            "emitter_aliases": ["N-010 Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [300, 900],
            "typical_altitude_m": [5000, 15000],
            "kg_support_count": 2,
        },
    ]

    assert (
        select_offline_hypotheses(observation, candidates, 1)[0]["hypothesis"]
        == "MiG-29SMT"
    )


def test_select_offline_hypotheses_returns_unique_graph_combinations():
    observation = parse_observation_line(SAMPLE, source_line=1)
    candidates = [
        {
            "hypothesis": "MiG-29SMT",
            "operator_nation": "Russia",
            "aircraft_variant": "MiG-29SMT",
            "emitter_variant": "N-010 Zhuk-M",
            "emitter_aliases": ["N-010 Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [300, 900],
            "typical_altitude_m": [5000, 15000],
            "kg_support_count": 2,
        },
        {
            "hypothesis": "MiG-29SMT",
            "operator_nation": "Ukraine",
            "aircraft_variant": "MiG-29SMT",
            "emitter_variant": "N-010 Zhuk-M",
            "emitter_aliases": ["N-010 Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [300, 900],
            "typical_altitude_m": [5000, 15000],
            "kg_support_count": 1,
        },
        {
            "hypothesis": "MiG-35",
            "operator_nation": "Russia",
            "aircraft_variant": "MiG-35",
            "emitter_variant": "Zhuk-M",
            "emitter_aliases": ["Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [300, 900],
            "typical_altitude_m": [5000, 15000],
            "kg_support_count": 1,
        },
    ]

    hypotheses = select_offline_hypotheses(observation, candidates, 3)

    assert [item["hypothesis"] for item in hypotheses].count("MiG-29SMT") == 2
    assert [item["hypothesis"] for item in hypotheses].count("MiG-35") == 1
    assert hypotheses[0]["operator_nation"] == "Russia"
    assert {
        item["operator_nation"]
        for item in hypotheses
        if item["hypothesis"] == "MiG-29SMT"
    } == {"Russia", "Ukraine"}


def test_select_offline_hypotheses_deduplicates_identical_graph_combinations():
    observation = parse_observation_line(SAMPLE, source_line=1)
    candidates = [
        {
            "hypothesis": "MiG-29SMT",
            "operator_nation": "Russia",
            "aircraft_variant": "MiG-29SMT",
            "emitter_variant": "N-010 Zhuk-M",
            "emitter_aliases": ["N-010 Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [300, 900],
            "typical_altitude_m": [5000, 15000],
            "kg_support_count": 2,
        },
        {
            "hypothesis": "MiG-29SMT",
            "operator_nation": "Russia",
            "aircraft_variant": "MiG-29SMT",
            "emitter_variant": "N-010 Zhuk-M",
            "emitter_aliases": ["N-010 Zhuk-M"],
            "platform_class": "Type: Multirole (Fighter/Attack)",
            "typical_speed_kt": [300, 900],
            "typical_altitude_m": [5000, 15000],
            "kg_support_count": 1,
        },
    ]

    assert len(select_offline_hypotheses(observation, candidates, 3)) == 1


def test_build_llm_hypothesis_prompt_contains_json_contract_and_graph_rows():
    observation = parse_observation_line(SAMPLE, source_line=1)
    prompt = build_llm_hypothesis_prompt(observation, [{"hypothesis": "MiG-29SMT"}], 10)

    assert "Generate exactly 10" in prompt
    assert '"hypotheses"' in prompt
    assert "MiG-29SMT" in prompt
    json.loads(prompt.split("Observation: ", 1)[1].split("\n", 1)[0])
