import json

import pytest

from combat_id_calibration.graph_ingest import (
    ExtractedFact,
    SourceDocument,
    _neo4j_connection_error_message,
    _ollama_response_text,
    _validate_neo4j_credentials,
    _write_fact,
    build_extraction_prompt,
    chunk_text,
    extract_facts,
    create_neo4j_schema,
    parse_extracted_facts,
    stable_id,
)


def test_ollama_response_text_prefers_response_field():
    data = {"response": '{"facts": []}', "thinking": '{"facts": [{"subject":"ignored"}]}'}

    assert _ollama_response_text(data) == '{"facts": []}'


def test_ollama_response_text_falls_back_to_thinking_json():
    data = {
        "response": "",
        "thinking": '{"facts": [{"subject": "Zhuk-MS", "predicate": "HAS_EXPORT_DESIGNATION", "object": "Zhuk-MSE"}]}',
    }

    assert _ollama_response_text(data) == data["thinking"]

def test_chunk_text_uses_overlap():
    chunks = chunk_text("abcdefghij", max_chars=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("abc", max_chars=5, overlap=5)


def test_build_extraction_prompt_requests_broad_varied_context():
    document = SourceDocument(
        source_id=stable_id("pdf", "prompt"),
        source_type="pdf",
        locator="prompt.pdf",
        title="Prompt",
        text="",
    )
    prompt = build_extraction_prompt(document, "The MiG-29 uses N019 radar and R-27 missiles.")

    assert "combat-identification evidence graph" in prompt
    assert "Your entire response must be exactly one JSON object and nothing else." in prompt
    assert "Do not include Markdown fences, prose, comments, explanations, chain-of-thought" in prompt
    assert "If the chunk supports no facts, return exactly" in prompt
    assert "Aim for 8-20 diverse facts" in prompt
    assert "Cover different subjects" in prompt
    assert "TYPICAL_SPEED_KT" in prompt
    assert "MAX_SPEED_KT" in prompt
    assert "SERVICE_CEILING_M" in prompt
    assert "maximum aircraft performance values" in prompt
    assert "Emission_latitude" in prompt
    assert "DISTINGUISHES_FROM" in prompt
    assert "only extract current operating-force or operating-nation relationships" in prompt
    assert "current operator section" in prompt
    assert "former operator sections" in prompt
    assert "vehicle, vessel, or base system" in prompt
    assert "source explicitly describes the entity" in prompt
    assert "Zhuk should remain a radar/emitter" in prompt



def test_extract_facts_emits_chunk_and_parse_diagnostics(monkeypatch, capsys):
    document = SourceDocument(
        source_id=stable_id("pdf", "diagnostics"),
        source_type="pdf",
        locator="diagnostics.pdf",
        title="Diagnostics",
        text="The MiG-29 uses N019 radar.",
    )

    def fake_generate(prompt, model, ollama_url):
        assert "The MiG-29 uses N019 radar." in prompt
        return '{"facts": [{"subject": "MiG-29", "predicate": "HAS_SENSOR", "object": "N019 radar", "confidence": 0.9}]}'

    monkeypatch.setattr("combat_id_calibration.graph_ingest.ollama_generate", fake_generate)

    facts = extract_facts([document], model="test-model", max_chars=100, overlap=10)

    captured = capsys.readouterr()
    assert len(facts) == 1
    assert "starting fact extraction" in captured.err
    assert "document 1: title='Diagnostics'" in captured.err
    assert "received" in captured.err
    assert "JSON candidate 1 contains facts=1" in captured.err
    assert "finished extraction: documents=1, chunks=1, facts=1" in captured.err


def test_extract_facts_diagnostics_report_empty_source(capsys):
    document = SourceDocument(
        source_id=stable_id("pdf", "empty-diagnostics"),
        source_type="pdf",
        locator="empty.pdf",
        title="Empty",
        text="",
    )

    assert extract_facts([document], diagnostics=True) == []

    captured = capsys.readouterr()
    assert "chunks=0" in captured.err
    assert "produced no chunks" in captured.err


def test_parse_extracted_facts_normalizes_model_json():
    document = SourceDocument(
        source_id=stable_id("wikipedia", "https://example.test/wiki"),
        source_type="wikipedia",
        locator="https://example.test/wiki",
        title="Example",
        text="",
    )
    response = "```json\n" + json.dumps(
        {
            "facts": [
                {
                    "subject": "AN/APG-68 radar",
                    "predicate": "detects aircraft",
                    "object": "airborne targets",
                    "evidence": "The radar detects airborne targets.",
                    "confidence": 1.5,
                },
                {"subject": "", "predicate": "IS_A", "object": "ignored"},
            ]
        }
    ) + "\n```"
    facts = parse_extracted_facts(response, document)
    assert len(facts) == 1
    assert facts[0].predicate == "DETECTS_AIRCRAFT"
    assert facts[0].confidence == 1.0
    assert facts[0].source_id == document.source_id


def test_parse_extracted_facts_skips_reasoning_json_before_answer():
    document = SourceDocument(
        source_id=stable_id("pdf", "reasoning"),
        source_type="pdf",
        locator="reasoning.pdf",
        title="Reasoning",
        text="",
    )
    response = (
        '<think>The prompt schema looks like {"facts": []}, but the chunk supports facts.</think>\n'
        '{"facts": [{"subject": "MiG-29", "predicate": "HAS_SENSOR", '
        '"object": "N019 radar", "confidence": 0.9}]}'
    )

    facts = parse_extracted_facts(response, document)

    assert len(facts) == 1
    assert facts[0].subject == "MiG-29"
    assert facts[0].object == "N019 radar"


def test_parse_extracted_facts_uses_later_facts_object_after_embedded_json():
    document = SourceDocument(
        source_id=stable_id("pdf", "embedded"),
        source_type="pdf",
        locator="embedded.pdf",
        title="Embedded",
        text="",
    )
    response = (
        'I will use the requested schema {"example": "not the answer"}.\n'
        '{"facts": [{"subject": "R-27", "predicate": "IS_A", '
        '"object": "missile", "confidence": 1.0}]}'
    )

    facts = parse_extracted_facts(response, document)

    assert len(facts) == 1
    assert facts[0].subject == "R-27"
    assert facts[0].predicate == "IS_A"


def test_parse_extracted_facts_ignores_empty_model_response():
    document = SourceDocument(
        source_id=stable_id("pdf", "empty"),
        source_type="pdf",
        locator="empty.pdf",
        title="Empty",
        text="",
    )
    assert parse_extracted_facts("", document) == []


def test_parse_extracted_facts_ignores_non_json_model_response():
    document = SourceDocument(
        source_id=stable_id("pdf", "non-json"),
        source_type="pdf",
        locator="non-json.pdf",
        title="Non JSON",
        text="",
    )
    assert parse_extracted_facts("I could not find any supported facts.", document) == []


def test_parse_extracted_facts_ignores_malformed_json_model_response():
    document = SourceDocument(
        source_id=stable_id("pdf", "malformed"),
        source_type="pdf",
        locator="malformed.pdf",
        title="Malformed",
        text="",
    )
    assert parse_extracted_facts('{"facts": [', document) == []


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, statement, **parameters):
        self.calls.append((statement, parameters))


def test_create_neo4j_schema_uses_unique_constraints():
    session = RecordingRunner()
    create_neo4j_schema(session)
    statements = [call[0] for call in session.calls]
    assert statements[:2] == [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
    ]
    assert "CREATE CONSTRAINT platform_entity_id IF NOT EXISTS FOR (p:Platform) REQUIRE p.id IS UNIQUE" in statements
    assert "CREATE CONSTRAINT sensor_entity_id IF NOT EXISTS FOR (s:Sensor) REQUIRE s.id IS UNIQUE" in statements
    assert "CREATE CONSTRAINT country_entity_id IF NOT EXISTS FOR (c:Country) REQUIRE c.id IS UNIQUE" in statements


def test_write_fact_emits_parameterized_neo4j_cypher():
    tx = RecordingRunner()
    fact = ExtractedFact(
        subject="MiG-29",
        predicate="HAS_SENSOR",
        object="N019 radar",
        source_id="source-1",
        source_type="wikipedia",
        locator="https://example.test/wiki/MiG-29",
        evidence="MiG-29 uses the N019 radar.",
        confidence=0.9,
    )
    _write_fact(tx, fact)
    assert len(tx.calls) == 2
    statement, parameters = tx.calls[0]
    assert "MERGE (subject:Entity" in statement
    assert "CREATE (subject)-[:FACT" in statement
    assert "MERGE (object)-[:MENTIONED_IN]->(source)" in statement
    assert parameters["subject"] == "MiG-29"
    assert parameters["object"] == "N019 radar"
    assert parameters["predicate"] == "HAS_SENSOR"
    assert parameters["confidence"] == 0.9
    typed_statement, typed_parameters = tx.calls[1]
    assert "MERGE (subject)-[rel:HAS_SENSOR]->(object)" in typed_statement
    assert typed_parameters["object_is_sensor"] is True


def test_neo4j_connection_error_message_is_actionable():
    message = _neo4j_connection_error_message("bolt://localhost:7687", "neo4j")
    assert "Unable to connect to Neo4j" in message
    assert "bolt://localhost:7687" in message
    assert "docker run" in message
    assert "bolt://127.0.0.1:7687" in message


def test_validate_neo4j_credentials_rejects_missing_password():
    with pytest.raises(ValueError, match="Neo4j password is required"):
        _validate_neo4j_credentials("neo4j", "")

    with pytest.raises(ValueError, match="Neo4j password is required"):
        _validate_neo4j_credentials("neo4j", None)


def test_validate_neo4j_credentials_rejects_missing_user():
    with pytest.raises(ValueError, match="Neo4j username is required"):
        _validate_neo4j_credentials("", "secret")


def test_validate_neo4j_credentials_accepts_values():
    assert _validate_neo4j_credentials(" neo4j ", "secret") == ("neo4j", "secret")
