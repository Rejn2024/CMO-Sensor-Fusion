import json

import pytest

from combat_id_calibration.graph_ingest import (
    ExtractedFact,
    SourceDocument,
    _neo4j_connection_error_message,
    _write_fact,
    chunk_text,
    create_neo4j_schema,
    parse_extracted_facts,
    stable_id,
)


def test_chunk_text_uses_overlap():
    chunks = chunk_text("abcdefghij", max_chars=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("abc", max_chars=5, overlap=5)


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
    assert statements == [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
    ]


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
    assert len(tx.calls) == 1
    statement, parameters = tx.calls[0]
    assert "MERGE (subject:Entity" in statement
    assert "CREATE (subject)-[:FACT" in statement
    assert "MERGE (object)-[:MENTIONED_IN]->(source)" in statement
    assert parameters["subject"] == "MiG-29"
    assert parameters["object"] == "N019 radar"
    assert parameters["predicate"] == "HAS_SENSOR"
    assert parameters["confidence"] == 0.9


def test_neo4j_connection_error_message_is_actionable():
    message = _neo4j_connection_error_message("bolt://localhost:7687", "neo4j")
    assert "Unable to connect to Neo4j" in message
    assert "bolt://localhost:7687" in message
    assert "docker run" in message
    assert "bolt://127.0.0.1:7687" in message

