import json

import pytest

from combat_id_calibration.graph_ingest import SourceDocument, chunk_text, parse_extracted_facts, stable_id


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
