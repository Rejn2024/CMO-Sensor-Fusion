import json
from pathlib import Path

import extract_cmo_jsonl_from_log as scraper


def test_extract_records_from_prefixed_lua_log_lines():
    lines = [
        "2026-06-18 12:00:00 Lua> harmless status message\n",
        '2026-06-18 12:00:01 Lua> {"export_schema":"cmo_combat_id_v1","source":"cmo_lua","name":"Bogey {One}","wrapper_snapshot":{"nested":true}} trailing text\n',
        'not a CMO export {"other":"json"}\n',
    ]

    records = list(scraper.extract_records_from_lines(lines))

    assert len(records) == 1
    assert records[0][0] == 2
    assert records[0][1]["name"] == "Bogey {One}"
    assert records[0][1]["wrapper_snapshot"] == {"nested": True}


def test_extract_log_file_writes_jsonl_and_can_deduplicate(tmp_path: Path):
    record = {"export_schema": "cmo_combat_id_v1", "source": "cmo_lua", "guid": "TRACK-A"}
    log_path = tmp_path / "LuaHistory.txt"
    output_path = tmp_path / "recovered.jsonl"
    log_path.write_text(
        f"prefix {json.dumps(record)}\nagain {json.dumps(record)}\n",
        encoding="utf-8",
    )

    count = scraper.extract_log_file(log_path, output_path, unique=True)

    assert count == 1
    assert [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()] == [record]
