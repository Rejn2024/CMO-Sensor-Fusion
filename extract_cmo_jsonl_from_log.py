#!/usr/bin/env python3
"""Extract CMO combat-ID JSONL records from a Lua console text log.

CMO installations may mirror Lua console ``print`` output into ``.txt`` files
under their Logs directory. ``cmo_scenario_export.lua`` can print each generated
JSONL object to the console; this helper recovers those JSON objects even when a
log line contains a timestamp or other text before/after the JSON payload.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

EXPORT_SCHEMA = "cmo_combat_id_v1"
EXPORT_SOURCE = "cmo_lua"


def _json_object_candidates(text: str) -> Iterator[str]:
    """Yield balanced JSON-object substrings found in a single log line."""
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]
                start = None


def extract_records_from_lines(lines: Iterable[str]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield CMO combat-ID JSON objects from log lines with their source line."""
    for line_number, line in enumerate(lines, start=1):
        for candidate in _json_object_candidates(line):
            try:
                record = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("export_schema") == EXPORT_SCHEMA or record.get("source") == EXPORT_SOURCE:
                yield line_number, record


def extract_log_file(input_path: Path, output_path: Path, *, unique: bool = False) -> int:
    """Write recovered JSONL records from ``input_path`` to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0

    with input_path.open("r", encoding="utf-8-sig", errors="replace") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for _, record in extract_records_from_lines(source):
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if unique and line in seen:
                continue
            seen.add(line)
            target.write(line + "\n")
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CMO Logs .txt file containing Lua console output")
    parser.add_argument("--output", required=True, help="JSONL file to write recovered CMO export records")
    parser.add_argument("--unique", action="store_true", help="Drop duplicate JSON records while preserving first occurrence")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    count = extract_log_file(Path(args.input), Path(args.output), unique=args.unique)
    print(json.dumps({"input": args.input, "output": args.output, "records": count}, indent=2, sort_keys=True))
    return count


if __name__ == "__main__":
    main()
