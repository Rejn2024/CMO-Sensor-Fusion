"""JSONL interchange for CMO/Graph DB candidate scores and ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_examples(path: str | Path, require_truth: bool = True) -> tuple[list[str], list[list[float]], list[int], list[dict[str, object]]]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("input contains no records")
    classes = sorted(records[0]["scores"])
    logits, labels = [], []
    for record in records:
        if sorted(record.get("scores", {})) != classes:
            raise ValueError("all records must have the same candidate classes")
        logits.append([float(record["scores"][name]) for name in classes])
        if require_truth:
            truth = record.get("truth")
            if truth not in classes:
                raise ValueError("each training/evaluation record needs truth matching a candidate")
            labels.append(classes.index(truth))
    return classes, logits, labels, records


def write_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
