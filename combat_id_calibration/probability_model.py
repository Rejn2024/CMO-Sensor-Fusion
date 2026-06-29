"""Probability model utilities for platform and origin attribution.

The graph feature extractor emits one row per (contact, hypothesis, time).  This
module groups those rows into candidate sets, converts feature rows into logits,
and applies a fitted calibrator so downstream code receives calibrated platform
and country-of-origin probabilities rather than LLM-generated estimates.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .calibrator import TemperatureCalibrator, _softmax
from .feature_extraction import ContactHypothesisFeatures, feature_logit
from .io import write_jsonl

PROBABILITY_SCHEMA = "platform_origin_probability_v1"


def _text(record: Mapping[str, object], *keys: str, default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _float(record: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default))
    except (TypeError, ValueError):
        return default


def feature_row_to_logit(record: Mapping[str, object]) -> float:
    """Return a model logit from either an explicit feature_logit or feature columns."""

    if "feature_logit" in record:
        return _float(record, "feature_logit")
    features = ContactHypothesisFeatures(
        scenario_id=_text(record, "scenario_id"),
        contact_id=_text(record, "contact_id"),
        observation_time=_text(record, "observation_time"),
        hypothesis=_text(record, "hypothesis", "platform", "platform_type"),
        supporting_path_count=_float(record, "supporting_path_count"),
        contradicting_path_count=_float(record, "contradicting_path_count"),
        mean_source_reliability=_float(record, "mean_source_reliability"),
        recency=_float(record, "recency"),
        shortest_path_to_platform_class=_float(record, "shortest_path_to_platform_class"),
        emission_match_score=_float(record, "emission_match_score"),
        kinematic_match_score=_float(record, "kinematic_match_score"),
        contradiction_score=_float(record, "contradiction_score"),
    )
    return feature_logit(features)


def platform_country_distribution(
    rows: Sequence[Mapping[str, object]], calibrator: TemperatureCalibrator | None = None
) -> dict[str, object]:
    """Produce calibrated platform probabilities and marginal country probabilities.

    ``rows`` must describe competing platform-level hypotheses for one contact at
    one observation time.  Candidate names are read from ``hypothesis`` (or
    ``platform``/``platform_type``), and country labels are read from
    ``country``/``country_of_origin``/``origin_country``.
    """

    if not rows:
        raise ValueError("probability model requires at least one candidate row")
    classes = [_text(row, "hypothesis", "platform", "platform_type") for row in rows]
    if any(not item for item in classes) or len(set(classes)) != len(classes):
        raise ValueError("candidate rows must have unique platform hypothesis names")
    logits = [feature_row_to_logit(row) for row in rows]
    if calibrator is None:
        platform_probs = dict(zip(classes, _softmax(logits, 1.0)))
        calibration = {"method": "softmax_uncalibrated_baseline"}
    else:
        if list(calibrator.classes) != classes:
            raise ValueError("calibrator classes must match grouped platform hypotheses in order")
        platform_probs = calibrator.calibrate(logits)
        calibration = calibrator.to_dict()

    country_probs: dict[str, float] = defaultdict(float)
    candidates: list[dict[str, object]] = []
    for row, platform in zip(rows, classes):
        country = _text(row, "country", "country_of_origin", "origin_country", default="unknown")
        probability = float(platform_probs[platform])
        country_probs[country] += probability
        candidates.append(
            {
                "platform": platform,
                "country_of_origin": country,
                "probability": probability,
                "logit": feature_row_to_logit(row),
                "evidence_query_id": _text(row, "evidence_query_id"),
            }
        )
    candidates.sort(key=lambda item: float(item["probability"]), reverse=True)
    country_distribution = dict(sorted(country_probs.items(), key=lambda item: item[1], reverse=True))
    first = rows[0]
    return {
        "schema": PROBABILITY_SCHEMA,
        "scenario_id": _text(first, "scenario_id"),
        "contact_id": _text(first, "contact_id"),
        "observation_time": _text(first, "observation_time"),
        "top_platform": candidates[0]["platform"],
        "top_platform_probability": candidates[0]["probability"],
        "top_country_of_origin": next(iter(country_distribution)),
        "top_country_probability": next(iter(country_distribution.values())),
        "platform_probabilities": {str(item["platform"]): item["probability"] for item in candidates},
        "country_probabilities": country_distribution,
        "candidates": candidates,
        "calibration": calibration,
    }


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def group_feature_rows(rows: Iterable[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (_text(row, "scenario_id"), _text(row, "contact_id"), _text(row, "observation_time"))
        groups[key].append(row)
    return [groups[key] for key in sorted(groups)]


def run_probability_model(input_path: str | Path, output_path: str | Path, model_path: str | Path | None = None) -> list[dict[str, object]]:
    calibrator = None
    if model_path:
        calibrator = TemperatureCalibrator.from_dict(json.loads(Path(model_path).read_text(encoding="utf-8")))
    records = [platform_country_distribution(group, calibrator) for group in group_feature_rows(read_jsonl(input_path))]
    write_jsonl(records, output_path)
    return records


def add_probability_model_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser("probability-model", help="assign calibrated platform and country probabilities from feature rows")
    parser.add_argument("input", help="JSONL feature rows with one row per candidate platform hypothesis")
    parser.add_argument("output", help="JSONL probability assignments grouped by contact/time")
    parser.add_argument("--model", help="optional temperature calibrator JSON whose classes match each candidate group")
    parser.set_defaults(handler=run_probability_model_command)


def run_probability_model_command(args: argparse.Namespace) -> None:
    records = run_probability_model(args.input, args.output, args.model)
    print(json.dumps({"input": args.input, "output": args.output, "assignments": len(records)}, indent=2))
