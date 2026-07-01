"""Probability model utilities for platform and operator-nation attribution.

The graph feature extractor emits one row per (contact, hypothesis, time).  This
module groups those rows into candidate sets, converts feature rows into logits,
and applies a fitted calibrator so downstream code receives calibrated platform
and operator-nation probabilities rather than LLM-generated estimates.
When emitter latitude/longitude are available, candidate logits are also nudged
by the great-circle distance between the emitter and the hypothesized operator
country centroid (or row-supplied operator-country coordinates).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .calibrator import TemperatureCalibrator, _softmax
from .feature_extraction import ContactHypothesisFeatures, feature_logit
from .io import write_jsonl

PROBABILITY_SCHEMA = "platform_operator_nation_probability_v1"
UNKNOWN_OPERATOR_NATION_LABELS = {"", "unknown", "unk", "n/a", "na", "none", "null", "not specified", "unspecified"}
EARTH_RADIUS_KM = 6371.0088
OPERATOR_NATION_DISTANCE_SCALE_KM = 2500.0
OPERATOR_NATION_DISTANCE_LOGIT_WEIGHT = 1.0
OPERATOR_NATION_CENTROIDS: dict[str, tuple[float, float]] = {
    "belarus": (53.7098, 27.9534),
    "china": (35.8617, 104.1954),
    "france": (46.2276, 2.2137),
    "germany": (51.1657, 10.4515),
    "india": (20.5937, 78.9629),
    "iran": (32.4279, 53.6880),
    "israel": (31.0461, 34.8516),
    "kazakhstan": (48.0196, 66.9237),
    "north korea": (40.3399, 127.5101),
    "pakistan": (30.3753, 69.3451),
    "poland": (51.9194, 19.1451),
    "russia": (61.5240, 105.3188),
    "russian federation": (61.5240, 105.3188),
    "south korea": (35.9078, 127.7669),
    "syria": (34.8021, 38.9968),
    "turkey": (38.9637, 35.2433),
    "ukraine": (48.3794, 31.1656),
    "united kingdom": (55.3781, -3.4360),
    "uk": (55.3781, -3.4360),
    "united states": (37.0902, -95.7129),
    "usa": (37.0902, -95.7129),
}


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


def _optional_float(record: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _country_centroid(operator_nation: str) -> tuple[float, float] | None:
    return OPERATOR_NATION_CENTROIDS.get(operator_nation.strip().casefold())


def haversine_distance_km(
    origin_latitude: float, origin_longitude: float, target_latitude: float, target_longitude: float
) -> float:
    """Return great-circle distance in kilometers between two latitude/longitude pairs."""

    origin_latitude_rad = math.radians(origin_latitude)
    target_latitude_rad = math.radians(target_latitude)
    latitude_delta = math.radians(target_latitude - origin_latitude)
    longitude_delta = math.radians(target_longitude - origin_longitude)
    a = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(origin_latitude_rad) * math.cos(target_latitude_rad) * math.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def operator_nation_distance_evidence(record: Mapping[str, object], operator_nation: str) -> dict[str, float] | None:
    """Return optional emitter-to-operator-country distance evidence for a candidate row."""

    emitter_latitude = _optional_float(record, "emitter_latitude", "emission_latitude", "latitude")
    emitter_longitude = _optional_float(record, "emitter_longitude", "emission_longitude", "longitude")
    if emitter_latitude is None or emitter_longitude is None:
        return None
    country_latitude = _optional_float(
        record, "operator_nation_latitude", "operator_country_latitude", "country_latitude"
    )
    country_longitude = _optional_float(
        record, "operator_nation_longitude", "operator_country_longitude", "country_longitude"
    )
    if country_latitude is None or country_longitude is None:
        centroid = _country_centroid(operator_nation)
        if centroid is None:
            return None
        country_latitude, country_longitude = centroid
    distance_km = haversine_distance_km(emitter_latitude, emitter_longitude, country_latitude, country_longitude)
    proximity_score = math.exp(-distance_km / OPERATOR_NATION_DISTANCE_SCALE_KM)
    return {
        "emitter_latitude": emitter_latitude,
        "emitter_longitude": emitter_longitude,
        "operator_nation_latitude": country_latitude,
        "operator_nation_longitude": country_longitude,
        "operator_nation_distance_km": distance_km,
        "operator_nation_distance_score": proximity_score,
        "operator_nation_distance_logit_adjustment": OPERATOR_NATION_DISTANCE_LOGIT_WEIGHT * proximity_score,
    }


def _is_unknown_operator_nation(label: object) -> bool:
    """Return true when an operator-nation label is a non-identifying placeholder."""

    return str(label or "").strip().casefold() in UNKNOWN_OPERATOR_NATION_LABELS


def _top_positive_operator_nation(operator_nation_distribution: Mapping[str, float]) -> tuple[str, float]:
    """Select the highest-probability positive operator-nation identification."""

    for operator_nation, probability in operator_nation_distribution.items():
        if not _is_unknown_operator_nation(operator_nation):
            return operator_nation, probability
    raise ValueError("probability model requires at least one positive operator-nation candidate")


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


def platform_operator_nation_distribution(
    rows: Sequence[Mapping[str, object]], calibrator: TemperatureCalibrator | None = None
) -> dict[str, object]:
    """Produce calibrated platform probabilities and marginal operator-nation probabilities.

    ``rows`` must describe competing platform-level hypotheses for one contact at
    one observation time.  Candidate platform names are read from ``hypothesis``
    (or ``platform``/``platform_type``), and may repeat when rows differ by
    emitter, operator, or other evidence.  Operator nation labels are read from
    ``operator_nation``/``operator_country``/``nation`` with legacy fallback to
    ``country``/``country_of_origin``/``origin_country``.
    """

    if not rows:
        raise ValueError("probability model requires at least one candidate row")
    classes = [_text(row, "hypothesis", "platform", "platform_type") for row in rows]
    if any(not item for item in classes):
        raise ValueError("candidate rows must include platform hypothesis names")
    operator_nations = [
        _text(
            row,
            "operator_nation",
            "operator_country",
            "nation",
            "country",
            "country_of_origin",
            "origin_country",
            default="unknown",
        )
        for row in rows
    ]
    base_logits = [feature_row_to_logit(row) for row in rows]
    distance_evidence = [
        operator_nation_distance_evidence(row, operator_nation) for row, operator_nation in zip(rows, operator_nations)
    ]
    logits = [
        base_logit + (evidence["operator_nation_distance_logit_adjustment"] if evidence else 0.0)
        for base_logit, evidence in zip(base_logits, distance_evidence)
    ]
    if calibrator is None:
        candidate_probabilities = _softmax(logits, 1.0)
        calibration = {"method": "softmax_uncalibrated_baseline"}
    else:
        if (
            len(calibrator.classes) == len(classes)
            and len(set(classes)) == len(classes)
            and list(calibrator.classes) != classes
        ):
            raise ValueError("calibrator classes must match grouped platform hypotheses in order")
        candidate_probabilities = _softmax(logits, calibrator.temperature)
        calibration = calibrator.to_dict()

    platform_probs: dict[str, float] = defaultdict(float)
    operator_nation_probs: dict[str, float] = defaultdict(float)
    candidates: list[dict[str, object]] = []
    for candidate_index, (row, platform, operator_nation, base_logit, logit, evidence, probability) in enumerate(
        zip(rows, classes, operator_nations, base_logits, logits, distance_evidence, candidate_probabilities)
    ):
        probability = float(probability)
        platform_probs[platform] += probability
        operator_nation_probs[operator_nation] += probability
        candidate = {
            "candidate_index": candidate_index,
            "platform": platform,
            "operator_nation": operator_nation,
            "probability": probability,
            "logit": logit,
            "base_logit": base_logit,
            "evidence_query_id": _text(row, "evidence_query_id"),
        }
        if evidence:
            candidate.update(evidence)
        candidates.append(candidate)
    candidates.sort(key=lambda item: float(item["probability"]), reverse=True)
    platform_distribution = dict(sorted(platform_probs.items(), key=lambda item: item[1], reverse=True))
    operator_nation_distribution = dict(sorted(operator_nation_probs.items(), key=lambda item: item[1], reverse=True))
    top_operator_nation, top_operator_nation_probability = _top_positive_operator_nation(operator_nation_distribution)
    first = rows[0]
    return {
        "schema": PROBABILITY_SCHEMA,
        "scenario_id": _text(first, "scenario_id"),
        "contact_id": _text(first, "contact_id"),
        "observation_time": _text(first, "observation_time"),
        "top_platform": next(iter(platform_distribution)),
        "top_platform_probability": next(iter(platform_distribution.values())),
        "top_operator_nation": top_operator_nation,
        "top_operator_nation_probability": top_operator_nation_probability,
        "platform_probabilities": platform_distribution,
        "operator_nation_probabilities": operator_nation_distribution,
        "candidates": candidates,
        "calibration": calibration,
    }


# Backwards-compatible alias for callers that have not yet adopted operator-nation terminology.
platform_country_distribution = platform_operator_nation_distribution

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
    records = [platform_operator_nation_distribution(group, calibrator) for group in group_feature_rows(read_jsonl(input_path))]
    write_jsonl(records, output_path)
    return records


def add_probability_model_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser("probability-model", help="assign calibrated platform and operator-nation probabilities from feature rows")
    parser.add_argument("input", help="JSONL feature rows with one row per candidate platform hypothesis")
    parser.add_argument("output", help="JSONL probability assignments grouped by contact/time")
    parser.add_argument("--model", help="optional temperature calibrator JSON whose classes match each candidate group")
    parser.set_defaults(handler=run_probability_model_command)


def run_probability_model_command(args: argparse.Namespace) -> None:
    records = run_probability_model(args.input, args.output, args.model)
    print(json.dumps({"input": args.input, "output": args.output, "assignments": len(records)}, indent=2))
