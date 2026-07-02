"""Probability model utilities for platform and operator-nation attribution.

The graph feature extractor emits one row per (contact, hypothesis, time).  This
module groups those rows into candidate sets, converts feature rows into logits,
and applies a fitted calibrator so downstream code receives calibrated platform
and operator-nation probabilities rather than LLM-generated estimates.
When emitter latitude/longitude are available, candidate logits are also nudged
by the great-circle distance between the emitter and the nearest modeled border
point for the hypothesized operator country (or row-supplied operator-country
border coordinates).
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
from .graph_ingest import _COUNTRY_NAMES
from .io import write_jsonl

PROBABILITY_SCHEMA = "platform_operator_nation_probability_v1"
UNKNOWN_OPERATOR_NATION_LABELS = {"", "unknown", "unk", "n/a", "na", "none", "null", "not specified", "unspecified"}
EARTH_RADIUS_KM = 6371.0088
OPERATOR_NATION_DISTANCE_SCALE_KM = 2000.0
OPERATOR_NATION_DISTANCE_LOGIT_WEIGHT = 1.0
_MIG29_SU27_OPERATOR_NATIONS: frozenset[str] = frozenset(
    {
        "algeria",
        "angola",
        "azerbaijan",
        "bangladesh",
        "belarus",
        "bulgaria",
        "china",
        "cuba",
        "czech republic",
        "czechia",
        "czechoslovakia",
        "democratic people's republic of korea",
        "east germany",
        "egypt",
        "eritrea",
        "ethiopia",
        "germany",
        "hungary",
        "india",
        "indonesia",
        "iran",
        "iraq",
        "israel",
        "kazakhstan",
        "libya",
        "malaysia",
        "moldova",
        "mongolia",
        "myanmar",
        "north korea",
        "peru",
        "poland",
        "romania",
        "russia",
        "russian federation",
        "serbia",
        "slovakia",
        "south yemen",
        "soviet union",
        "sudan",
        "syria",
        "turkmenistan",
        "ukraine",
        "united states",
        "united states of america",
        "usa",
        "uzbekistan",
        "venezuela",
        "vietnam",
        "yemen",
        "yugoslavia",
    }
)

_KNOWN_OPERATOR_NATION_BORDER_POLYGONS: dict[str, tuple[tuple[float, float], ...]] = {
    # Lightweight, dependency-free border approximations for known operators of
    # the MiG-29 and Su-27 families (including major variants such as China's
    # Su-27-derived J-11). Coordinates are ordered as (latitude, longitude)
    # vertices and intentionally favor broad national extents over centroids so
    # nearby border activity receives stronger evidence than activity near the
    # geographic center.
    "algeria": ((18.97, -8.67), (18.97, 11.99), (37.09, 11.99), (37.09, -8.67)),
    "angola": ((-18.04, 11.67), (-18.04, 24.09), (-4.38, 24.09), (-4.38, 11.67)),
    "azerbaijan": ((38.39, 44.77), (38.39, 50.37), (41.91, 50.37), (41.91, 44.77)),
    "bangladesh": ((20.74, 88.03), (20.74, 92.67), (26.63, 92.67), (26.63, 88.03)),
    "belarus": ((51.25, 23.18), (52.15, 31.78), (56.17, 28.15), (55.85, 23.50)),
    "bulgaria": ((41.24, 22.36), (41.24, 28.61), (44.22, 28.61), (44.22, 22.36)),
    "china": ((18.16, 73.50), (18.16, 134.77), (53.56, 134.77), (53.56, 73.50)),
    "cuba": ((19.83, -84.95), (19.83, -74.13), (23.27, -74.13), (23.27, -84.95)),
    "czech republic": ((48.55, 12.09), (48.55, 18.86), (51.06, 18.86), (51.06, 12.09)),
    "czechia": ((48.55, 12.09), (48.55, 18.86), (51.06, 18.86), (51.06, 12.09)),
    "czechoslovakia": ((47.73, 12.09), (47.73, 22.57), (51.06, 22.57), (51.06, 12.09)),
    "east germany": ((50.20, 9.92), (50.20, 15.04), (54.98, 15.04), (54.98, 9.92)),
    "egypt": ((22.00, 24.70), (22.00, 36.90), (31.67, 36.90), (31.67, 24.70)),
    "eritrea": ((12.35, 36.43), (12.35, 43.13), (18.02, 43.13), (18.02, 36.43)),
    "ethiopia": ((3.40, 32.99), (3.40, 47.99), (14.89, 47.99), (14.89, 32.99)),
    "germany": ((47.27, 5.87), (47.27, 15.04), (55.06, 15.04), (55.06, 5.87)),
    "hungary": ((45.74, 16.11), (45.74, 22.90), (48.59, 22.90), (48.59, 16.11)),
    "india": ((6.75, 68.11), (6.75, 97.40), (35.67, 97.40), (35.67, 68.11)),
    "indonesia": ((-11.01, 95.01), (-11.01, 141.02), (6.08, 141.02), (6.08, 95.01)),
    "iran": ((25.06, 44.03), (25.06, 63.33), (39.78, 63.33), (39.78, 44.03)),
    "iraq": ((29.06, 38.79), (29.06, 48.58), (37.38, 48.58), (37.38, 38.79)),
    "israel": ((29.45, 34.27), (29.45, 35.90), (33.34, 35.90), (33.34, 34.27)),
    "kazakhstan": ((40.56, 46.49), (45.00, 87.32), (55.44, 84.95), (55.38, 46.49)),
    "libya": ((19.50, 9.32), (19.50, 25.15), (33.17, 25.15), (33.17, 9.32)),
    "malaysia": ((0.85, 99.64), (0.85, 119.27), (7.36, 119.27), (7.36, 99.64)),
    "moldova": ((45.47, 26.62), (45.47, 30.16), (48.49, 30.16), (48.49, 26.62)),
    "mongolia": ((41.58, 87.75), (41.58, 119.93), (52.15, 119.93), (52.15, 87.75)),
    "myanmar": ((9.78, 92.19), (9.78, 101.17), (28.55, 101.17), (28.55, 92.19)),
    "north korea": ((37.67, 124.18), (37.67, 130.67), (43.01, 130.67), (43.01, 124.18)),
    "democratic people's republic of korea": ((37.67, 124.18), (37.67, 130.67), (43.01, 130.67), (43.01, 124.18)),
    "peru": ((-18.35, -81.33), (-18.35, -68.65), (-0.04, -68.65), (-0.04, -81.33)),
    "poland": ((49.00, 14.12), (49.00, 24.15), (54.84, 24.15), (54.84, 14.12)),
    "romania": ((43.62, 20.26), (43.62, 29.70), (48.27, 29.70), (48.27, 20.26)),
    "russia": ((41.19, 19.64), (41.19, 180.00), (81.86, 180.00), (81.86, 19.64)),
    "russian federation": ((41.19, 19.64), (41.19, 180.00), (81.86, 180.00), (81.86, 19.64)),
    "serbia": ((42.23, 18.84), (42.23, 23.01), (46.19, 23.01), (46.19, 18.84)),
    "slovakia": ((47.73, 16.83), (47.73, 22.57), (49.61, 22.57), (49.61, 16.83)),
    "south yemen": ((12.11, 42.55), (12.11, 54.54), (18.99, 54.54), (18.99, 42.55)),
    "soviet union": ((35.14, 19.64), (35.14, 180.00), (81.86, 180.00), (81.86, 19.64)),
    "sudan": ((8.68, 21.81), (8.68, 38.61), (22.23, 38.61), (22.23, 21.81)),
    "syria": ((32.31, 35.70), (32.31, 42.38), (37.32, 42.38), (37.32, 35.70)),
    "turkmenistan": ((35.13, 52.44), (35.13, 66.71), (42.80, 66.71), (42.80, 52.44)),
    "ukraine": ((44.39, 22.14), (45.35, 40.23), (52.38, 40.23), (52.38, 22.14)),
    "united states": ((24.52, -124.77), (24.52, -66.95), (49.38, -66.95), (49.38, -124.77)),
    "united states of america": ((24.52, -124.77), (24.52, -66.95), (49.38, -66.95), (49.38, -124.77)),
    "usa": ((24.52, -124.77), (24.52, -66.95), (49.38, -66.95), (49.38, -124.77)),
    "uzbekistan": ((37.18, 55.99), (37.18, 73.13), (45.59, 73.13), (45.59, 55.99)),
    "venezuela": ((0.65, -73.35), (0.65, -59.80), (12.20, -59.80), (12.20, -73.35)),
    "vietnam": ((8.18, 102.14), (8.18, 109.46), (23.39, 109.46), (23.39, 102.14)),
    "yemen": ((12.11, 42.55), (12.11, 54.54), (18.99, 54.54), (18.99, 42.55)),
    "yugoslavia": ((40.85, 13.49), (40.85, 23.04), (46.88, 23.04), (46.88, 13.49)),
}


_KNOWN_OPERATOR_NATION_CENTROIDS: dict[str, tuple[float, float]] = {
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
    "united kingdom of great britain and northern ireland": (55.3781, -3.4360),
    "uk": (55.3781, -3.4360),
    "united states": (37.0902, -95.7129),
    "united states of america": (37.0902, -95.7129),
    "usa": (37.0902, -95.7129),
}
OPERATOR_NATION_CENTROIDS: dict[str, tuple[float, float]] = {
    country_name.casefold(): _KNOWN_OPERATOR_NATION_CENTROIDS.get(country_name.casefold(), (0.0, 0.0))
    for country_name in _COUNTRY_NAMES
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
    """Return the first finite float found in top-level or nested feature fields."""

    records: list[Mapping[str, object]] = [record]
    features = record.get("features")
    if isinstance(features, Mapping):
        records.append(features)
    for source in records:
        for key in keys:
            value = source.get(key)
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


def _country_border_polygon(operator_nation: str) -> tuple[tuple[float, float], ...] | None:
    return _KNOWN_OPERATOR_NATION_BORDER_POLYGONS.get(operator_nation.strip().casefold())


def _parse_border_polygon(record: Mapping[str, object]) -> tuple[tuple[float, float], ...] | None:
    for key in (
        "operator_nation_border_coordinates",
        "operator_country_border_coordinates",
        "country_border_coordinates",
    ):
        value = record.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        vertices: list[tuple[float, float]] = []
        if not isinstance(value, Iterable):
            continue
        for item in value:
            if isinstance(item, Mapping):
                latitude = _optional_float(item, "latitude", "lat")
                longitude = _optional_float(item, "longitude", "lon", "lng")
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
                try:
                    latitude = float(item[0])
                    longitude = float(item[1])
                except (TypeError, ValueError):
                    continue
            else:
                continue
            if latitude is not None and longitude is not None:
                vertices.append((latitude, longitude))
        if len(vertices) >= 2:
            return tuple(vertices)
    return None


def _point_to_segment_distance_km(
    latitude: float, longitude: float, start: tuple[float, float], end: tuple[float, float]
) -> tuple[float, float, float]:
    reference_latitude = math.radians(latitude)
    kilometers_per_degree_latitude = math.pi * EARTH_RADIUS_KM / 180.0
    kilometers_per_degree_longitude = kilometers_per_degree_latitude * math.cos(reference_latitude)
    px = longitude * kilometers_per_degree_longitude
    py = latitude * kilometers_per_degree_latitude
    ax = start[1] * kilometers_per_degree_longitude
    ay = start[0] * kilometers_per_degree_latitude
    bx = end[1] * kilometers_per_degree_longitude
    by = end[0] * kilometers_per_degree_latitude
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return haversine_distance_km(latitude, longitude, start[0], start[1]), start[0], start[1]
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest_latitude = (ay + t * dy) / kilometers_per_degree_latitude
    nearest_longitude = (ax + t * dx) / kilometers_per_degree_longitude
    return (
        haversine_distance_km(latitude, longitude, nearest_latitude, nearest_longitude),
        nearest_latitude,
        nearest_longitude,
    )


def border_distance_km(latitude: float, longitude: float, polygon: Sequence[tuple[float, float]]) -> float:
    """Return the approximate distance from a point to the nearest country-border segment."""

    distance_km, _, _ = _nearest_border_point(latitude, longitude, polygon)
    return distance_km


def _nearest_border_point(
    latitude: float, longitude: float, polygon: Sequence[tuple[float, float]]
) -> tuple[float, float, float]:
    """Return distance and nearest approximated border point for a country border polygon."""

    if len(polygon) < 2:
        raise ValueError("border polygon requires at least two vertices")
    vertices = list(polygon)
    segments = zip(vertices, vertices[1:] + [vertices[0]]) if len(vertices) > 2 else [(vertices[0], vertices[1])]
    return min(_point_to_segment_distance_km(latitude, longitude, start, end) for start, end in segments)


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


def operator_nation_distance_evidence(record: Mapping[str, object], operator_nation: str) -> dict[str, object] | None:
    """Return optional emitter-to-nearest-operator-country-border distance evidence for a candidate row."""

    emitter_latitude = _optional_float(record, "emitter_latitude", "emission_latitude", "latitude", "lat", "lattitude", "observed_latitude")
    emitter_longitude = _optional_float(record, "emitter_longitude", "emission_longitude", "longitude", "lon", "lng", "observed_longitude")
    if emitter_latitude is None or emitter_longitude is None:
        return None
    border_polygon = _parse_border_polygon(record) or _country_border_polygon(operator_nation)
    country_latitude = _optional_float(
        record, "operator_nation_latitude", "operator_country_latitude", "country_latitude"
    )
    country_longitude = _optional_float(
        record, "operator_nation_longitude", "operator_country_longitude", "country_longitude"
    )
    if border_polygon:
        distance_km, reference_latitude, reference_longitude = _nearest_border_point(
            emitter_latitude, emitter_longitude, border_polygon
        )
        reference_type = "border"
    else:
        if country_latitude is None or country_longitude is None:
            centroid = _country_centroid(operator_nation)
            if centroid is None:
                return None
            country_latitude, country_longitude = centroid
        distance_km = haversine_distance_km(emitter_latitude, emitter_longitude, country_latitude, country_longitude)
        reference_latitude = country_latitude
        reference_longitude = country_longitude
        reference_type = "centroid"
    proximity_score = math.exp(-distance_km / OPERATOR_NATION_DISTANCE_SCALE_KM)
    return {
        "emitter_latitude": emitter_latitude,
        "emitter_longitude": emitter_longitude,
        "operator_nation_latitude": reference_latitude,
        "operator_nation_longitude": reference_longitude,
        "operator_nation_distance_reference": reference_type,
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

    print(f'operator_nations : {operator_nations}')
    # print(f'rows: {rows}')
    distance_evidence = [
        operator_nation_distance_evidence(row, operator_nation) for row, operator_nation in zip(rows, operator_nations)
    ]
    print(f'distance_evidence : {distance_evidence}')
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
