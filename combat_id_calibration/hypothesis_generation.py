"""Generate and probe knowledge-graph hypotheses for CMO emissions.

This module contains the notebook hypothesis-generation helpers as reusable,
testable code.  The probe functions intentionally return diagnostics even when
no candidates are found so Neo4j-backed notebooks can explain why a query would
otherwise produce ``0 candidate hypotheses``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .cmo_observation_ingest import EmissionObservation
from .graph_ingest import (
    _COUNTRY_NAMES,
    _neo4j_connection_error_message,
    _validate_neo4j_credentials,
)

_RELATIONSHIP_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _normalized_relationship_types(relationship_types: Sequence[str] | None) -> list[str]:
    """Return deduplicated, vetted Neo4j relationship type names."""

    normalized = []
    for relationship_type in relationship_types or []:
        value = str(relationship_type).strip().upper()
        if not _RELATIONSHIP_TYPE_RE.fullmatch(value):
            raise ValueError(f"Invalid Neo4j relationship type: {relationship_type!r}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _relationship_pattern(relationship_types: Sequence[str] | None) -> str:
    """Return a Cypher variable-length relationship pattern for vetted types."""

    normalized = _normalized_relationship_types(relationship_types)
    if not normalized:
        return "[*1..4]"
    return f"[:{'|'.join(normalized)}*1..4]"


def emitter_aliases(sensor_name: str) -> list[str]:
    """Return normalized aliases for a CMO emission sensor string.

    CMO sensor labels often mix NATO reporting names with bracketed model
    designators, punctuation, and variant suffixes.  The graph may store a
    related sensor under a nearby canonical name (for example ``N010 Zhuk``
    rather than ``Slot Back [N-010 Zhuk-M]``), so expose both exact label
    fragments and semantic search forms that preserve the meaningful model and
    family tokens.
    """

    text = sensor_name.strip()
    aliases: list[str] = []
    if text:
        aliases.append(text)
        if "[" in text and "]" in text:
            bracketed = text.split("[", 1)[1].split("]", 1)[0].strip()
            prefix = text.split("[", 1)[0].strip()
            aliases.extend(part.strip() for part in (bracketed, prefix) if part.strip())
            aliases.extend(_semantic_alias_variants(bracketed))
        aliases.extend(_semantic_alias_variants(text))
    deduped: list[str] = []
    for alias in aliases:
        if alias and alias.lower() not in {existing.lower() for existing in deduped}:
            deduped.append(alias)
    return deduped


def emitter_semantic_tokens(aliases: Sequence[str]) -> list[str]:
    """Return stable tokens used for semantic emitter-name matching."""

    tokens: list[str] = []
    for alias in aliases:
        for token in re.findall(r"[a-z0-9]+", alias.lower().replace("n-", "n")):
            if len(token) > 1 and token not in tokens:
                tokens.append(token)
    return tokens


def _semantic_alias_variants(text: str) -> list[str]:
    normalized = re.sub(r"(?i)\bn[-\s]?(\d+)", r"N\1", text)
    normalized = re.sub(r"[\[\]]", " ", normalized)
    compact = re.sub(r"[-_/]+", " ", normalized)
    compact = re.sub(r"\s+", " ", compact).strip()
    variants = [compact] if compact and compact != text else []
    # Variant suffixes such as -M/-ME/-AE are often omitted in canonical KG
    # sensor names; keep a family-level alias as a semantic backoff.
    family = re.sub(r"(?i)\b(N\d+\s+\S+?)(?:\s+[A-Z]{1,3})$", r"\1", compact)
    if family != compact:
        variants.append(family)
    words = compact.split()
    if len(words) >= 2:
        variants.append(" ".join(words[-2:]))
    if words and len(words[-1]) > 1:
        variants.append(words[-1])
    return variants


_INVALID_PLATFORM_LABELS = {"COUNTRY", "SENSOR", "OPERATOR", "LOCATION"}
_PLATFORM_LABELS = {"PLATFORM", "AIRCRAFT", "CANDIDATEIDENTITY"}
_COUNTRY_LABELS = {"COUNTRY"}
_UNKNOWN_OPERATOR_NATIONS = {"", "UNKNOWN", "UNK", "N/A", "NONE"}

_NATION_DERIVED_OPERATOR_TERMS = {
    "AMERICAN": "United States",
    "UNITED STATES": "United States",
    "US": "United States",
    "U.S.": "United States",
    "U S": "United States",
    "BRITISH": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom",
    "UK": "United Kingdom",
    "INDIAN": "India",
    "INDIA": "India",
    "ROMANIAN": "Romania",
    "ROMANIA": "Romania",
    "RUSSIAN": "Russia",
    "RUSSIA": "Russia",
    "CHINESE": "China",
    "CHINA": "China",
    "FRENCH": "France",
    "FRANCE": "France",
    "GERMAN": "Germany",
    "GERMANY": "Germany",
    "ITALIAN": "Italy",
    "ITALY": "Italy",
    "JAPANESE": "Japan",
    "JAPAN": "Japan",
    "AUSTRALIAN": "Australia",
    "AUSTRALIA": "Australia",
    "CANADIAN": "Canada",
    "CANADA": "Canada",
    "PAKISTANI": "Pakistan",
    "PAKISTAN": "Pakistan",
    "TURKISH": "Turkey",
    "TURKEY": "Turkey",
    "TÜRKIYE": "Türkiye",
    "ISRAELI": "Israel",
    "ISRAEL": "Israel",
    "EGYPTIAN": "Egypt",
    "EGYPT": "Egypt",
    "IRANIAN": "Iran",
    "IRAN": "Iran",
    "IRAQI": "Iraq",
    "IRAQ": "Iraq",
    "SAUDI": "Saudi Arabia",
    "SAUDI ARABIA": "Saudi Arabia",
    "SOUTH KOREAN": "South Korea",
    "REPUBLIC OF KOREA": "South Korea",
    "NORTH KOREAN": "North Korea",
}
for _country_name in _COUNTRY_NAMES:
    _NATION_DERIVED_OPERATOR_TERMS.setdefault(_country_name, _country_name.title())


def _operator_nation_from_text(text: object) -> str | None:
    normalized = re.sub(r"[^A-Z0-9]+", " ", str(text or "").upper()).strip()
    if not normalized:
        return None
    for term in sorted(_NATION_DERIVED_OPERATOR_TERMS, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(term)}(?![A-Z0-9])", normalized):
            return _NATION_DERIVED_OPERATOR_TERMS[term]
    return None


def _aircraft_variant_without_operator_terms(
    variant: object, fallback: str
) -> tuple[str, str | None]:
    variant_text = str(variant or "").strip()
    inferred_operator_nation = _operator_nation_from_text(variant_text)
    if inferred_operator_nation:
        return fallback, inferred_operator_nation
    return (variant_text or fallback), None


def _labels(row: Mapping[str, Any], key: str) -> set[str]:
    value = row.get(key) or []
    if isinstance(value, str):
        value = [value]
    return {str(label).strip().upper() for label in value if str(label).strip()}


def _valid_graph_hypothesis_row(row: Mapping[str, Any]) -> bool:
    """Return True when graph labels are compatible with the candidate contract."""

    platform_labels = _labels(row, "platform_labels")
    if platform_labels:
        if platform_labels & _INVALID_PLATFORM_LABELS:
            return False
        if not (platform_labels & _PLATFORM_LABELS):
            return False

    aircraft_variant_labels = _labels(row, "aircraft_variant_labels")
    if aircraft_variant_labels:
        if aircraft_variant_labels & _INVALID_PLATFORM_LABELS:
            return False
        if not (aircraft_variant_labels & _PLATFORM_LABELS):
            return False

    operator_nation = str(row.get("operator_nation") or "").strip().upper()
    operator_country_labels = _labels(row, "operator_country_labels")
    if operator_nation not in _UNKNOWN_OPERATOR_NATIONS:
        if operator_country_labels and not (operator_country_labels & _COUNTRY_LABELS):
            return False
        if _labels(row, "operator_labels") & {"SENSOR"}:
            return False

    return True


def _normalized_alias_match_terms(aliases: Sequence[str]) -> list[str]:
    """Return lower-case, punctuation-normalized aliases for graph matching."""

    terms: list[str] = []
    for alias in aliases:
        normalized = re.sub(r"[-_/\[\]().]", " ", str(alias).lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized and normalized not in terms:
            terms.append(normalized)
    return terms


def _compact_alias_length(alias: str) -> int:
    """Return the alphanumeric length used to reject tiny substring aliases."""

    return len(re.sub(r"[^a-z0-9]", "", alias.lower()))


def _safe_alias_substring_match(
    alias: str, sensor_name: str, minimum_chars: int = 3
) -> bool:
    """Return whether alias is specific enough to count as a substring match."""

    if _compact_alias_length(alias) < minimum_chars:
        return False
    return alias in sensor_name


def graph_hypothesis_query(
    aliases: Sequence[str],
    relationship_types: Sequence[str] | int | None = None,
    limit: int | None = None,
) -> tuple[str, dict[str, object]]:
    """Build the parameterized Neo4j query for candidate platform/operator rows."""

    if isinstance(relationship_types, int) and limit is None:
        limit = relationship_types
        relationship_types = None
    if limit is None:
        raise TypeError("limit is required")
    platform_relationship_types = _normalized_relationship_types(
        relationship_types if not isinstance(relationship_types, int) else None
    )

    query = f"""
    MATCH (emitter)
    WITH emitter, toLower(coalesce(emitter.name, emitter.id, properties(emitter)['title'], '')) AS emitter_name
    WITH emitter, emitter_name,
         reduce(normalized = emitter_name, punctuation IN ['-', '_', '/', '[', ']', '(', ')', '.'] | replace(normalized, punctuation, ' ')) AS normalized_emitter_name
    WITH emitter, emitter_name, normalized_emitter_name,
         [token IN $emitter_semantic_tokens WHERE normalized_emitter_name CONTAINS token] AS matched_tokens,
         [alias IN $emitter_alias_match_terms
          WHERE normalized_emitter_name = alias
             OR normalized_emitter_name CONTAINS alias
             OR (alias CONTAINS normalized_emitter_name
                 AND size(replace(normalized_emitter_name, ' ', '')) >= $minimum_reverse_alias_chars)] AS exact_aliases
    WHERE size(exact_aliases) > 0 OR size(matched_tokens) >= $minimum_semantic_token_matches
    OPTIONAL MATCH platform_path = (platform)-[*1..4]-(emitter)
    WHERE (size($platform_relationship_types) = 0 OR all(rel IN relationships(platform_path) WHERE type(rel) IN $platform_relationship_types))
      AND any(label IN labels(platform) WHERE label IN ['Platform', 'Aircraft'])
      AND none(label IN labels(platform) WHERE label IN ['Country', 'Sensor', 'Operator', 'Location'])
    OPTIONAL MATCH aircraft_variant_path = (platform)-[*0..1]-(aircraft_variant)
    WHERE (aircraft_variant IS NULL OR all(rel IN relationships(aircraft_variant_path) WHERE type(rel) IN $aircraft_variant_relationship_types))
      AND (aircraft_variant IS NULL OR (
        any(label IN labels(aircraft_variant) WHERE label IN ['Platform', 'Aircraft'])
        AND none(label IN labels(aircraft_variant) WHERE label IN ['Country', 'Sensor', 'Operator', 'Location'])
      ))
    OPTIONAL MATCH emitter_variant_path = (emitter)-[*0..1]-(emitter_variant)
    WHERE (emitter_variant IS NULL OR all(rel IN relationships(emitter_variant_path) WHERE type(rel) IN $emitter_variant_relationship_types))
      AND (emitter_variant IS NULL OR any(label IN labels(emitter_variant) WHERE label IN ['Sensor', 'Entity']))
    OPTIONAL MATCH (platform)-[operator_relationship]-(operator)
    WHERE type(operator_relationship) IN $operator_relationship_types
      AND any(label IN labels(operator) WHERE label = 'Operator')
    OPTIONAL MATCH (platform)-[operator_country_relationship]-(operator_country)
    WHERE type(operator_country_relationship) IN $operator_country_relationship_types
      AND any(label IN labels(operator_country) WHERE label = 'Country')
    OPTIONAL MATCH (operator)-[operator_country_via_operator_relationship]-(operator_country_via_operator)
    WHERE type(operator_country_via_operator_relationship) IN $operator_country_via_operator_relationship_types
      AND any(label IN labels(operator_country_via_operator) WHERE label = 'Country')
    OPTIONAL MATCH (aircraft_variant)-[aircraft_variant_operator_relationship]-(aircraft_variant_operator)
    WHERE operator_country IS NULL
      AND operator_country_via_operator IS NULL
      AND type(aircraft_variant_operator_relationship) IN $operator_relationship_types
      AND any(label IN labels(aircraft_variant_operator) WHERE label = 'Operator')
    OPTIONAL MATCH (aircraft_variant)-[aircraft_variant_operator_country_relationship]-(aircraft_variant_operator_country)
    WHERE operator_country IS NULL
      AND operator_country_via_operator IS NULL
      AND type(aircraft_variant_operator_country_relationship) IN $operator_country_relationship_types
      AND any(label IN labels(aircraft_variant_operator_country) WHERE label = 'Country')
    OPTIONAL MATCH (aircraft_variant_operator)-[aircraft_variant_country_via_operator_relationship]-(aircraft_variant_country_via_operator)
    WHERE operator_country IS NULL
      AND operator_country_via_operator IS NULL
      AND type(aircraft_variant_country_via_operator_relationship) IN $operator_country_via_operator_relationship_types
      AND any(label IN labels(aircraft_variant_country_via_operator) WHERE label = 'Country')
    WITH emitter, platform, aircraft_variant, emitter_variant, operator, operator_country, operator_country_via_operator, aircraft_variant_operator, aircraft_variant_operator_country, aircraft_variant_country_via_operator, platform_path, exact_aliases, matched_tokens
    WHERE platform IS NOT NULL
    OPTIONAL MATCH (kinematic_subject)-[kinematic_fact]->(kinematic_value)
    WHERE kinematic_subject IN [platform, aircraft_variant]
      AND type(kinematic_fact) = $kinematic_fact_relationship_type
      AND kinematic_fact.predicate IN ['MAX_SPEED_KT', 'TYPICAL_SPEED_KT', 'CRUISE_SPEED_KT', 'SERVICE_CEILING_M', 'MAX_ALTITUDE_M', 'TYPICAL_ALTITUDE_M']
    WITH coalesce(platform.name, platform.id, properties(platform)['title']) AS hypothesis,
         coalesce(aircraft_variant.name, aircraft_variant.id, properties(aircraft_variant)['title'], platform.name, platform.id, properties(platform)['title']) AS aircraft_variant,
         coalesce(emitter_variant.name, emitter_variant.id, properties(emitter_variant)['title'], emitter.name, emitter.id, properties(emitter)['title']) AS emitter_variant,
         coalesce(operator_country.name, operator_country.id, properties(operator_country)['title'],
                  operator_country_via_operator.name, operator_country_via_operator.id, properties(operator_country_via_operator)['title'],
                  aircraft_variant_operator_country.name, aircraft_variant_operator_country.id, properties(aircraft_variant_operator_country)['title'],
                  aircraft_variant_country_via_operator.name, aircraft_variant_country_via_operator.id, properties(aircraft_variant_country_via_operator)['title'],
                  'Unknown') AS operator_nation,
         collect(DISTINCT coalesce(emitter.name, emitter.id, properties(emitter)['title'])) AS matched_aliases,
         count(DISTINCT platform_path) AS support_count,
         collect(DISTINCT [rel IN relationships(platform_path) | type(rel)]) AS evidence_paths,
         labels(platform) AS platform_labels,
         CASE WHEN aircraft_variant IS NULL THEN [] ELSE labels(aircraft_variant) END AS aircraft_variant_labels,
         CASE
             WHEN operator IS NOT NULL THEN labels(operator)
             WHEN aircraft_variant_operator IS NOT NULL THEN labels(aircraft_variant_operator)
             ELSE []
         END AS operator_labels,
         CASE
             WHEN operator_country IS NOT NULL THEN labels(operator_country)
             WHEN operator_country_via_operator IS NOT NULL THEN labels(operator_country_via_operator)
             WHEN aircraft_variant_operator_country IS NOT NULL THEN labels(aircraft_variant_operator_country)
             WHEN aircraft_variant_country_via_operator IS NOT NULL THEN labels(aircraft_variant_country_via_operator)
             ELSE []
         END AS operator_country_labels,
         max(size(exact_aliases) * 10 + size(matched_tokens)) AS semantic_match_score,
         collect(DISTINCT CASE WHEN kinematic_fact.predicate = 'MAX_SPEED_KT' THEN coalesce(kinematic_value.name, kinematic_value.id, properties(kinematic_value)['title']) END) AS max_speed_kt_values,
         collect(DISTINCT CASE WHEN kinematic_fact.predicate = 'TYPICAL_SPEED_KT' THEN coalesce(kinematic_value.name, kinematic_value.id, properties(kinematic_value)['title']) END) AS typical_speed_kt_values,
         collect(DISTINCT CASE WHEN kinematic_fact.predicate = 'CRUISE_SPEED_KT' THEN coalesce(kinematic_value.name, kinematic_value.id, properties(kinematic_value)['title']) END) AS cruise_speed_kt_values,
         collect(DISTINCT CASE WHEN kinematic_fact.predicate = 'SERVICE_CEILING_M' THEN coalesce(kinematic_value.name, kinematic_value.id, properties(kinematic_value)['title']) END) AS service_ceiling_m_values,
         collect(DISTINCT CASE WHEN kinematic_fact.predicate = 'MAX_ALTITUDE_M' THEN coalesce(kinematic_value.name, kinematic_value.id, properties(kinematic_value)['title']) END) AS max_altitude_m_values,
         collect(DISTINCT CASE WHEN kinematic_fact.predicate = 'TYPICAL_ALTITUDE_M' THEN coalesce(kinematic_value.name, kinematic_value.id, properties(kinematic_value)['title']) END) AS typical_altitude_m_values
    RETURN hypothesis,
           operator_nation,
           aircraft_variant,
           emitter_variant,
           matched_aliases,
           support_count,
           evidence_paths,
           platform_labels,
           aircraft_variant_labels,
           operator_labels,
           operator_country_labels,
           semantic_match_score,
           max_speed_kt_values,
           typical_speed_kt_values,
           cruise_speed_kt_values,
           service_ceiling_m_values,
           max_altitude_m_values,
           typical_altitude_m_values
    ORDER BY semantic_match_score DESC, support_count DESC, hypothesis ASC, operator_nation ASC, aircraft_variant ASC, emitter_variant ASC
    LIMIT $limit
    """
    semantic_tokens = emitter_semantic_tokens(aliases)
    return query, {
        "emitter_aliases": list(aliases),
        "emitter_alias_match_terms": _normalized_alias_match_terms(aliases),
        "emitter_semantic_tokens": semantic_tokens,
        "minimum_reverse_alias_chars": 3,
        "minimum_semantic_token_matches": (
            min(2, len(semantic_tokens)) if semantic_tokens else 1
        ),
        "platform_relationship_types": platform_relationship_types,
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
        "limit": int(limit),
    }


def graph_probe_queries(
    aliases: Sequence[str],
) -> list[tuple[str, str, dict[str, object]]]:
    """Return named diagnostic Cypher probes for inspecting graph coverage."""

    semantic_tokens = emitter_semantic_tokens(aliases)
    params = {
        "emitter_aliases": list(aliases),
        "emitter_semantic_tokens": semantic_tokens,
        "minimum_semantic_token_matches": (
            min(2, len(semantic_tokens)) if semantic_tokens else 1
        ),
    }
    return [
        (
            "alias_nodes",
            """
            MATCH (node)
            WITH node, toLower(coalesce(node.name, node.id, properties(node)['title'], '')) AS node_name
            WITH node, node_name, reduce(normalized = node_name, punctuation IN ['-', '_', '/', '[', ']', '(', ')', '.'] | replace(normalized, punctuation, ' ')) AS normalized_node_name
            WHERE any(alias IN $emitter_aliases WHERE node_name CONTAINS toLower(alias))
               OR size([token IN $emitter_semantic_tokens WHERE normalized_node_name CONTAINS token]) >= $minimum_semantic_token_matches
            RETURN labels(node) AS labels, coalesce(node.name, node.id, properties(node)['title']) AS name
            ORDER BY name
            LIMIT 25
            """,
            params,
        ),
        (
            "alias_neighbours",
            """
            MATCH (node)-[rel]-(neighbour)
            WITH node, rel, neighbour, toLower(coalesce(node.name, node.id, properties(node)['title'], '')) AS node_name
            WITH node, rel, neighbour, node_name, reduce(normalized = node_name, punctuation IN ['-', '_', '/', '[', ']', '(', ')', '.'] | replace(normalized, punctuation, ' ')) AS normalized_node_name
            WHERE any(alias IN $emitter_aliases WHERE node_name CONTAINS toLower(alias))
               OR size([token IN $emitter_semantic_tokens WHERE normalized_node_name CONTAINS token]) >= $minimum_semantic_token_matches
            RETURN coalesce(node.name, node.id, properties(node)['title']) AS alias_node,
                   type(rel) AS relationship,
                   labels(neighbour) AS neighbour_labels,
                   coalesce(neighbour.name, neighbour.id, properties(neighbour)['title']) AS neighbour
            ORDER BY alias_node, relationship, neighbour
            LIMIT 50
            """,
            params,
        ),
    ]


def evidence_paths_query_id(evidence_paths: object) -> str:
    """Return a stable text identifier for graph evidence paths.

    Neo4j evidence-path rows are often lists of relationship-type lists, while
    offline seed rows may already be strings.  Notebook feature rows need a
    scalar ``evidence_query_id`` value, so normalize both shapes without
    assuming every path item is directly joinable as text.
    """

    if not evidence_paths:
        return ""
    if isinstance(evidence_paths, str):
        return evidence_paths
    if not isinstance(evidence_paths, Iterable):
        return str(evidence_paths)

    normalized: list[str] = []
    for path in evidence_paths:
        if isinstance(path, str):
            normalized.append(path)
        elif isinstance(path, Iterable):
            normalized.append(">".join(str(segment) for segment in path))
        else:
            normalized.append(str(path))
    return "|".join(item for item in normalized if item)


def _rows(result: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in result]


def _first_numeric_value(
    value: object, preferred_units: Sequence[str] = ()
) -> float | None:
    """Return the first parseable number from nested Neo4j scalar/list values.

    When source strings include multiple unit conversions, prefer the value next
    to the requested unit so ranges use knots/meters rather than Mach or feet.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        for unit in preferred_units:
            unit_match = re.search(
                rf"([-+]?\d+(?:,\d{{3}})*(?:\.\d+)?)\s*{unit}\b",
                value,
                flags=re.IGNORECASE,
            )
            if unit_match:
                return float(unit_match.group(1).replace(",", ""))
        match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", value)
        if match:
            return float(match.group(0).replace(",", ""))
        return None
    if isinstance(value, Iterable):
        for item in value:
            parsed = _first_numeric_value(item, preferred_units)
            if parsed is not None:
                return parsed
    return None


def _kinematic_range(
    row: Mapping[str, Any],
    keys: Sequence[str],
    default_upper: float,
    preferred_units: Sequence[str] = (),
) -> list[float]:
    """Build the candidate [low, high] range from extracted max performance facts."""

    for key in keys:
        parsed = _first_numeric_value(row.get(key), preferred_units)
        if parsed is not None and parsed > 0:
            return [0.0, parsed]
    return [0.0, default_upper]


def relationship_type_counts_query() -> str:
    """Return Cypher for counting every relationship type in the graph."""

    return """
    MATCH ()-[rel]->()
    RETURN type(rel) AS relationship_type, count(*) AS count
    ORDER BY count DESC, relationship_type ASC
    """


def fetch_relationship_type_counts_with_session(
    session: object,
) -> list[dict[str, object]]:
    """Return relationship types and counts from a Neo4j session-like object."""

    return _rows(session.run(relationship_type_counts_query()))


def fetch_relationship_type_counts(
    uri: str, user: str, password: str, database: str | None = None
) -> list[dict[str, object]]:
    """Fetch relationship type counts from a live Neo4j database."""

    user, password = _validate_neo4j_credentials(user, password)
    neo4j = importlib.import_module("neo4j")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        try:
            driver.verify_connectivity()
            with driver.session(
                **({"database": database} if database else {})
            ) as session:
                return fetch_relationship_type_counts_with_session(session)
        except neo4j.exceptions.ServiceUnavailable as error:
            raise RuntimeError(
                _neo4j_connection_error_message(uri, database)
            ) from error
        except neo4j.exceptions.AuthError as error:
            raise RuntimeError(
                f"Neo4j rejected the credentials for {uri!r}."
            ) from error
    finally:
        driver.close()


def probe_knowledge_graph_with_session(
    session: object, obs: EmissionObservation | Mapping[str, Any]
) -> dict[str, object]:
    """Run diagnostic graph probes for the observation's emitter aliases."""

    sensor_name = _observation_value(obs, "emission_sensor_name")
    aliases = emitter_aliases(sensor_name)
    probes = []
    for name, query, params in graph_probe_queries(aliases):
        probes.append({"name": name, "rows": _rows(session.run(query, **params))})
    return {"emitter_aliases": aliases, "probes": probes}


def fetch_graph_hypotheses_with_session(
    session: object,
    obs: EmissionObservation | Mapping[str, Any],
    n: int,
    relationship_types: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Query a Neo4j session-like object for candidate hypotheses."""
    ea = emitter_aliases(_observation_value(obs, "emission_sensor_name"))
    print(f'emitter_aliases: {ea}')

    query, params = graph_hypothesis_query(
        ea,
        relationship_types,
        max(n * 4, n),
    )
    rows = _rows(session.run(query, **params))
    hypotheses: list[dict[str, object]] = []
    for row in rows:
        if not _valid_graph_hypothesis_row(row):
            continue
        hypothesis = str(row.get("hypothesis") or "").strip()
        if not hypothesis:
            continue
        aircraft_variant, inferred_operator_nation = _aircraft_variant_without_operator_terms(
            row.get("aircraft_variant"), hypothesis
        )
        operator_nation = str(row.get("operator_nation") or "Unknown").strip() or "Unknown"
        if operator_nation.upper() in _UNKNOWN_OPERATOR_NATIONS and inferred_operator_nation:
            operator_nation = inferred_operator_nation
        hypotheses.append(
            {
                "hypothesis": hypothesis,
                "operator_nation": operator_nation,
                "aircraft_variant": aircraft_variant,
                "emitter_variant": str(row.get("emitter_variant") or ""),
                "emitter_aliases": list(row.get("matched_aliases") or []),
                "platform_class": _observation_value(obs, "emission_target_type"),
                "typical_speed_kt": _kinematic_range(
                    row,
                    [
                        "max_speed_kt_values",
                        "typical_speed_kt_values",
                        "cruise_speed_kt_values",
                    ],
                    float("nan"),
                    ["kt", "kts", "knot", "knots"],
                ),
                "typical_altitude_m": _kinematic_range(
                    row,
                    [
                        "service_ceiling_m_values",
                        "max_altitude_m_values",
                        "typical_altitude_m_values",
                    ],
                    float("nan"),
                    ["m", "meter", "meters"],
                ),
                "kg_support_count": int(row.get("support_count") or 0),
                "evidence_paths": row.get("evidence_paths") or [],
                "semantic_match_score": float(row.get("semantic_match_score") or 0.0),
            }
        )
    return hypotheses[:n]


def fetch_graph_hypotheses(
    obs: EmissionObservation | Mapping[str, Any],
    n: int,
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
    relationship_types: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Fetch hypotheses from a live Neo4j database."""

    user, password = _validate_neo4j_credentials(user, password)
    neo4j = importlib.import_module("neo4j")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        try:
            driver.verify_connectivity()
            with driver.session(
                **({"database": database} if database else {})
            ) as session:
                return fetch_graph_hypotheses_with_session(
                    session, obs, n, relationship_types
                )
        except neo4j.exceptions.ServiceUnavailable as error:
            raise RuntimeError(
                _neo4j_connection_error_message(uri, database)
            ) from error
        except neo4j.exceptions.AuthError as error:
            raise RuntimeError(
                f"Neo4j rejected the credentials for {uri!r}."
            ) from error
    finally:
        driver.close()


def select_offline_hypotheses(
    obs: EmissionObservation | Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    n: int,
) -> list[dict[str, object]]:
    """Rank graph rows deterministically with observation compatibility checks."""

    sensor_name = _observation_value(obs, "emission_sensor_name").lower()
    target_type = _observation_value(obs, "emission_target_type")
    speed = _optional_float(_observation_value(obs, "emission_speed"))
    altitude = _optional_float(_observation_value(obs, "emission_altitude"))

    def range_score(value: float | None, low: float, high: float) -> float:
        if value is None:
            return 0.0
        return 1.0 if low <= value <= high else 0.0

    def score(candidate: Mapping[str, Any]) -> tuple[float, str, str]:
        aliases = [str(alias).lower() for alias in candidate.get("emitter_aliases", [])]
        observed_tokens = set(emitter_semantic_tokens([sensor_name]))
        candidate_tokens = set(emitter_semantic_tokens(aliases))
        token_score = (
            len(observed_tokens & candidate_tokens) / max(len(candidate_tokens), 1)
            if candidate_tokens
            else 0.0
        )
        alias_score = max(
            1.0
            if any(_safe_alias_substring_match(alias, sensor_name) for alias in aliases)
            else 0.0,
            token_score,
        )
        class_score = (
            1.0
            if target_type and target_type == candidate.get("platform_class")
            else 0.0
        )
        speed_low, speed_high = candidate.get("typical_speed_kt", [0.0, 2500.0])
        alt_low, alt_high = candidate.get("typical_altitude_m", [0.0, 25000.0])
        kg_score = math.log1p(max(float(candidate.get("kg_support_count") or 0.0), 0.0))
        return (
            kg_score
            + alias_score * 3.0
            + class_score
            + range_score(speed, float(speed_low), float(speed_high))
            + range_score(altitude, float(alt_low), float(alt_high)),
            str(candidate.get("hypothesis") or ""),
            str(candidate.get("operator_nation") or ""),
        )

    ranked = sorted(candidates, key=score, reverse=True)
    selected: list[dict[str, object]] = []
    seen_hypotheses: set[tuple[str, str, str, str]] = set()
    for candidate in ranked:
        hypothesis = str(candidate.get("hypothesis") or "").strip()
        identity = _hypothesis_identity(candidate)
        if not hypothesis or identity in seen_hypotheses:
            continue
        seen_hypotheses.add(identity)
        selected.append(dict(candidate))
        if len(selected) >= n:
            break
    return selected


def build_llm_hypothesis_prompt(
    obs: EmissionObservation | Mapping[str, Any],
    kg_rows: Sequence[Mapping[str, Any]],
    n: int,
) -> str:
    return "\n".join(
        [
            f"Generate exactly {n} candidate emitter-platform/operator hypotheses from the knowledge-graph rows.",
            'Return JSON only: {"hypotheses":[{"hypothesis": str, "operator_nation": str, "rationale": str}]}',
            f"Observation: {json.dumps(_observation_dict(obs), sort_keys=True)}",
            f"Knowledge-graph evidence rows: {json.dumps(list(kg_rows), sort_keys=True)}",
        ]
    )


def _observation_dict(obs: EmissionObservation | Mapping[str, Any]) -> dict[str, Any]:
    if is_dataclass(obs):
        return asdict(obs)
    return dict(obs)


def _observation_value(obs: EmissionObservation | Mapping[str, Any], key: str) -> str:
    value = getattr(obs, key) if is_dataclass(obs) else obs.get(key)
    return "" if value is None else str(value)


def _hypothesis_identity(candidate: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the fields that distinguish a graph candidate hypothesis."""

    hypothesis = str(candidate.get("hypothesis") or "").strip()
    return (
        hypothesis,
        str(candidate.get("operator_nation") or "").strip(),
        str(candidate.get("aircraft_variant") or hypothesis).strip(),
        str(candidate.get("emitter_variant") or "").strip(),
    )


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def add_hypothesis_generation_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = commands.add_parser(
        "generate-hypotheses",
        help="probe Neo4j and generate candidate hypotheses for one observation JSON file",
    )
    parser.add_argument(
        "--observation-json",
        required=True,
        help="JSON object containing an emission observation",
    )
    parser.add_argument(
        "--count", type=int, default=10, help="number of candidate hypotheses to return"
    )
    parser.add_argument(
        "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI"
    )
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--neo4j-database", help="optional Neo4j database name")
    parser.add_argument(
        "--relationship-type",
        action="append",
        dest="relationship_types",
        help="relationship type to allow in emitter-to-platform path expansion; may be repeated",
    )
    parser.add_argument(
        "--list-relationship-types",
        action="store_true",
        help="print graph relationship types/counts and exit",
    )
    parser.set_defaults(handler=run_hypothesis_generation_command)


def run_hypothesis_generation_command(args: argparse.Namespace) -> None:
    if args.list_relationship_types:
        rows = fetch_relationship_type_counts(
            args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database
        )
        print(json.dumps({"relationship_types": rows}, indent=2))
        return
    observation = json.loads(Path(args.observation_json).read_text(encoding="utf-8"))
    rows = fetch_graph_hypotheses(
        observation,
        args.count,
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.neo4j_database,
        args.relationship_types,
    )
    print(
        json.dumps(
            {
                "hypotheses": select_offline_hypotheses(observation, rows, args.count),
                "row_count": len(rows),
            },
            indent=2,
        )
    )
