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
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .cmo_observation_ingest import EmissionObservation
from .graph_ingest import _neo4j_connection_error_message, _validate_neo4j_credentials


def emitter_aliases(sensor_name: str) -> list[str]:
    """Return normalized aliases for a CMO emission sensor string."""

    text = sensor_name.strip()
    aliases: list[str] = []
    if text:
        aliases.append(text)
        if "[" in text and "]" in text:
            bracketed = text.split("[", 1)[1].split("]", 1)[0].strip()
            prefix = text.split("[", 1)[0].strip()
            aliases.extend(part.strip() for part in (bracketed, prefix) if part.strip())
    deduped: list[str] = []
    for alias in aliases:
        if alias.lower() not in {existing.lower() for existing in deduped}:
            deduped.append(alias)
    return deduped


def graph_hypothesis_query(aliases: Sequence[str], limit: int) -> tuple[str, dict[str, object]]:
    """Build the parameterized Neo4j query for candidate platform/operator rows."""

    query = """
    MATCH (emitter)
    WHERE any(alias IN $emitter_aliases WHERE
      toLower(coalesce(emitter.name, emitter.id, emitter.title, '')) CONTAINS toLower(alias)
      OR toLower(alias) CONTAINS toLower(coalesce(emitter.name, emitter.id, emitter.title, '')))
    OPTIONAL MATCH platform_path = (emitter)-[*1..3]-(platform)
    WHERE any(label IN labels(platform) WHERE label IN ['Platform', 'Aircraft', 'Entity', 'CandidateIdentity'])
    OPTIONAL MATCH (platform)-[:OPERATED_BY|OPERATOR|USED_BY|SERVICE_WITH|ASSIGNED_TO]-(operator)
    WITH emitter, platform, operator, platform_path
    WHERE platform IS NOT NULL
    WITH coalesce(platform.name, platform.id, platform.title) AS hypothesis,
         coalesce(operator.name, operator.id, operator.title, 'Unknown') AS operator_nation,
         collect(DISTINCT coalesce(emitter.name, emitter.id, emitter.title)) AS matched_aliases,
         count(DISTINCT platform_path) AS support_count,
         collect(DISTINCT [rel IN relationships(platform_path) | type(rel)]) AS evidence_paths,
         labels(platform) AS platform_labels
    RETURN hypothesis,
           operator_nation,
           matched_aliases,
           support_count,
           evidence_paths,
           platform_labels
    ORDER BY support_count DESC, hypothesis ASC, operator_nation ASC
    LIMIT $limit
    """
    return query, {"emitter_aliases": list(aliases), "limit": int(limit)}


def graph_probe_queries(aliases: Sequence[str]) -> list[tuple[str, str, dict[str, object]]]:
    """Return named diagnostic Cypher probes for inspecting graph coverage."""

    params = {"emitter_aliases": list(aliases)}
    return [
        (
            "alias_nodes",
            """
            MATCH (node)
            WHERE any(alias IN $emitter_aliases WHERE toLower(coalesce(node.name, node.id, node.title, '')) CONTAINS toLower(alias))
            RETURN labels(node) AS labels, coalesce(node.name, node.id, node.title) AS name
            ORDER BY name
            LIMIT 25
            """,
            params,
        ),
        (
            "alias_neighbours",
            """
            MATCH (node)-[rel]-(neighbour)
            WHERE any(alias IN $emitter_aliases WHERE toLower(coalesce(node.name, node.id, node.title, '')) CONTAINS toLower(alias))
            RETURN coalesce(node.name, node.id, node.title) AS alias_node,
                   type(rel) AS relationship,
                   labels(neighbour) AS neighbour_labels,
                   coalesce(neighbour.name, neighbour.id, neighbour.title) AS neighbour
            ORDER BY alias_node, relationship, neighbour
            LIMIT 50
            """,
            params,
        ),
    ]


def _rows(result: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in result]


def probe_knowledge_graph_with_session(session: object, obs: EmissionObservation | Mapping[str, Any]) -> dict[str, object]:
    """Run diagnostic graph probes for the observation's emitter aliases."""

    sensor_name = _observation_value(obs, "emission_sensor_name")
    aliases = emitter_aliases(sensor_name)
    probes = []
    for name, query, params in graph_probe_queries(aliases):
        probes.append({"name": name, "rows": _rows(session.run(query, **params))})
    return {"emitter_aliases": aliases, "probes": probes}


def fetch_graph_hypotheses_with_session(session: object, obs: EmissionObservation | Mapping[str, Any], n: int) -> list[dict[str, object]]:
    """Query a Neo4j session-like object for candidate hypotheses."""

    query, params = graph_hypothesis_query(emitter_aliases(_observation_value(obs, "emission_sensor_name")), max(n * 4, n))
    rows = _rows(session.run(query, **params))
    hypotheses: list[dict[str, object]] = []
    for row in rows:
        hypothesis = str(row.get("hypothesis") or "").strip()
        if not hypothesis:
            continue
        hypotheses.append(
            {
                "hypothesis": hypothesis,
                "operator_nation": str(row.get("operator_nation") or "Unknown"),
                "emitter_aliases": list(row.get("matched_aliases") or []),
                "platform_class": _observation_value(obs, "emission_target_type"),
                "typical_speed_kt": [0.0, 2500.0],
                "typical_altitude_m": [0.0, 25000.0],
                "kg_support_count": int(row.get("support_count") or 0),
                "evidence_paths": row.get("evidence_paths") or [],
            }
        )
    return hypotheses[:n]


def fetch_graph_hypotheses(obs: EmissionObservation | Mapping[str, Any], n: int, uri: str, user: str, password: str, database: str | None = None) -> list[dict[str, object]]:
    """Fetch hypotheses from a live Neo4j database."""

    user, password = _validate_neo4j_credentials(user, password)
    neo4j = importlib.import_module("neo4j")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        try:
            driver.verify_connectivity()
            with driver.session(**({"database": database} if database else {})) as session:
                return fetch_graph_hypotheses_with_session(session, obs, n)
        except neo4j.exceptions.ServiceUnavailable as error:
            raise RuntimeError(_neo4j_connection_error_message(uri, database)) from error
        except neo4j.exceptions.AuthError as error:
            raise RuntimeError(f"Neo4j rejected the credentials for {uri!r}.") from error
    finally:
        driver.close()


def select_offline_hypotheses(obs: EmissionObservation | Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], n: int) -> list[dict[str, object]]:
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
        alias_score = 1.0 if any(alias and alias in sensor_name for alias in aliases) else 0.0
        class_score = 1.0 if target_type and target_type == candidate.get("platform_class") else 0.0
        speed_low, speed_high = candidate.get("typical_speed_kt", [0.0, 2500.0])
        alt_low, alt_high = candidate.get("typical_altitude_m", [0.0, 25000.0])
        kg_score = float(candidate.get("kg_support_count") or 0.0)
        return (
            kg_score + alias_score * 3.0 + class_score + range_score(speed, float(speed_low), float(speed_high)) + range_score(altitude, float(alt_low), float(alt_high)),
            str(candidate.get("hypothesis") or ""),
            str(candidate.get("operator_nation") or ""),
        )

    return [dict(candidate) for candidate in sorted(candidates, key=score, reverse=True)[:n]]


def build_llm_hypothesis_prompt(obs: EmissionObservation | Mapping[str, Any], kg_rows: Sequence[Mapping[str, Any]], n: int) -> str:
    return "\n".join(
        [
            f"Generate exactly {n} candidate emitter-platform/operator hypotheses from the knowledge-graph rows.",
            "Return JSON only: {\"hypotheses\":[{\"hypothesis\": str, \"operator_nation\": str, \"rationale\": str}]}",
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


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def add_hypothesis_generation_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser("generate-hypotheses", help="probe Neo4j and generate candidate hypotheses for one observation JSON file")
    parser.add_argument("--observation-json", required=True, help="JSON object containing an emission observation")
    parser.add_argument("--count", type=int, default=10, help="number of candidate hypotheses to return")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--neo4j-database", help="optional Neo4j database name")
    parser.set_defaults(handler=run_hypothesis_generation_command)


def run_hypothesis_generation_command(args: argparse.Namespace) -> None:
    observation = json.loads(Path(args.observation_json).read_text(encoding="utf-8"))
    rows = fetch_graph_hypotheses(observation, args.count, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
    print(json.dumps({"hypotheses": select_offline_hypotheses(observation, rows, args.count), "row_count": len(rows)}, indent=2))
