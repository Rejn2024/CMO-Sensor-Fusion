"""Parse CMO Lua emission observations and ingest them into the Neo4j evidence graph.

The Lua exporter emits text lines beginning with ``PY_CONTACT_LOG`` followed by
comma-separated ``key : value`` fields.  These records are observations in the
Phase-2 graph-database layer from ``GraphDB_Probability_Architecture_Seed.txt``:
they preserve timestamped source evidence, contact/emission/sensor context,
kinematics, and classification attributes for downstream feature extraction.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .graph_ingest import _neo4j_connection_error_message, _validate_neo4j_credentials, stable_id

LOG_PREFIX = "PY_CONTACT_LOG"
OBSERVATION_SCHEMA = "cmo_emission_observation_v1"

_FIELD_RE = re.compile(r"\s*([^:,]+?)\s*:\s*(.*?)\s*(?=\s*,\s*[^:,]+?\s*:|\s*$)")


@dataclass(frozen=True)
class EmissionObservation:
    """Normalized CMO emission observation parsed from one Lua log line."""

    observation_id: str
    time: int | float | str
    sensor_aircraft: str
    emission_sensor_name: str
    emission_age: float | None = None
    emission_solid: bool | None = None
    emission_type: int | str | None = None
    emission_role: int | str | None = None
    emission_latitude: float | None = None
    emission_longitude: float | None = None
    emission_heading: float | None = None
    emission_altitude: float | None = None
    emission_speed: float | None = None
    emission_target_type: str = ""
    emission_classificationlevel: int | str | None = None
    source_line: int | None = None
    source: str = "cmo_lua"
    schema: str = OBSERVATION_SCHEMA


def _coerce_value(value: str) -> object:
    """Convert Lua-exported scalar text into bool/int/float where safe."""

    text = value.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", text) or re.fullmatch(
        r"[-+]?\d+[eE][-+]?\d+", text
    ):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def parse_observation_line(line: str, *, source_line: int | None = None) -> EmissionObservation | None:
    """Parse a single ``PY_CONTACT_LOG`` line into an observation, if present."""

    if LOG_PREFIX not in line:
        return None
    payload = line.split(LOG_PREFIX, 1)[1]
    fields = {key.strip(): _coerce_value(value) for key, value in _FIELD_RE.findall(payload)}
    required = ["Time", "Sensor_aircraft", "Emission_sensor_name"]
    if any(not str(fields.get(key, "")).strip() for key in required):
        return None

    observation_id = stable_id(
        "cmo-observation",
        str(fields.get("Time", "")),
        str(fields.get("Sensor_aircraft", "")),
        str(fields.get("Emission_sensor_name", "")),
        str(fields.get("Emission_latitude", "")),
        str(fields.get("Emission_longitude", "")),
        str(source_line or ""),
    )
    return EmissionObservation(
        observation_id=observation_id,
        time=fields["Time"],
        sensor_aircraft=str(fields["Sensor_aircraft"]),
        emission_sensor_name=str(fields["Emission_sensor_name"]),
        emission_age=_optional_float(fields.get("Emission_age")),
        emission_solid=_optional_bool(fields.get("Emission_solid")),
        emission_type=fields.get("Emission_type"),
        emission_role=fields.get("Emission_role"),
        emission_latitude=_optional_float(fields.get("Emission_latitude")),
        emission_longitude=_optional_float(fields.get("Emission_longitude")),
        emission_heading=_optional_float(fields.get("Emission_heading")),
        emission_altitude=_optional_float(fields.get("Emission_altitude")),
        emission_speed=_optional_float(fields.get("Emission_speed")),
        emission_target_type=str(fields.get("Emission_target_type", "")),
        emission_classificationlevel=fields.get("Emission_classificationlevel"),
        source_line=source_line,
    )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def parse_observations(lines: Iterable[str]) -> Iterator[EmissionObservation]:
    """Yield all CMO emission observations from text log lines."""

    for line_number, line in enumerate(lines, start=1):
        observation = parse_observation_line(line, source_line=line_number)
        if observation is not None:
            yield observation


def write_observations_jsonl(observations: Sequence[EmissionObservation], path: str | Path) -> None:
    """Write parsed observations as reviewable JSONL before graph ingestion."""

    Path(path).write_text("".join(json.dumps(asdict(obs), sort_keys=True) + "\n" for obs in observations), encoding="utf-8")


def create_cmo_observation_schema(session: object) -> None:
    """Create constraints for the typed CMO observation ontology."""

    for statement in [
        "CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT contact_id IF NOT EXISTS FOR (c:Contact) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT sensor_id IF NOT EXISTS FOR (s:Sensor) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT platform_id IF NOT EXISTS FOR (p:Platform) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT emission_id IF NOT EXISTS FOR (e:Emission) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT platform_class_id IF NOT EXISTS FOR (pc:PlatformClass) REQUIRE pc.id IS UNIQUE",
        "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
    ]:
        session.run(statement)


def _write_observation(tx: object, observation: EmissionObservation) -> None:
    """Write one parsed observation into the shared Neo4j evidence graph."""

    contact_name = f"Contact observed at {observation.time} by {observation.sensor_aircraft}"
    contact_id = stable_id("contact", observation.observation_id)
    sensor_id = stable_id("sensor", observation.emission_sensor_name.lower())
    platform_id = stable_id("platform", observation.sensor_aircraft.lower())
    emission_id = stable_id("emission", observation.observation_id, observation.emission_sensor_name.lower())
    class_id = stable_id("platform-class", observation.emission_target_type.lower()) if observation.emission_target_type else ""
    source_id = stable_id("source", observation.source, str(observation.source_line or ""))

    tx.run(
        """
        MERGE (obs:Observation {id: $observation_id})
          SET obs.schema = $schema, obs.time = $time, obs.source = $source,
              obs.source_line = $source_line, obs.age_seconds = $emission_age,
              obs.solid = $emission_solid, obs.latitude = $emission_latitude,
              obs.longitude = $emission_longitude, obs.heading = $emission_heading,
              obs.altitude = $emission_altitude, obs.speed = $emission_speed,
              obs.classification_level = $emission_classificationlevel
        MERGE (contact:Contact {id: $contact_id}) SET contact.name = $contact_name
        MERGE (platform:Platform {id: $platform_id}) SET platform.name = $sensor_aircraft
        MERGE (sensor:Sensor {id: $sensor_id}) SET sensor.name = $emission_sensor_name
        MERGE (emission:Emission {id: $emission_id})
          SET emission.sensor_name = $emission_sensor_name, emission.type = $emission_type, emission.role = $emission_role
        MERGE (source:Source {id: $source_id}) SET source.source_type = $source, source.locator = $source_locator
        MERGE (contact)-[:HAS_OBSERVATION]->(obs)
        MERGE (obs)-[:OBSERVED_BY]->(platform)
        MERGE (platform)-[:HAS_SENSOR]->(sensor)
        MERGE (contact)-[:EMITTED]->(emission)
        MERGE (emission)-[:DETECTED_BY]->(sensor)
        MERGE (obs)-[:DERIVED_FROM]->(source)
        WITH obs, contact
        CALL {
          WITH obs, contact
          WITH obs, contact WHERE $class_id <> ''
          MERGE (platform_class:PlatformClass {id: $class_id}) SET platform_class.name = $emission_target_type
          MERGE (contact)-[:CLASSIFIED_AS {level: $emission_classificationlevel}]->(platform_class)
        }
        """,
        **asdict(observation),
        contact_id=contact_id,
        contact_name=contact_name,
        sensor_id=sensor_id,
        platform_id=platform_id,
        emission_id=emission_id,
        class_id=class_id,
        source_id=source_id,
        source_locator=f"line:{observation.source_line}" if observation.source_line else "log",
    )


def populate_observations_neo4j(observations: Sequence[EmissionObservation], uri: str, user: str, password: str, database: str | None = None) -> None:
    """Populate Neo4j with parsed CMO observations."""

    user, password = _validate_neo4j_credentials(user, password)
    neo4j = importlib.import_module("neo4j")
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        try:
            driver.verify_connectivity()
            session_kwargs = {"database": database} if database else {}
            with driver.session(**session_kwargs) as session:
                create_cmo_observation_schema(session)
                for observation in observations:
                    session.execute_write(_write_observation, observation)
        except neo4j.exceptions.ServiceUnavailable as error:
            raise RuntimeError(_neo4j_connection_error_message(uri, database)) from error
        except neo4j.exceptions.AuthError as error:
            raise RuntimeError(f"Neo4j rejected the credentials for {uri!r}.") from error
    finally:
        driver.close()


def add_cmo_observation_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser("ingest-cmo-observations", help="parse PY_CONTACT_LOG emission observations and ingest them into Neo4j")
    parser.add_argument("--input", required=True, help="CMO Lua history/log file containing PY_CONTACT_LOG lines")
    parser.add_argument("--observations-jsonl", help="optional JSONL output for parsed observations")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--neo4j-database", help="optional Neo4j database name")
    parser.set_defaults(handler=run_cmo_observation_command)


def run_cmo_observation_command(args: argparse.Namespace) -> None:
    with Path(args.input).open("r", encoding="utf-8-sig", errors="replace") as handle:
        observations = list(parse_observations(handle))
    if args.observations_jsonl:
        write_observations_jsonl(observations, args.observations_jsonl)
    populate_observations_neo4j(observations, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
    print(json.dumps({"input": args.input, "observations": len(observations), "neo4j_uri": args.neo4j_uri, "neo4j_database": args.neo4j_database}, indent=2))
