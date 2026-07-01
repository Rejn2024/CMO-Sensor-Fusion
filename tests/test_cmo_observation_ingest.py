from combat_id_calibration.cmo_observation_ingest import (
    _write_observation,
    create_cmo_observation_schema,
    parse_observation_line,
    parse_observations,
)


SAMPLE = "PY_CONTACT_LOG  Time : 1844772240 , Sensor_aircraft : Typhoon FGR.4 , Emission_sensor_name : Slot Back [N-010 Zhuk-M] , Emission_age : 33.900054931641 , Emission_solid : true , Emission_type : 2001 , Emission_role : 2122 , Emission_latitude : 44.640933862914 , Emission_longitude : 31.890743606263 , Emission_heading : 331.40036010742 , Emission_altitude : 10316.349609375 , Emission_speed : 479.64691162109 , Emission_target_type : Type: Multirole (Fighter/Attack) , Emission_classificationlevel : 2"


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, statement, **parameters):
        self.calls.append((statement, parameters))


def test_parse_observation_line_extracts_typed_fields():
    observation = parse_observation_line(SAMPLE, source_line=7)

    assert observation is not None
    assert observation.time == 1844772240
    assert observation.sensor_aircraft == "Typhoon FGR.4"
    assert observation.emission_sensor_name == "Slot Back [N-010 Zhuk-M]"
    assert observation.emission_solid is True
    assert observation.emission_type == 2001
    assert observation.emission_latitude == 44.640933862914
    assert observation.emission_target_type == "Type: Multirole (Fighter/Attack)"
    assert observation.emission_classificationlevel == 2
    assert observation.source_line == 7


def test_parse_observations_ignores_non_observation_lines():
    observations = list(parse_observations(["noise", SAMPLE]))

    assert len(observations) == 1
    assert observations[0].source_line == 2


def test_create_cmo_observation_schema_uses_typed_constraints():
    session = RecordingRunner()
    create_cmo_observation_schema(session)

    statements = [call[0] for call in session.calls]
    assert "CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE" in statements
    assert "CREATE CONSTRAINT platform_class_id IF NOT EXISTS FOR (pc:PlatformClass) REQUIRE pc.id IS UNIQUE" in statements
    assert "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE" in statements


def test_write_observation_emits_phase_two_ontology_cypher():
    observation = parse_observation_line(SAMPLE, source_line=1)
    tx = RecordingRunner()

    _write_observation(tx, observation)

    assert len(tx.calls) == 1
    statement, parameters = tx.calls[0]
    assert "MERGE (obs:Observation" in statement
    assert "MERGE (contact:Contact" in statement
    assert "MERGE (sensor:Sensor" in statement
    assert "MERGE (emission:Emission" in statement
    assert "HAS_OBSERVATION" in statement
    assert "OBSERVED_BY" in statement
    assert "EMITTED" in statement
    assert "DETECTS" in statement
    assert "MERGE (platform)-[:HAS_SENSOR]->(sensor)" not in statement
    assert "CLASSIFIED_AS" in statement
    assert parameters["sensor_aircraft"] == "Typhoon FGR.4"
    assert parameters["emission_sensor_name"] == "Slot Back [N-010 Zhuk-M]"
    assert parameters["emission_speed"] == 479.64691162109
