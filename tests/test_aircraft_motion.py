import math

import pytest

from combat_id_calibration.aircraft_motion import (
    AircraftSample,
    parse_contact_lines,
    to_local_coordinates,
)


LINE = (
    "noise PY_CONTACT_LOG  Time : 100 , Sensor_aircraft : Typhoon FGR.4 , "
    "Emission_sensor_name : Radar , Emission_latitude : 51.0 , "
    "Emission_longitude : -1.0 , Emission_heading : 370 , "
    "Emission_altitude : 9000 , Emission_speed : 450 , "
    "Emission_target_type : Aircraft"
)


def test_parse_contact_lines_extracts_and_deduplicates_emission_records():
    samples = parse_contact_lines(["unrelated", LINE, LINE])

    assert samples == [AircraftSample("100", "Typhoon FGR.4", 51.0, -1.0, 10.0, 9000.0, 450.0, "Aircraft")]


def test_parse_contact_lines_ignores_invalid_numeric_values():
    assert parse_contact_lines([LINE.replace("450", "fast")]) == []


def test_local_coordinates_use_east_north_and_altitude_kilometres():
    samples = [
        AircraftSample("0", "sensor", 0.0, 0.0, 0, 1000, 100),
        AircraftSample("1", "sensor", 1.0, 1.0, 90, 2000, 200),
    ]

    east, north, altitude = to_local_coordinates(samples)

    assert east[0] == north[0] == 0.0
    assert east[1] == pytest.approx(111.195, rel=1e-4)
    assert north[1] == pytest.approx(111.195, rel=1e-4)
    assert altitude == [1.0, 2.0]
    assert math.isfinite(east[1])
