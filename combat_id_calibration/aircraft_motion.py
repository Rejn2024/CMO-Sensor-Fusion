"""Parse CMO contact logs and visualize aircraft motion in three dimensions.

The parser consumes the ``PY_CONTACT_LOG`` records emitted by
``event_export_lua_02.lua``.  Plotting dependencies are imported lazily so the
core calibration package remains dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import cos, radians, sin
from pathlib import Path
from typing import Iterable, Sequence


EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class AircraftSample:
    """One contact kinematic sample exported by CMO."""

    time: str
    sensor_aircraft: str
    latitude: float
    longitude: float
    heading_deg: float
    altitude_m: float
    speed_kts: float
    target_type: str = ""


def _parse_record(line: str) -> dict[str, str] | None:
    marker = "PY_CONTACT_LOG"
    start = line.find(marker)
    if start < 0:
        return None
    payload = line[start + len(marker) :].strip()
    fields: dict[str, str] = {}
    for item in payload.split(" , "):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_contact_lines(lines: Iterable[str], *, deduplicate: bool = True) -> list[AircraftSample]:
    """Parse Lua-history lines into ordered aircraft samples.

    CMO writes one line per detected emission, so several records can contain
    identical kinematics.  Exact kinematic duplicates are removed by default.
    Malformed/non-contact lines are ignored.
    """

    samples: list[AircraftSample] = []
    seen: set[tuple[object, ...]] = set()
    required = {
        "Time",
        "Sensor_aircraft",
        "Emission_latitude",
        "Emission_longitude",
        "Emission_heading",
        "Emission_altitude",
        "Emission_speed",
    }
    for line in lines:
        values = _parse_record(line)
        if values is None or not required.issubset(values):
            continue
        try:
            sample = AircraftSample(
                time=values["Time"],
                sensor_aircraft=values["Sensor_aircraft"],
                latitude=float(values["Emission_latitude"]),
                longitude=float(values["Emission_longitude"]),
                heading_deg=float(values["Emission_heading"]) % 360.0,
                altitude_m=float(values["Emission_altitude"]),
                speed_kts=float(values["Emission_speed"]),
                target_type=values.get("Emission_target_type", ""),
            )
        except (TypeError, ValueError):
            continue
        identity = (
            sample.time,
            sample.sensor_aircraft,
            sample.latitude,
            sample.longitude,
            sample.heading_deg,
            sample.altitude_m,
            sample.speed_kts,
            sample.target_type,
        )
        if not deduplicate or identity not in seen:
            samples.append(sample)
            seen.add(identity)
    return sorted(samples, key=lambda row: (_time_key(row.time), row.sensor_aircraft, row.target_type))


def _time_key(value: str) -> tuple[int, object]:
    try:
        return (0, float(value))
    except ValueError:
        pass
    try:
        return (1, datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return (2, value)


def load_contact_log(path: str | Path, *, deduplicate: bool = True) -> list[AircraftSample]:
    """Load ``PY_CONTACT_LOG`` samples from a CMO Lua-history text file."""

    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        return parse_contact_lines(stream, deduplicate=deduplicate)


def to_local_coordinates(
    samples: Sequence[AircraftSample],
    *,
    origin: tuple[float, float] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Convert latitude/longitude to local east/north distances in kilometres."""

    if not samples:
        return [], [], []
    lat0, lon0 = origin or (samples[0].latitude, samples[0].longitude)
    north: list[float] = []
    east: list[float] = []
    altitude: list[float] = []
    for sample in samples:
        north.append(EARTH_RADIUS_M * radians(sample.latitude - lat0) / 1000.0)
        east.append(
            EARTH_RADIUS_M
            * cos(radians(lat0))
            * radians(sample.longitude - lon0)
            / 1000.0
        )
        altitude.append(sample.altitude_m / 1000.0)
    return east, north, altitude


def plot_aircraft_motion(
    samples: Sequence[AircraftSample],
    *,
    title: str = "CMO aircraft contact motion",
    arrow_stride: int = 1,
    arrow_length_km: float = 2.0,
    ax=None,
):
    """Plot a speed-coloured 3D trajectory and heading arrows with Matplotlib.

    Returns ``(figure, axes, scatter)``.  Heading follows the aviation
    convention: zero degrees points north and 90 degrees points east.
    """

    if not samples:
        raise ValueError("at least one aircraft sample is required")
    if arrow_stride < 1:
        raise ValueError("arrow_stride must be at least 1")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the visualization extra: pip install -e '.[visualization]'") from exc

    east, north, altitude = to_local_coordinates(samples)
    speeds = [sample.speed_kts for sample in samples]
    if ax is None:
        figure = plt.figure(figsize=(11, 7))
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure
    ax.plot(east, north, altitude, color="0.45", linewidth=1.5, alpha=0.8)
    scatter = ax.scatter(east, north, altitude, c=speeds, cmap="viridis", s=45)

    indices = range(0, len(samples), arrow_stride)
    u = [sin(radians(samples[index].heading_deg)) for index in indices]
    v = [cos(radians(samples[index].heading_deg)) for index in indices]
    arrow_indices = range(0, len(samples), arrow_stride)
    ax.quiver(
        [east[index] for index in arrow_indices],
        [north[index] for index in arrow_indices],
        [altitude[index] for index in arrow_indices],
        u,
        v,
        [0.0] * len(u),
        length=arrow_length_km,
        normalize=True,
        color="tab:red",
        arrow_length_ratio=0.25,
    )
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.1, shrink=0.75)
    colorbar.set_label("Speed (knots)")
    ax.set(xlabel="East (km)", ylabel="North (km)", zlabel="Altitude (km)", title=title)
    return figure, ax, scatter
