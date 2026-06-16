"""Multiclass temperature scaling without third-party dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

_EPS = 1e-15


def _softmax(logits: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be a positive finite number")
    scaled = [float(value) / temperature for value in logits]
    pivot = max(scaled)
    exps = [math.exp(value - pivot) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def _validate(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> int:
    if not logits or len(logits) != len(labels):
        raise ValueError("logits and labels must be non-empty and have equal length")
    width = len(logits[0])
    if width < 2 or any(len(row) != width for row in logits):
        raise ValueError("every logits row must contain the same two or more classes")
    if any(label < 0 or label >= width for label in labels):
        raise ValueError("label index is outside the class range")
    if any(not math.isfinite(float(value)) for row in logits for value in row):
        raise ValueError("logits must be finite")
    return width


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    return -sum(math.log(max(_softmax(row, temperature)[label], _EPS)) for row, label in zip(logits, labels)) / len(labels)


@dataclass(frozen=True)
class TemperatureCalibrator:
    """A fitted scalar temperature for preserving candidate ranking while calibrating confidence."""

    classes: tuple[str, ...]
    temperature: float = 1.0

    @classmethod
    def fit(
        cls,
        logits: Sequence[Sequence[float]],
        labels: Sequence[int],
        classes: Sequence[str],
        minimum: float = 0.05,
        maximum: float = 20.0,
        iterations: int = 120,
    ) -> "TemperatureCalibrator":
        """Fit temperature by deterministic golden-section minimization of log loss."""
        width = _validate(logits, labels)
        if len(classes) != width or len(set(classes)) != width:
            raise ValueError("classes must be unique and match logits width")
        if minimum <= 0 or maximum <= minimum or iterations < 1:
            raise ValueError("invalid temperature search bounds or iteration count")
        # Search log-temperature so narrow and broad temperatures receive equal resolution.
        left, right = math.log(minimum), math.log(maximum)
        ratio = (math.sqrt(5) - 1) / 2
        c = right - ratio * (right - left)
        d = left + ratio * (right - left)
        for _ in range(iterations):
            if _nll(logits, labels, math.exp(c)) < _nll(logits, labels, math.exp(d)):
                right, d = d, c
                c = right - ratio * (right - left)
            else:
                left, c = c, d
                d = left + ratio * (right - left)
        return cls(tuple(classes), math.exp((left + right) / 2))

    def calibrate(self, logits: Sequence[float]) -> dict[str, float]:
        if len(logits) != len(self.classes):
            raise ValueError("logits width does not match fitted classes")
        return dict(zip(self.classes, _softmax(logits, self.temperature)))

    def calibrate_many(self, logits: Iterable[Sequence[float]]) -> list[dict[str, float]]:
        return [self.calibrate(row) for row in logits]

    def to_dict(self) -> dict[str, object]:
        return {"method": "temperature_scaling", "classes": list(self.classes), "temperature": self.temperature}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TemperatureCalibrator":
        if value.get("method") != "temperature_scaling":
            raise ValueError("unsupported calibration method")
        classes = value.get("classes")
        temperature = value.get("temperature")
        if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
            raise ValueError("model classes must be a string list")
        if not isinstance(temperature, (int, float)):
            raise ValueError("model temperature must be numeric")
        return cls(tuple(classes), float(temperature))
