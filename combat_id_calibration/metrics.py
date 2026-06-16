"""Calibration metrics and reliability-bin summaries."""

from __future__ import annotations

import math
from typing import Sequence

_EPS = 1e-15


def calibration_report(probabilities: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 10) -> dict[str, object]:
    """Measure multiclass log loss, Brier score, accuracy, and top-label ECE."""
    if not probabilities or len(probabilities) != len(labels) or bins < 1:
        raise ValueError("probabilities/labels must be non-empty and equal length; bins must be positive")
    width = len(probabilities[0])
    if width < 2:
        raise ValueError("at least two classes are required")
    reliability = [{"count": 0, "confidence_sum": 0.0, "correct": 0} for _ in range(bins)]
    log_loss = brier = correct = 0.0
    for row, label in zip(probabilities, labels):
        if len(row) != width or label < 0 or label >= width or any(value < 0 or not math.isfinite(value) for value in row):
            raise ValueError("invalid probability row or label")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("each probability row must sum to one")
        prediction = max(range(width), key=row.__getitem__)
        confidence = row[prediction]
        hit = int(prediction == label)
        correct += hit
        log_loss -= math.log(max(row[label], _EPS))
        brier += sum((value - int(index == label)) ** 2 for index, value in enumerate(row))
        bucket = min(int(confidence * bins), bins - 1)
        reliability[bucket]["count"] += 1
        reliability[bucket]["confidence_sum"] += confidence
        reliability[bucket]["correct"] += hit
    details = []
    ece = 0.0
    for index, bucket in enumerate(reliability):
        count = bucket["count"]
        if not count:
            continue
        mean_confidence = bucket["confidence_sum"] / count
        observed_accuracy = bucket["correct"] / count
        ece += count / len(labels) * abs(mean_confidence - observed_accuracy)
        details.append({"lower": index / bins, "upper": (index + 1) / bins, "count": count, "mean_confidence": mean_confidence, "observed_accuracy": observed_accuracy})
    return {"samples": len(labels), "accuracy": correct / len(labels), "log_loss": log_loss / len(labels), "brier_score": brier / len(labels), "expected_calibration_error": ece, "reliability_bins": details}
