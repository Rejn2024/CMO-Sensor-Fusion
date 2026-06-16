import json
import subprocess
import sys

import pytest

from combat_id_calibration import TemperatureCalibrator, calibration_report


def test_temperature_scaling_reduces_overconfident_log_loss():
    logits = [[6, 0], [6, 0], [6, 0], [0, 6]]
    labels = [0, 0, 1, 1]
    raw = [list(TemperatureCalibrator(("friendly", "hostile")).calibrate(row).values()) for row in logits]
    fitted = TemperatureCalibrator.fit(logits, labels, ["friendly", "hostile"])
    calibrated = [list(item.values()) for item in fitted.calibrate_many(logits)]
    assert fitted.temperature > 1
    assert calibration_report(calibrated, labels)["log_loss"] < calibration_report(raw, labels)["log_loss"]
    assert all(sum(row) == pytest.approx(1) for row in calibrated)


def test_cli_fit_apply_evaluate(tmp_path):
    source = tmp_path / "observations.jsonl"
    records = [
        {"scenario_id": "s1", "contact_id": "c1", "scores": {"friendly": 3, "hostile": 0}, "truth": "friendly"},
        {"scenario_id": "s2", "contact_id": "c2", "scores": {"friendly": 3, "hostile": 0}, "truth": "hostile"},
        {"scenario_id": "s3", "contact_id": "c3", "scores": {"friendly": 0, "hostile": 3}, "truth": "hostile"},
    ]
    source.write_text("".join(json.dumps(x) + "\n" for x in records))
    model, output = tmp_path / "model.json", tmp_path / "output.jsonl"
    subprocess.run([sys.executable, "-m", "combat_id_calibration", "fit", str(source), str(model)], check=True)
    subprocess.run([sys.executable, "-m", "combat_id_calibration", "apply", str(model), str(source), str(output)], check=True)
    report = subprocess.run([sys.executable, "-m", "combat_id_calibration", "evaluate", str(model), str(source)], check=True, text=True, capture_output=True)
    assert "calibrated_probabilities" in json.loads(output.read_text().splitlines()[0])
    assert "expected_calibration_error" in json.loads(report.stdout)["calibrated"]
