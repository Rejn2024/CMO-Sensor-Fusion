"""Command-line interface for fitting, applying, and evaluating calibration."""

import argparse
import json
from pathlib import Path

from .calibrator import TemperatureCalibrator, _softmax
from .io import read_examples, write_jsonl
from .metrics import calibration_report
from .graph_ingest import add_ingest_parser, run_ingest_command
from .cmo_observation_ingest import add_cmo_observation_parser


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Graph DB-supported combat identification scores")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit a temperature from labelled CMO outcomes")
    fit.add_argument("input"); fit.add_argument("model")
    apply = commands.add_parser("apply", help="add calibrated candidate probabilities")
    apply.add_argument("model"); apply.add_argument("input"); apply.add_argument("output")
    evaluate = commands.add_parser("evaluate", help="compare raw softmax and calibrated probabilities")
    evaluate.add_argument("model"); evaluate.add_argument("input"); evaluate.add_argument("--bins", type=int, default=10)
    add_ingest_parser(commands)
    add_cmo_observation_parser(commands)
    args = parser.parse_args()
    if args.command == "fit":
        classes, logits, labels, _ = read_examples(args.input)
        model = TemperatureCalibrator.fit(logits, labels, classes)
        Path(args.model).write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    elif args.command == "apply":
        model = TemperatureCalibrator.from_dict(json.loads(Path(args.model).read_text(encoding="utf-8")))
        classes, logits, _, records = read_examples(args.input, require_truth=False)
        if classes != list(model.classes): raise ValueError("input classes differ from model classes")
        for record, probabilities in zip(records, model.calibrate_many(logits)):
            record["calibrated_probabilities"] = probabilities
            record["combat_id"] = max(probabilities, key=probabilities.get)
            record["combat_id_probability"] = probabilities[record["combat_id"]]
        write_jsonl(records, args.output)
    elif args.command == "evaluate":
        model = TemperatureCalibrator.from_dict(json.loads(Path(args.model).read_text(encoding="utf-8")))
        classes, logits, labels, _ = read_examples(args.input)
        if classes != list(model.classes): raise ValueError("input classes differ from model classes")
        result = {"temperature": model.temperature, "raw": calibration_report([_softmax(row, 1.0) for row in logits], labels, args.bins), "calibrated": calibration_report([list(row.values()) for row in model.calibrate_many(logits)], labels, args.bins)}
        print(json.dumps(result, indent=2))
    elif args.command == "ingest-graph":
        run_ingest_command(args)
    else:
        args.handler(args)

if __name__ == "__main__":
    main()
