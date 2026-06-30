"""Evidence-grounded explanation payloads for calibrated combat-ID outputs.

The functions here prepare deterministic text and optional LLM prompts from
model probabilities and graph evidence.  They intentionally never alter model
probabilities; any LLM is asked only to explain the supplied numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .io import write_jsonl
from .probability_model import _is_unknown_operator_nation, _top_positive_operator_nation

EXPLANATION_SCHEMA = "llm_explainer_payload_v1"


def _items(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _evidence_lines(items: Sequence[object], prefix: str) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            text = item.get("text") or item.get("evidence") or item.get("description") or item.get("id")
            source = item.get("source") or item.get("evidence_query_id")
            suffix = f" ({source})" if source else ""
            lines.append(f"{prefix}{index}. {text}{suffix}")
        else:
            lines.append(f"{prefix}{index}. {item}")
    return lines


def build_explanation_payload(probability_record: Mapping[str, object], evidence: Mapping[str, object] | None = None) -> dict[str, object]:
    """Build an explanation record from calibrated probabilities and evidence."""

    evidence = evidence or {}
    top_platform = probability_record.get("top_platform")
    top_platform_probability = float(probability_record.get("top_platform_probability", 0.0))
    operator_nation_probabilities = probability_record.get(
        "operator_nation_probabilities", probability_record.get("country_probabilities", {})
    )
    if not isinstance(operator_nation_probabilities, Mapping):
        operator_nation_probabilities = {}
    positive_operator_nation_probabilities = {
        str(operator_nation): float(probability)
        for operator_nation, probability in operator_nation_probabilities.items()
        if not _is_unknown_operator_nation(operator_nation)
    }
    recorded_top_operator_nation = probability_record.get("top_operator_nation") or probability_record.get("top_country_of_origin")
    if recorded_top_operator_nation and not _is_unknown_operator_nation(recorded_top_operator_nation):
        top_operator_nation = recorded_top_operator_nation
        top_operator_nation_probability = float(
            probability_record.get("top_operator_nation_probability", probability_record.get("top_country_probability", 0.0))
        )
    else:
        top_operator_nation, top_operator_nation_probability = _top_positive_operator_nation(positive_operator_nation_probabilities)
    support = _items(evidence.get("supporting_evidence") or probability_record.get("supporting_evidence"))
    contradict = _items(evidence.get("contradicting_evidence") or probability_record.get("contradicting_evidence"))
    missing = _items(evidence.get("missing_evidence") or probability_record.get("missing_evidence"))
    uncertainty: list[str] = []
    if top_platform_probability < 0.6:
        uncertainty.append("Top platform probability is below 0.60, so the assignment should be treated as tentative.")
    if len(probability_record.get("platform_probabilities", {}) or {}) > 1:
        probs = sorted((probability_record.get("platform_probabilities") or {}).values(), reverse=True)
        if len(probs) > 1 and float(probs[0]) - float(probs[1]) < 0.15:
            uncertainty.append("The two leading platform hypotheses are separated by less than 0.15 probability.")
    if len(positive_operator_nation_probabilities) != len(operator_nation_probabilities):
        uncertainty.append(
            "Non-identifying operator-nation labels were excluded; the summary uses the highest-probability positive operator-nation candidate."
        )
    summary = (
        f"Most likely emitter platform is {top_platform} with probability {top_platform_probability:.3f}; "
        f"most likely operator nation is {top_operator_nation} with probability {top_operator_nation_probability:.3f}."
    )
    prompt_lines = [
        "Explain the supplied calibrated combat-identification probabilities without changing them.",
        summary,
        f"Platform distribution: {json.dumps(probability_record.get('platform_probabilities', {}), sort_keys=True)}",
        f"Operator-nation distribution (positive labels only): {json.dumps(positive_operator_nation_probabilities, sort_keys=True)}",
        "Supporting evidence:",
        *(_evidence_lines(support, "  ") or ["  none supplied"]),
        "Contradicting evidence:",
        *(_evidence_lines(contradict, "  ") or ["  none supplied"]),
        "Missing evidence / recommended collection:",
        *(_evidence_lines(missing, "  ") or ["  request additional emissions, kinematics, IFF, location, and source reliability evidence"]),
    ]
    return {
        "schema": EXPLANATION_SCHEMA,
        "scenario_id": probability_record.get("scenario_id", ""),
        "contact_id": probability_record.get("contact_id", ""),
        "observation_time": probability_record.get("observation_time", ""),
        "summary": summary,
        "confidence_limits": uncertainty,
        "supporting_evidence": support,
        "contradicting_evidence": contradict,
        "missing_evidence": missing,
        "llm_prompt": "\n".join(prompt_lines),
    }


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_explainer(input_path: str | Path, output_path: str | Path) -> list[dict[str, object]]:
    records = [build_explanation_payload(record) for record in read_jsonl(input_path)]
    write_jsonl(records, output_path)
    return records


def add_llm_explainer_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = commands.add_parser("explain", help="prepare LLM explanation payloads from calibrated probability records")
    parser.add_argument("input", help="JSONL probability assignments")
    parser.add_argument("output", help="JSONL explanation payloads/prompts")
    parser.set_defaults(handler=run_llm_explainer_command)


def run_llm_explainer_command(args: argparse.Namespace) -> None:
    records = run_explainer(args.input, args.output)
    print(json.dumps({"input": args.input, "output": args.output, "explanations": len(records)}, indent=2))
