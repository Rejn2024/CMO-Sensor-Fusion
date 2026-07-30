# Passing Combat-Identification Results to an LLM

## Purpose and boundary

The LLM is the final **explanation layer** in the combat-identification
pipeline:

```text
CMO observations
  -> graph evidence and deterministic features
  -> probability model and optional temperature calibration
  -> explanation-payload builder
  -> external LLM caller
  -> human-readable interpretation
```

The probability model, not the LLM, decides the probability of each candidate.
The LLM's role is limited to communicating those supplied results, the evidence
for and against them, their limitations, and useful follow-up collection. It
must not recalculate, replace, or invent probabilities.

This repository currently implements the probability output and the
explanation-payload builder. It does **not** call an LLM inference API. The
generated `llm_prompt` is the hand-off point: an integrating application must
submit that string to its chosen LLM and capture the response.

## End-to-end hand-off process

### 1. Produce one probability record per contact and observation time

The `probability-model` command reads JSON Lines (JSONL) feature records. Each
input line represents a competing platform hypothesis. Records are grouped by
the tuple `scenario_id`, `contact_id`, and `observation_time` before scores are
converted into a probability distribution.

```bash
python -m combat_id_calibration probability-model \
  feature-records.jsonl probability-assignments.jsonl \
  --model calibration-model.json
```

When `--model` is supplied, its learned temperature is applied to the candidate
logits. Without it, the command uses an uncalibrated softmax baseline intended
for development, identified in the output as
`softmax_uncalibrated_baseline`.

The resulting JSONL contains one object for each contact/time group. That
object includes:

| Field | Meaning |
| --- | --- |
| `schema` | Probability-record schema identifier (`platform_operator_nation_probability_v1`). |
| `scenario_id`, `contact_id`, `observation_time` | Traceability back to the observation. |
| `top_platform`, `top_platform_probability` | Highest-probability platform and its model-produced probability. |
| `top_operator_nation`, `top_operator_nation_probability` | Highest-probability *positive* operator-nation identification. Labels such as `unknown` are not selected as the displayed top identification. |
| `platform_probabilities` | Complete platform-level distribution. Repeated candidate rows for a platform are summed. |
| `operator_nation_probabilities` | Complete operator-nation distribution, including non-identifying labels if present. |
| `candidates` | Candidate-row details: platform, operator nation, probability, adjusted and base logits, evidence-query ID, candidate index, and optional distance evidence. |
| `calibration` | Calibration method/model metadata. |

These probability records are the authoritative numerical result. The
explanation phase must preserve them.

### 2. Convert probability records to explanation payloads

Run the explainer-preparation command:

```bash
python -m combat_id_calibration explain \
  probability-assignments.jsonl explanation-payloads.jsonl
```

The command reads each probability JSONL object, calls
`build_explanation_payload`, and writes one explanation JSON object per line.
It is deterministic and makes no network or model request.

For programmatic use, the builder also accepts a separate evidence mapping:

```python
from combat_id_calibration.llm_explainer import build_explanation_payload

payload = build_explanation_payload(
    probability_record,
    {
        "supporting_evidence": [
            {"text": "N001-family emission matched Su-27SM references", "source": "graph:q1"}
        ],
        "contradicting_evidence": [
            "Observed range conflicts with the leading hypothesis"
        ],
        "missing_evidence": ["Collect another emitter scan"],
    },
)
```

The separately supplied evidence takes precedence over same-named evidence
lists embedded in the probability record. The CLI does not load a separate
evidence file, so evidence must already be present on its input probability
records if it is to appear when using the CLI alone.

### 3. Submit `llm_prompt` through an external integration

The explanation payload is an auditable envelope. The exact content intended
for submission to the LLM is the payload's `llm_prompt` string. A caller should:

1. Read an object from `explanation-payloads.jsonl`.
2. Submit `llm_prompt` as the user prompt (optionally with a deployment-specific
   system message that reinforces the non-modification rule).
3. Store the LLM response with `scenario_id`, `contact_id`, and
   `observation_time` so it remains associated with the source result.
4. Preserve the explanation payload and authoritative probability record for
   audit. Do not parse new probabilities out of the prose and feed them back
   into the model result.

Model selection, authentication, transport, retry behavior, response storage,
and presentation are deliberately outside the implementation in this
repository.

## What is passed to the LLM

The constructed prompt is plain UTF-8 text with labeled sections. It contains:

1. An instruction to explain the calibrated probabilities **without changing
   them**.
2. A fixed summary naming the most likely platform and positive operator nation
   with their probabilities, rounded to three decimal places for display.
3. The complete platform distribution serialized as an inline JSON object.
4. The operator-nation distribution serialized as inline JSON after removing
   configured non-identifying labels such as `unknown`, `unk`, `n/a`, `none`,
   `null`, `not specified`, `unspecified`, and blank values.
5. Numbered supporting evidence, including a source or evidence-query ID in
   parentheses where provided.
6. Numbered contradicting evidence.
7. Numbered missing evidence or recommended collection. If none is supplied, a
   default request for more emissions, kinematics, IFF, location, and source
   reliability evidence is inserted.

Evidence entries may be strings or JSON objects. For an object, the displayed
text is selected from the first available key in this order: `text`,
`evidence`, `description`, then `id`. Its source annotation is selected from
`source`, then `evidence_query_id`.

The following information remains in the explanation envelope for traceability
but is **not currently included in `llm_prompt`**:

- `scenario_id`, `contact_id`, and `observation_time`;
- the machine-readable `confidence_limits` list;
- the original structured evidence lists; and
- per-candidate logits, candidate indexes, evidence-query references,
  calibration metadata, and other fields from the probability record.

In particular, the builder calculates confidence-limit warnings when the top
platform probability is below `0.60`, when the two leading platform
probabilities differ by less than `0.15`, or when non-identifying
operator-nation labels were excluded. Those warnings are available to a UI or
external caller in `confidence_limits`, but an integration must explicitly add
them to the LLM request if it wants the LLM to discuss the exact warnings.

## Explanation-payload format

Each line of `explanation-payloads.jsonl` follows this logical shape:

```json
{
  "schema": "llm_explainer_payload_v1",
  "scenario_id": "raid-001",
  "contact_id": "C-101",
  "observation_time": "2026-06-15T10:00:00Z",
  "summary": "Most likely emitter platform is Su-27SM with probability 0.720; most likely operator nation is Belarus with probability 0.720.",
  "confidence_limits": [],
  "supporting_evidence": [
    {
      "text": "N001-family emission matched Su-27SM references",
      "source": "graph:q1"
    }
  ],
  "contradicting_evidence": [
    "Observed range conflicts with the leading hypothesis"
  ],
  "missing_evidence": [
    "Collect another emitter scan"
  ],
  "llm_prompt": "Explain the supplied calibrated combat-identification probabilities without changing them.\nMost likely emitter platform is Su-27SM with probability 0.720; most likely operator nation is Belarus with probability 0.720.\nPlatform distribution: {\"MiG-29MT\": 0.28, \"Su-27SM\": 0.72}\nOperator-nation distribution (positive labels only): {\"Belarus\": 0.72, \"Kazakhstan\": 0.28}\nSupporting evidence:\n  1. N001-family emission matched Su-27SM references (graph:q1)\nContradicting evidence:\n  1. Observed range conflicts with the leading hypothesis\nMissing evidence / recommended collection:\n  1. Collect another emitter scan"
}
```

This example illustrates the schema and formatting; its values are not a
bundled operational result.

## Intended human-readable output

The LLM should return concise prose suitable for an analyst or operator. The
interpretation should:

- state the leading platform and operator-nation assessment using the supplied
  probabilities exactly;
- explain which supplied evidence supports that assessment;
- acknowledge supplied contradictions, close alternatives, and missing
  information rather than hiding uncertainty;
- recommend useful additional collection when evidence is missing; and
- distinguish model results from narrative interpretation.

For example, a response may say that the Su-27SM is most likely at `0.720`
because the referenced emission matched, while noting the conflicting range
evidence and recommending another emitter scan. It must not change `0.720`,
claim that the LLM independently calculated it, introduce unsupported evidence,
or present a new confidence score.

The current prompt requests prose but does not impose a response JSON schema,
word count, section headings, or other machine-readable output contract. An
application that requires structured responses should define and validate that
contract in its external LLM integration while retaining the probability model
as the sole authority for numerical probabilities.

## Operational safeguards

- Treat `probability-assignments.jsonl` as the source of truth and the LLM text
  as a derived explanation.
- Use a fitted calibration model in operational workflows; clearly label the
  uncalibrated fallback in development.
- Retain source/evidence-query identifiers so an analyst can verify claims.
- Do not assume an evidence list was sent merely because it exists elsewhere;
  verify the rendered `llm_prompt`.
- Validate the returned prose against the supplied distributions before display
  if numerical fidelity is safety-critical.
- Do not allow the narrative response to overwrite model fields or become a
  training label without a separate review process.
