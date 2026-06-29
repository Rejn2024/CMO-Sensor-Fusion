"""Populate a Neo4j knowledge graph from PDFs and Wikipedia pages via Ollama.

The pipeline is intentionally dependency-light at import time. Runtime ingestion
requires a local Ollama server and optional packages for the selected sources and
sink: ``pypdf`` for PDFs and ``neo4j`` for writing the graph database.
Wikipedia/HTTP page fetching uses the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import importlib
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _emit_diagnostic(enabled: bool, message: str) -> None:
    """Print extraction diagnostics to stderr when enabled."""

    if enabled:
        print(f"[graph-ingest] {message}", file=sys.stderr)


@dataclass(frozen=True)
class SourceDocument:
    """Text extracted from a source that should be mined into graph facts."""

    source_id: str
    source_type: str
    locator: str
    title: str
    text: str


@dataclass(frozen=True)
class ExtractedFact:
    """A normalized entity-relationship fact ready for Neo4j insertion."""

    subject: str
    predicate: str
    object: str
    source_id: str
    source_type: str
    locator: str
    evidence: str = ""
    confidence: float = 0.0


def stable_id(*parts: str) -> str:
    """Create a deterministic short ID for graph nodes and sources."""

    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 500) -> list[str]:
    """Split text into overlapping chunks sized for local LLM extraction."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


class _TextExtractor(html.parser.HTMLParser):
    """Small HTML-to-text extractor for Wikipedia pages."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "table", "sup"}:
            self._skip_depth += 1
        if tag in {"p", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "table", "sup"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            value = data.strip()
            if value:
                self.parts.append(value)

    def text(self) -> str:
        return re.sub(r"\n\s*\n+", "\n\n", " ".join(self.parts))


def read_pdf(path: str | Path) -> SourceDocument:
    """Extract text from a PDF using the optional pypdf package."""

    pypdf = importlib.import_module("pypdf")
    pdf_path = Path(path)
    print(path)
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    return SourceDocument(
        source_id=stable_id("pdf", str(pdf_path.resolve())),
        source_type="pdf",
        locator=str(pdf_path),
        title=pdf_path.stem,
        text=text,
    )


def read_wikipedia(url: str) -> SourceDocument:
    """Fetch and extract readable text from a Wikipedia page URL."""
    print(url)
    request = urllib.request.Request(url, headers={"User-Agent": "CMO-Sensor-Fusion/0.1 graph-ingest"})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    extractor = _TextExtractor()
    extractor.feed(html)
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+-\s+Wikipedia$", "", title_match.group(1).strip()) if title_match else url.rsplit("/", 1)[-1]
    return SourceDocument(
        source_id=stable_id("wikipedia", url),
        source_type="wikipedia",
        locator=url,
        title=title,
        text=extractor.text(),
    )


def _ollama_response_text(data: dict[str, object]) -> str:
    """Return the parseable model text from an Ollama API response."""

    response = data.get("response", "")
    if isinstance(response, str) and response.strip():
        return response

    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content

    # Some reasoning-capable Ollama models place the final JSON in ``thinking``
    # when the request does not disable thinking, while leaving ``response``
    # empty. Treat that field as a compatibility fallback so older notebook runs
    # and model variants still produce extractable facts.
    thinking = data.get("thinking", "")
    if isinstance(thinking, str) and thinking.strip():
        return thinking
    return ""


def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL) -> str:
    """Call Ollama's local generation API and return parseable model text."""

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        return ""
    return _ollama_response_text(data)


def build_extraction_prompt(document: SourceDocument, chunk: str) -> str:
    """Build a broad prompt for extracting auditable graph triples."""

    return f"""You extract a broad, varied set of knowledge-graph facts for a general information evidence graph about aeroplanes and radars.

JSON output contract:
- Your entire response must be exactly one JSON object and nothing else.
- Start the response with {{ and end it with }}.
- Do not include Markdown fences, prose, comments, explanations, chain-of-thought, or reasoning text.
- Do not wrap the JSON object in a string.
- Use double quotes for every JSON key and string value.
- Escape embedded double quotes and control characters inside JSON strings.
- Do not use trailing commas, NaN, Infinity, undefined, null facts, or Python-style literals.
- If the chunk supports no facts, return exactly {{"facts":[]}}.
- Otherwise, return only this schema:
{{"facts":[{{"subject":"entity name","predicate":"RELATION_IN_UPPER_SNAKE_CASE","object":"entity/value","evidence":"short supporting quote or paraphrase","confidence":0.0}}]}}

Extraction goal:
- Capture as much source-supported context as practical from this chunk, not just one headline fact.
- Prefer many small, specific facts over a single generic summary fact.
- Aim for 8-20 diverse facts when the chunk contains enough information; return fewer only when the source chunk is sparse.

Include varied fact types where supported:
- Entity taxonomy and aliases: IS_A, ALSO_KNOWN_AS, VARIANT_OF, PART_OF.
- Platform, sensor, emitter, weapon, subsystem, signature, and track characteristics: HAS_SENSOR, HAS_WEAPON, HAS_SUBSYSTEM, EMITS, DETECTS, HAS_SIGNATURE, HAS_TRACK_FEATURE, HAS_EMITTER_TYPE.
- Operators, manufacturers, organizations, roles, missions, doctrine, and tactics: OPERATED_BY, MANUFACTURED_BY, HAS_ROLE, HAS_MISSION, USES_DOCTRINE, USES_TACTIC.
- Capabilities, limitations, performance, ranges, frequencies, modes, datalinks, and interoperability: HAS_CAPABILITY, HAS_LIMITATION, HAS_RANGE, USES_FREQUENCY, HAS_MODE, USES_DATALINK, INTEROPERATES_WITH.
- Geography, basing, deployment, timeline, conflicts, exercises, and operational context: LOCATED_IN, BASED_AT, DEPLOYED_TO, ENTERED_SERVICE, RETIRED_FROM_SERVICE, USED_IN, PARTICIPATED_IN.
- CMO emission observations: emitter type, latitude/longitude, heading, altitude, speed, sensor/emitter name, classification level, candidate platform identity, aircraft variant, and operator country hypotheses.
- Identification evidence and caveats: SUPPORTS_IDENTIFICATION, CONTRADICTS_IDENTIFICATION, DISTINGUISHES_FROM, INDICATES, DERIVED_FROM.
- Quantitative values and named attributes that help disambiguate entities.

Rules:
- Extract factual relationships useful for identifying platforms, sensors, emitters, weapons, military organizations, roles, capabilities, locations, kinematics, doctrine, operational history, and discriminating context.
- For CMO PY_CONTACT_LOG-style emission rows, preserve emission sensor name and numeric emitter role/type, location, speed/heading/altitude, and any target class as evidence that SUPPORTS_IDENTIFICATION of the emitting platform identity, variant, and operator country.
- Cover different subjects mentioned in the chunk instead of repeatedly describing only the first or most prominent subject.
- Use concise canonical entity names and preserve meaningful model numbers, designations, frequencies, ranges, dates, units, and locations.
- Use upper snake case predicates; prefer the predicates above, but create similarly specific predicates when needed.
- Do not invent facts not supported by the source chunk.
- Do not emit duplicate facts or vague facts whose object is only "information", "context", or "data".
- Use confidence from 0.0 to 1.0 based only on how explicit the source chunk is.

Source title: {document.title}
Source type: {document.source_type}
Source locator: {document.locator}

Text chunk:
{chunk}
"""


def _json_object_candidates(text: str) -> Iterable[str]:
    """Yield balanced top-level JSON object candidates embedded in text."""

    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def _extract_json_object(
    response: str,
    diagnostic: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Return the first facts JSON object from an LLM response, or an empty payload."""

    text = response.strip()
    if not text:
        if diagnostic:
            diagnostic("model response was empty")
        return {"facts": []}
    original_length = len(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if diagnostic and len(text) != original_length:
        diagnostic(f"removed reasoning block(s); parseable response length is {len(text)} characters")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
        if diagnostic:
            diagnostic(f"using fenced JSON block with {len(text)} characters")

    fallback: dict[str, object] | None = None
    candidate_count = 0
    decode_errors = 0
    for candidate in _json_object_candidates(text):
        candidate_count += 1
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as error:
            decode_errors += 1
            if diagnostic:
                diagnostic(f"JSON candidate {candidate_count} failed to decode at char {error.pos}: {error.msg}")
            continue
        if not isinstance(payload, dict):
            if diagnostic:
                diagnostic(f"JSON candidate {candidate_count} decoded to {type(payload).__name__}, not an object")
            continue
        if "facts" in payload:
            if diagnostic:
                facts_value = payload.get("facts")
                fact_count = len(facts_value) if isinstance(facts_value, list) else "non-list"
                diagnostic(f"JSON candidate {candidate_count} contains facts={fact_count}")
            return payload
        if diagnostic:
            diagnostic(f"JSON candidate {candidate_count} has keys {sorted(payload.keys())}, but no 'facts' key")
        if fallback is None:
            fallback = payload
    if diagnostic:
        diagnostic(f"found {candidate_count} JSON object candidate(s), {decode_errors} decode error(s), and no facts object")
    return fallback or {"facts": []}


def parse_extracted_facts(
    response: str,
    document: SourceDocument,
    diagnostic: Callable[[str], None] | None = None,
) -> list[ExtractedFact]:
    """Parse the model's JSON response and normalize fact records."""

    payload = _extract_json_object(response, diagnostic=diagnostic)
    raw_facts = payload.get("facts", [])
    if not isinstance(raw_facts, list):
        if diagnostic:
            diagnostic(f"'facts' value is {type(raw_facts).__name__}, not a list")
        return []

    facts: list[ExtractedFact] = []
    skipped = 0
    for index, item in enumerate(raw_facts, start=1):
        if not isinstance(item, dict):
            skipped += 1
            if diagnostic:
                diagnostic(f"skipping fact {index}: item is {type(item).__name__}, not an object")
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = re.sub(r"[^A-Z0-9_]+", "_", str(item.get("predicate", "RELATED_TO")).upper()).strip("_") or "RELATED_TO"
        obj = str(item.get("object", "")).strip()
        if not subject or not obj:
            skipped += 1
            if diagnostic:
                diagnostic(f"skipping fact {index}: missing subject or object")
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            skipped += 1
            if diagnostic:
                diagnostic(f"skipping fact {index}: confidence is not numeric")
            continue
        facts.append(
            ExtractedFact(
                subject=subject,
                predicate=predicate,
                object=obj,
                evidence=str(item.get("evidence", "")).strip(),
                confidence=confidence,
                source_id=document.source_id,
                source_type=document.source_type,
                locator=document.locator,
            )
        )
    if diagnostic:
        diagnostic(f"normalized {len(facts)} fact(s); skipped {skipped} invalid item(s)")
    return facts


def extract_facts(
    documents: Iterable[SourceDocument],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_chars: int = 2000,
    overlap: int = 200,
    diagnostics: bool = True,
) -> list[ExtractedFact]:
    """Extract graph facts from source documents with a local Ollama model."""

    facts: list[ExtractedFact] = []
    document_count = 0
    chunk_count = 0
    _emit_diagnostic(
        diagnostics,
        f"starting fact extraction with model={model!r}, ollama_url={ollama_url!r}, max_chars={max_chars}, overlap={overlap}",
    )
    for document in documents:
        document_count += 1
        chunks = chunk_text(document.text, max_chars=max_chars, overlap=overlap)
        _emit_diagnostic(
            diagnostics,
            (
                f"document {document_count}: title={document.title!r}, source_id={document.source_id}, "
                f"type={document.source_type}, text_chars={len(document.text)}, chunks={len(chunks)}"
            ),
        )
        if not chunks:
            _emit_diagnostic(diagnostics, f"document {document_count} produced no chunks; check source text extraction")
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_count += 1
            _emit_diagnostic(
                diagnostics,
                f"document {document_count} chunk {chunk_index}/{len(chunks)}: sending {len(chunk)} chars to Ollama",
            )
            response = ollama_generate(build_extraction_prompt(document, chunk), model=model, ollama_url=ollama_url)
            preview = response[:300].replace("\n", "\\n")
            _emit_diagnostic(
                diagnostics,
                f"document {document_count} chunk {chunk_index}/{len(chunks)}: received {len(response)} chars; preview={preview!r}",
            )
            before = len(facts)
            facts.extend(
                parse_extracted_facts(
                    response,
                    document,
                    diagnostic=(
                        lambda message, doc_index=document_count, current_chunk=chunk_index, total_chunks=len(chunks): _emit_diagnostic(
                            diagnostics, f"document {doc_index} chunk {current_chunk}/{total_chunks}: {message}"
                        )
                    ),
                )
            )
            _emit_diagnostic(
                diagnostics,
                f"document {document_count} chunk {chunk_index}/{len(chunks)}: added {len(facts) - before} fact(s)",
            )
    _emit_diagnostic(diagnostics, f"finished extraction: documents={document_count}, chunks={chunk_count}, facts={len(facts)}")
    return facts


def write_facts_jsonl(facts: Sequence[ExtractedFact], path: str | Path) -> None:
    """Persist extracted facts for review before graph ingestion."""

    rows = [fact.__dict__ for fact in facts]
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def create_neo4j_schema(session: object) -> None:
    """Create the minimal Neo4j constraints used by this pipeline."""

    statements = [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
    ]
    for statement in statements:
        session.run(statement)


def _write_fact(tx: object, fact: ExtractedFact) -> None:
    """Write one extracted fact to Neo4j inside a transaction."""

    subject_id = stable_id("entity", fact.subject.lower())
    object_id = stable_id("entity", fact.object.lower())
    tx.run(
        """
        MERGE (subject:Entity {id: $subject_id})
          SET subject.name = $subject
        MERGE (object:Entity {id: $object_id})
          SET object.name = $object
        MERGE (source:Source {id: $source_id})
          SET source.source_type = $source_type,
              source.locator = $locator
        CREATE (subject)-[:FACT {
            predicate: $predicate,
            source_id: $source_id,
            evidence: $evidence,
            confidence: $confidence
        }]->(object)
        MERGE (subject)-[:MENTIONED_IN]->(source)
        MERGE (object)-[:MENTIONED_IN]->(source)
        """,
        subject_id=subject_id,
        object_id=object_id,
        subject=fact.subject,
        object=fact.object,
        source_id=fact.source_id,
        source_type=fact.source_type,
        locator=fact.locator,
        predicate=fact.predicate,
        evidence=fact.evidence,
        confidence=fact.confidence,
    )


def _validate_neo4j_credentials(user: str | None, password: str | None) -> tuple[str, str]:
    """Validate Neo4j credentials before building the driver auth token."""

    username = "" if user is None else str(user).strip()
    if not username:
        raise ValueError("Neo4j username is required; set --neo4j-user or NEO4J_USER before ingestion.")
    if password is None or not str(password):
        raise ValueError(
            "Neo4j password is required; set --neo4j-password or NEO4J_PASSWORD before ingestion. "
            "In the notebook, replace the placeholder empty string with your Neo4j password."
        )
    return username, str(password)


def _neo4j_connection_error_message(uri: str, database: str | None = None) -> str:
    """Build actionable guidance for Neo4j connection failures."""

    database_text = f" database {database!r}" if database else ""
    return (
        f"Unable to connect to Neo4j at {uri!r}{database_text}. "
        "Make sure a Neo4j server is running and reachable over Bolt, the URI/port are correct, "
        "and your username/password match the configured database. For a local default setup, "
        "start Neo4j Desktop or run `docker run --rm -p 7474:7474 -p 7687:7687 "
        "-e NEO4J_AUTH=neo4j/<password> neo4j:latest`, then retry with "
        "`bolt://127.0.0.1:7687` if `localhost` resolves incorrectly."
    )


def populate_neo4j(
    facts: Sequence[ExtractedFact],
    uri: str,
    user: str,
    password: str,
    database: str | None = None,
) -> None:
    """Populate a Neo4j database with extracted facts."""

    user, password = _validate_neo4j_credentials(user, password)
    neo4j = importlib.import_module("neo4j")
    service_unavailable = neo4j.exceptions.ServiceUnavailable
    auth_error = neo4j.exceptions.AuthError
    driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    try:
        try:
            driver.verify_connectivity()
            session_kwargs = {"database": database} if database else {}
            with driver.session(**session_kwargs) as session:
                create_neo4j_schema(session)
                for fact in facts:
                    session.execute_write(_write_fact, fact)
        except service_unavailable as error:
            raise RuntimeError(_neo4j_connection_error_message(uri, database)) from error
        except auth_error as error:
            raise RuntimeError(
                f"Neo4j rejected the credentials for {uri!r}. Check the --neo4j-user/--neo4j-password "
                "values or reset the database password, then retry ingestion."
            ) from error
    finally:
        driver.close()


def load_documents(pdf_paths: Sequence[str], wikipedia_urls: Sequence[str]) -> list[SourceDocument]:
    """Load all configured PDF and Wikipedia documents."""

    return [read_pdf(path) for path in pdf_paths] + [read_wikipedia(url) for url in wikipedia_urls]


def add_ingest_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the graph-ingestion CLI command."""

    parser = commands.add_parser("ingest-graph", help="extract facts with local Ollama and populate a Neo4j graph")
    parser.add_argument("--pdf", action="append", default=[], help="PDF path to ingest; may be repeated")
    parser.add_argument("--wikipedia", action="append", default=[], help="Wikipedia page URL to ingest; may be repeated")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--neo4j-database", help="optional Neo4j database name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="local Ollama model name")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--facts-jsonl", help="optional JSONL file for extracted facts")
    parser.add_argument("--max-chars", type=int, default=6000, help="maximum characters per LLM chunk")
    parser.add_argument("--overlap", type=int, default=500, help="overlap characters between chunks")
    parser.add_argument("--quiet-extraction", action="store_true", help="suppress fact-extraction diagnostics on stderr")


def run_ingest_command(args: argparse.Namespace) -> None:
    """Run graph ingestion from parsed CLI arguments."""

    documents = load_documents(args.pdf, args.wikipedia)
    if not documents:
        raise ValueError("provide at least one --pdf or --wikipedia source")
    facts = extract_facts(
        documents,
        model=args.model,
        ollama_url=args.ollama_url,
        max_chars=args.max_chars,
        overlap=args.overlap,
        diagnostics=not args.quiet_extraction,
    )
    if args.facts_jsonl:
        write_facts_jsonl(facts, args.facts_jsonl)
    populate_neo4j(facts, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
    print(json.dumps({"documents": len(documents), "facts": len(facts), "neo4j_uri": args.neo4j_uri, "neo4j_database": args.neo4j_database}, indent=2))
