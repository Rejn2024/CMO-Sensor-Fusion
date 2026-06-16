"""Populate a Kuzu knowledge graph from PDFs and Wikipedia pages via Ollama.

The pipeline is intentionally dependency-light at import time. Runtime ingestion
requires a local Ollama server and optional packages for the selected sources and
sink: ``pypdf`` for PDFs and ``kuzu`` for writing the graph database.
Wikipedia/HTTP page fetching uses the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import importlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


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
    """A normalized entity-relationship fact ready for Kuzu insertion."""

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


def ollama_generate(prompt: str, model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL) -> str:
    """Call Ollama's local generation API and return the raw response text."""

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response", ""))


def build_extraction_prompt(document: SourceDocument, chunk: str) -> str:
    """Build a constrained prompt for extracting auditable graph triples."""

    return f"""You extract knowledge-graph facts for a combat-identification evidence graph.
Return only valid JSON matching this schema:
{{"facts":[{{"subject":"entity name","predicate":"RELATION_IN_UPPER_SNAKE_CASE","object":"entity/value","evidence":"short supporting quote or paraphrase","confidence":0.0}}]}}

Rules:
- Extract factual relationships useful for identifying platforms, sensors, emitters, weapons, military organizations, roles, capabilities, locations, or doctrine.
- Use concise canonical entity names.
- Use predicates such as IS_A, HAS_SENSOR, HAS_WEAPON, OPERATED_BY, DETECTS, EMITS, LOCATED_IN, SUPPORTS_IDENTIFICATION, CONTRADICTS_IDENTIFICATION, HAS_ROLE, HAS_CAPABILITY, DERIVED_FROM.
- Do not invent facts not supported by the source chunk.
- Use confidence from 0.0 to 1.0 based only on how explicit the source chunk is.

Source title: {document.title}
Source type: {document.source_type}
Source locator: {document.locator}

Text chunk:
{chunk}
"""


def parse_extracted_facts(response: str, document: SourceDocument) -> list[ExtractedFact]:
    """Parse the model's JSON response and normalize fact records."""

    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    payload = json.loads(text)
    facts: list[ExtractedFact] = []
    for item in payload.get("facts", []):
        subject = str(item.get("subject", "")).strip()
        predicate = re.sub(r"[^A-Z0-9_]+", "_", str(item.get("predicate", "RELATED_TO")).upper()).strip("_") or "RELATED_TO"
        obj = str(item.get("object", "")).strip()
        if not subject or not obj:
            continue
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
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
    return facts


def extract_facts(
    documents: Iterable[SourceDocument],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_chars: int = 6000,
    overlap: int = 500,
) -> list[ExtractedFact]:
    """Extract graph facts from source documents with a local Ollama model."""

    facts: list[ExtractedFact] = []
    for document in documents:
        for chunk in chunk_text(document.text, max_chars=max_chars, overlap=overlap):
            response = ollama_generate(build_extraction_prompt(document, chunk), model=model, ollama_url=ollama_url)
            facts.extend(parse_extracted_facts(response, document))
    return facts


def write_facts_jsonl(facts: Sequence[ExtractedFact], path: str | Path) -> None:
    """Persist extracted facts for review before graph ingestion."""

    rows = [fact.__dict__ for fact in facts]
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def create_kuzu_schema(connection: object) -> None:
    """Create the minimal Kuzu schema used by this pipeline."""

    statements = [
        "CREATE NODE TABLE IF NOT EXISTS Entity(id STRING, name STRING, PRIMARY KEY(id))",
        "CREATE NODE TABLE IF NOT EXISTS Source(id STRING, source_type STRING, locator STRING, PRIMARY KEY(id))",
        "CREATE REL TABLE IF NOT EXISTS FACT(FROM Entity TO Entity, predicate STRING, source_id STRING, evidence STRING, confidence DOUBLE)",
        "CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(FROM Entity TO Source)",
    ]
    for statement in statements:
        connection.execute(statement)


def populate_kuzu(facts: Sequence[ExtractedFact], db_path: str | Path) -> None:
    """Populate a Kuzu database with extracted facts."""

    kuzu = importlib.import_module("kuzu")
    database = kuzu.Database(str(db_path))
    connection = kuzu.Connection(database)
    create_kuzu_schema(connection)
    for fact in facts:
        subject_id = stable_id("entity", fact.subject.lower())
        object_id = stable_id("entity", fact.object.lower())
        connection.execute("MERGE (e:Entity {id: $id}) SET e.name = $name", {"id": subject_id, "name": fact.subject})
        connection.execute("MERGE (e:Entity {id: $id}) SET e.name = $name", {"id": object_id, "name": fact.object})
        connection.execute(
            "MERGE (s:Source {id: $id}) SET s.source_type = $source_type, s.locator = $locator",
            {"id": fact.source_id, "source_type": fact.source_type, "locator": fact.locator},
        )
        connection.execute(
            "MATCH (a:Entity {id: $subject_id}), (b:Entity {id: $object_id}) "
            "CREATE (a)-[:FACT {predicate: $predicate, source_id: $source_id, evidence: $evidence, confidence: $confidence}]->(b)",
            {
                "subject_id": subject_id,
                "object_id": object_id,
                "predicate": fact.predicate,
                "source_id": fact.source_id,
                "evidence": fact.evidence,
                "confidence": fact.confidence,
            },
        )
        connection.execute(
            "MATCH (e:Entity {id: $entity_id}), (s:Source {id: $source_id}) MERGE (e)-[:MENTIONED_IN]->(s)",
            {"entity_id": subject_id, "source_id": fact.source_id},
        )
        connection.execute(
            "MATCH (e:Entity {id: $entity_id}), (s:Source {id: $source_id}) MERGE (e)-[:MENTIONED_IN]->(s)",
            {"entity_id": object_id, "source_id": fact.source_id},
        )


def load_documents(pdf_paths: Sequence[str], wikipedia_urls: Sequence[str]) -> list[SourceDocument]:
    """Load all configured PDF and Wikipedia documents."""

    return [read_pdf(path) for path in pdf_paths] + [read_wikipedia(url) for url in wikipedia_urls]


def add_ingest_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the graph-ingestion CLI command."""

    parser = commands.add_parser("ingest-graph", help="extract facts with local Ollama and populate a Kuzu graph")
    parser.add_argument("--pdf", action="append", default=[], help="PDF path to ingest; may be repeated")
    parser.add_argument("--wikipedia", action="append", default=[], help="Wikipedia page URL to ingest; may be repeated")
    parser.add_argument("--db", required=True, help="Kuzu database directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="local Ollama model name")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--facts-jsonl", help="optional JSONL file for extracted facts")
    parser.add_argument("--max-chars", type=int, default=6000, help="maximum characters per LLM chunk")
    parser.add_argument("--overlap", type=int, default=500, help="overlap characters between chunks")


def run_ingest_command(args: argparse.Namespace) -> None:
    """Run graph ingestion from parsed CLI arguments."""

    documents = load_documents(args.pdf, args.wikipedia)
    if not documents:
        raise ValueError("provide at least one --pdf or --wikipedia source")
    facts = extract_facts(documents, model=args.model, ollama_url=args.ollama_url, max_chars=args.max_chars, overlap=args.overlap)
    if args.facts_jsonl:
        write_facts_jsonl(facts, args.facts_jsonl)
    populate_kuzu(facts, args.db)
    print(json.dumps({"documents": len(documents), "facts": len(facts), "db": args.db}, indent=2))
