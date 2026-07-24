#!/usr/bin/env python3
"""Extract and rank bibliography entries already parsed into the RAG database.

The command is deliberately review-first: it writes CSV/Markdown candidates and
never downloads or indexes a referenced work.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TOPIC = (
    "subgrid scale SGS stress tensor nonlinear gradient model NGM "
    "large eddy simulation LES LCR Leonard cross Reynolds hierarchy "
    "moment expansion two dimensional turbulence closure"
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z][a-z0-9-]{2,}")
AUTHOR_START_RE = re.compile(
    r"^(?:[A-Z][A-Za-z'’-]+,\s|(?:[A-Z]\.\s){1,4}[A-Z][A-Za-z'’-]+)"
)
STOPWORDS = {
    "and", "the", "for", "from", "with", "using", "model", "models",
    "paper", "journal", "vol", "volume", "press", "university",
}


@dataclass
class Candidate:
    raw_reference: str
    cited_by: list[str]
    doi: str = ""
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    metadata_status: str = "unresolved"
    score: float = 0.0
    recommendation: str = "maybe"
    reason: str = ""


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def reference_key(text: str) -> str:
    doi = DOI_RE.search(text)
    if doi:
        return "doi:" + doi.group(0).lower().rstrip(".,")
    words = [word for word in WORD_RE.findall(text.lower()) if word not in STOPWORDS]
    year = YEAR_RE.search(text)
    if year:
        words.append(year.group(0).lower())
    return "text:" + " ".join(words[:18])


def looks_like_reference(text: str) -> bool:
    text = normalize(text)
    return len(text) >= 45 and bool(YEAR_RE.search(text)) and (
        AUTHOR_START_RE.match(text) is not None or DOI_RE.search(text) is not None
    )


def extract_from_db(
    database: Path, title_patterns: list[str],
) -> list[Candidate]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        """
        SELECT d.title, c.ordinal, c.heading_path_json, c.text
        FROM chunks c
        JOIN document_versions v ON v.id = c.document_version_id
        JOIN documents d ON d.id = v.document_id
        WHERE d.deleted_at IS NULL AND v.id = d.current_version_id
        ORDER BY d.title, c.ordinal
        """
    ).fetchall()
    selected = [
        row for row in rows
        if any(pattern.lower() in row[0].lower() for pattern in title_patterns)
    ]
    if not selected:
        raise SystemExit(f"No indexed documents matched: {title_patterns}")

    max_ordinal: dict[str, int] = defaultdict(int)
    for title, ordinal, _heading, _text in selected:
        max_ordinal[title] = max(max_ordinal[title], ordinal)

    grouped: dict[str, Candidate] = {}
    for title, ordinal, heading_json, text in selected:
        heading = " ".join(json.loads(heading_json or "[]")).lower()
        in_reference_heading = "reference" in heading or "bibliograph" in heading
        in_tail = ordinal >= int(max_ordinal[title] * 0.55)
        if not (in_reference_heading or (in_tail and looks_like_reference(text))):
            continue
        reference = normalize(text)
        if not looks_like_reference(reference):
            continue
        key = reference_key(reference)
        candidate = grouped.get(key)
        if candidate is None:
            grouped[key] = Candidate(reference, [title])
        elif title not in candidate.cited_by:
            candidate.cited_by.append(title)
    return list(grouped.values())


def resolve_crossref(candidate: Candidate, timeout: float = 15.0) -> None:
    doi_match = DOI_RE.search(candidate.raw_reference)
    if doi_match:
        url = "https://api.crossref.org/works/" + doi_match.group(0).rstrip(".,")
    else:
        query = urlencode({
            "query.bibliographic": candidate.raw_reference,
            "rows": 1,
            "select": "DOI,title,author,published,container-title",
        })
        url = "https://api.crossref.org/works?" + query
    request = Request(url, headers={"User-Agent": "AcaCite/0.1"})
    try:
        message = None
        for attempt in range(4):
            try:
                with urlopen(request, timeout=timeout) as response:
                    message = json.load(response)["message"]
                break
            except HTTPError as exc:
                if exc.code != 429 or attempt == 3:
                    raise
                time.sleep(attempt + 1)
        assert message is not None
        item = message if "DOI" in message else message.get("items", [{}])[0]
        candidate.doi = str(item.get("DOI", ""))
        candidate.title = " ".join(item.get("title", [])).strip()
        candidate.authors = "; ".join(
            " ".join(filter(None, [author.get("given"), author.get("family")]))
            for author in item.get("author", [])
        )
        date_parts = item.get("published", {}).get("date-parts", [[]])
        candidate.year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
        candidate.venue = " ".join(item.get("container-title", [])).strip()
        candidate.metadata_status = "resolved" if candidate.title else "unresolved"
    except Exception as exc:  # review output should survive transient API failures
        candidate.metadata_status = f"error:{type(exc).__name__}"


def rank(candidate: Candidate, topic: str) -> None:
    topic_words = set(WORD_RE.findall(topic.lower())) - STOPWORDS
    haystack = f"{candidate.title} {candidate.raw_reference}".lower()
    matches = sorted(word for word in topic_words if word in haystack)
    shared = len(candidate.cited_by) > 1
    score = min(70.0, 7.0 * len(matches)) + (20.0 if shared else 0.0)
    if candidate.metadata_status == "resolved":
        score += 5.0
    if DOI_RE.search(candidate.raw_reference) or candidate.doi:
        score += 5.0
    candidate.score = min(100.0, score)
    candidate.recommendation = (
        "include" if candidate.score >= 45 else
        "maybe" if candidate.score >= 20 else "skip"
    )
    reasons = []
    if shared:
        reasons.append("cited by both source papers")
    if matches:
        reasons.append("topic terms: " + ", ".join(matches[:8]))
    if not reasons:
        reasons.append("weak direct match to the current research topic")
    candidate.reason = "; ".join(reasons)


def write_outputs(candidates: list[Candidate], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "decision", "score", "recommendation", "title", "authors", "year",
        "venue", "doi", "cited_by", "reason", "metadata_status", "raw_reference",
    ]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow({
                "decision": "", "score": f"{item.score:.1f}",
                "recommendation": item.recommendation, "title": item.title,
                "authors": item.authors, "year": item.year, "venue": item.venue,
                "doi": item.doi, "cited_by": " | ".join(item.cited_by),
                "reason": item.reason, "metadata_status": item.metadata_status,
                "raw_reference": item.raw_reference,
            })
    with output.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# Referenced-paper review\n\n")
        handle.write("Set `decision` in the CSV to `include`, `maybe`, or `skip`.\n\n")
        for index, item in enumerate(candidates, 1):
            title = item.title or item.raw_reference[:110]
            handle.write(
                f"{index}. **[{item.recommendation.upper()} · {item.score:.0f}] "
                f"{title}**\n   - {item.reason}\n"
                f"   - Cited by: {', '.join(item.cited_by)}"
                f"{f'; DOI: {item.doi}' if item.doi else ''}\n\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--paper-title", action="append", required=True)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = extract_from_db(args.database, args.paper_title)
    if args.resolve:
        for candidate in candidates:
            resolve_crossref(candidate)
            time.sleep(0.12)
    for candidate in candidates:
        rank(candidate, args.topic)
    candidates.sort(key=lambda item: (-item.score, item.title or item.raw_reference))
    write_outputs(candidates, args.output)
    print(f"Wrote {len(candidates)} candidates to {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
