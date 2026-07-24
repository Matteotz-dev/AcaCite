#!/usr/bin/env python3
"""Resolve first-hop and second-hop paper citations, optionally downloading OA PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
SPACE_RE = re.compile(r"\s+")
AUTHOR_START_RE = re.compile(
    r"^(?:[A-Z][A-Za-z'’-]+,\s|(?:[A-Z]\.\s){1,4}[A-Z][A-Za-z'’-]+)"
)
USER_AGENT = "AcaCite/0.1 (https://github.com/Matteotz-dev/AcaCite)"


@dataclass
class Work:
    key: str
    depth: int
    title: str = ""
    doi: str = ""
    year: str = ""
    venue: str = ""
    authors: list[str] = field(default_factory=list)
    openalex_id: str = ""
    raw_reference: str = ""
    cited_by: list[str] = field(default_factory=list)
    referenced_works: list[str] = field(default_factory=list)
    oa_url: str = ""
    pdf_url: str = ""
    pdf_path: str = ""
    status: str = "unresolved"


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def looks_like_reference(text: str) -> bool:
    text = normalize(text)
    return len(text) >= 45 and bool(YEAR_RE.search(text)) and (
        AUTHOR_START_RE.match(text) is not None or DOI_RE.search(text) is not None
    )


def extract_seed_references(database: Path, title_patterns: list[str]) -> list[Work]:
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

    grouped: dict[str, Work] = {}
    for title, ordinal, heading_json, text in selected:
        heading = " ".join(json.loads(heading_json or "[]")).lower()
        in_reference_heading = "reference" in heading or "bibliograph" in heading
        in_tail = ordinal >= int(max_ordinal[title] * 0.55)
        reference = normalize(text)
        if not (in_reference_heading or in_tail) or not looks_like_reference(reference):
            continue
        doi_match = DOI_RE.search(reference)
        key = "doi:" + doi_match.group(0).lower().rstrip(".,") if doi_match else (
            "ref:" + hashlib.sha256(reference.encode()).hexdigest()[:16]
        )
        work = grouped.setdefault(key, Work(key=key, depth=1, raw_reference=reference))
        if title not in work.cited_by:
            work.cited_by.append(title)
    return list(grouped.values())


def request_json(url: str, timeout: float) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def resolve_crossref(work: Work, timeout: float) -> None:
    doi_match = DOI_RE.search(work.raw_reference)
    if doi_match:
        url = "https://api.crossref.org/works/" + quote(doi_match.group(0).rstrip(".,"))
    else:
        query = urlencode({
            "query.bibliographic": work.raw_reference,
            "rows": 1,
            "select": "DOI,title,author,published,container-title",
        })
        url = "https://api.crossref.org/works?" + query
    try:
        message = request_json(url, timeout)["message"]
        item = message if "DOI" in message else message.get("items", [{}])[0]
        work.doi = str(item.get("DOI", "")).lower()
        work.title = " ".join(item.get("title", [])).strip()
        work.authors = [
            " ".join(filter(None, [author.get("given"), author.get("family")]))
            for author in item.get("author", [])
        ]
        date_parts = item.get("published", {}).get("date-parts", [[]])
        work.year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
        work.venue = " ".join(item.get("container-title", [])).strip()
        work.key = "doi:" + work.doi if work.doi else work.key
        work.status = "crossref"
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        work.status = f"crossref_error:{type(exc).__name__}"


def hydrate_openalex(work: Work, timeout: float) -> None:
    if work.doi:
        url = "https://api.openalex.org/works/doi:" + quote("https://doi.org/" + work.doi)
    elif work.openalex_id:
        url = work.openalex_id
    else:
        return
    try:
        item = request_json(url, timeout)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return
    apply_openalex(work, item)


def apply_openalex(work: Work, item: dict) -> None:
    work.openalex_id = item.get("id", work.openalex_id)
    work.title = item.get("title") or work.title
    work.doi = (item.get("doi") or work.doi).replace("https://doi.org/", "").lower()
    work.year = str(item.get("publication_year") or work.year)
    host = item.get("primary_location", {}).get("source") or {}
    work.venue = host.get("display_name") or work.venue
    work.authors = [
        entry.get("author", {}).get("display_name", "")
        for entry in item.get("authorships", [])
        if entry.get("author", {}).get("display_name")
    ]
    work.referenced_works = list(item.get("referenced_works") or [])
    open_access = item.get("open_access") or {}
    best_oa = item.get("best_oa_location") or {}
    primary = item.get("primary_location") or {}
    work.oa_url = open_access.get("oa_url") or best_oa.get("landing_page_url") or ""
    work.pdf_url = best_oa.get("pdf_url") or primary.get("pdf_url") or ""
    work.status = "openalex"


def fetch_openalex_id(openalex_id: str, depth: int, cited_by: str, timeout: float) -> Work | None:
    try:
        item = request_json(openalex_id, timeout)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    work = Work(key="openalex:" + openalex_id.rsplit("/", 1)[-1], depth=depth, cited_by=[cited_by])
    apply_openalex(work, item)
    if work.doi:
        work.key = "doi:" + work.doi
    return work


def safe_pdf_name(work: Work) -> str:
    stem = work.doi.replace("/", "_") if work.doi else work.key.replace(":", "_")
    title = re.sub(r"[^A-Za-z0-9._-]+", "-", (work.title or stem).strip())[:80].strip("-")
    return f"{stem}-{title}.pdf"


def download_pdf(work: Work, directory: Path, timeout: float, max_bytes: int) -> None:
    if not work.pdf_url or urlparse(work.pdf_url).scheme not in {"http", "https"}:
        return
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / safe_pdf_name(work)
    if target.exists():
        work.pdf_path = str(target)
        return
    request = Request(work.pdf_url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError):
        return
    if len(data) > max_bytes or b"%PDF" not in data[:1024] or "html" in content_type.lower():
        return
    target.write_bytes(data)
    work.pdf_path = str(target)


def expand(seeds: list[Work], depth: int, timeout: float, polite_sleep: float) -> dict[str, Work]:
    works: dict[str, Work] = {}
    queue: deque[Work] = deque(seeds)
    while queue:
        work = queue.popleft()
        existing = works.get(work.key)
        if existing:
            for source in work.cited_by:
                if source not in existing.cited_by:
                    existing.cited_by.append(source)
            continue
        if work.depth == 1:
            resolve_crossref(work, timeout)
            time.sleep(polite_sleep)
        hydrate_openalex(work, timeout)
        time.sleep(polite_sleep)
        works[work.key] = work
        if work.depth >= depth:
            continue
        parent = work.title or work.doi or work.key
        for openalex_id in work.referenced_works:
            child = fetch_openalex_id(openalex_id, work.depth + 1, parent, timeout)
            time.sleep(polite_sleep)
            if child is not None:
                queue.append(child)
    return works


def write_manifest(works: dict[str, Work], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(works.values(), key=lambda item: (item.depth, item.title or item.key))
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for work in ordered:
            handle.write(json.dumps(asdict(work), sort_keys=True) + "\n")
    with output.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# Citation Expansion\n\n")
        for work in ordered:
            title = work.title or work.raw_reference[:100] or work.key
            handle.write(f"- depth {work.depth}: **{title}**")
            if work.year:
                handle.write(f" ({work.year})")
            if work.doi:
                handle.write(f" DOI: `{work.doi}`")
            if work.pdf_path:
                handle.write(f" PDF: `{work.pdf_path}`")
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--paper-title", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=2, choices=(1, 2))
    parser.add_argument("--download-oa-pdfs", action="store_true")
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--polite-sleep", type=float, default=0.2)
    parser.add_argument("--max-pdf-mb", type=int, default=80)
    args = parser.parse_args()

    seeds = extract_seed_references(args.database, args.paper_title)
    works = expand(seeds, args.depth, args.timeout, args.polite_sleep)
    if args.download_oa_pdfs:
        pdf_dir = args.pdf_dir or args.output.parent / "pdfs"
        for work in works.values():
            download_pdf(work, pdf_dir, args.timeout, args.max_pdf_mb * 1024 * 1024)
            time.sleep(args.polite_sleep)
    write_manifest(works, args.output)
    print(f"Wrote {len(works)} works to {args.output.with_suffix('.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
