"""Construction and resolution of opaque, server-owned citations."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

from app.db.provenance import ProvenanceRepository, validate_approved_path
from app.models import Citation, SourceType


CITATION_PATTERN = re.compile(
    r"\[SRC:(?P<id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})\]"
)
SHAPED_CITATION_PATTERN = re.compile(r"\[SRC:(?P<id>[^\]]*)\]")


def citation_id(chunk_id: UUID) -> str:
    return f"[SRC:{chunk_id}]"


def extract_citation_ids(answer: str) -> tuple[str, ...]:
    """Return unique shaped citation tokens in first-occurrence order.

    Well-formed UUIDs are normalized; malformed model output is retained so the
    validator can report it rather than silently ignoring a hallucinated ID.
    """
    citations: list[str] = []
    for match in SHAPED_CITATION_PATTERN.finditer(answer):
        raw = match.group("id")
        try:
            rendered = citation_id(UUID(raw))
        except ValueError:
            rendered = match.group(0)
        if rendered not in citations:
            citations.append(rendered)
    return tuple(citations)


class CitationResolver:
    def __init__(
        self,
        repository: ProvenanceRepository,
        approved_roots: tuple[Path, ...] = (),
    ):
        self.repository = repository
        self.approved_roots = approved_roots

    def resolve(self, chunk_id: UUID | str) -> Citation | None:
        identifier = UUID(str(chunk_id).removeprefix("[SRC:").removesuffix("]"))
        row = self.repository.citation_row(identifier)
        if row is None:
            return None
        source_type = SourceType(row["source_type"])
        display = self._display(row, source_type)
        metadata = {
            "dataset": row["dataset"],
            "project": row["project"],
            "document_version_id": row["document_version_id"],
            "heading_path": tuple(__import__("json").loads(row["heading_path_json"])),
            "doi": row["doi"],
            "publication_date": row["publication_date"],
            "symbol": row["symbol"],
        }
        local_path = _file_uri_path(row["canonical_uri"])
        if local_path is not None and self.approved_roots:
            try:
                metadata["open_path"] = str(
                    validate_approved_path(local_path, self.approved_roots)
                )
            except (OSError, ValueError):
                metadata["open_path_blocked"] = True
        return Citation(
            citation_id=citation_id(identifier), chunk_id=identifier, display=display,
            source_type=source_type, canonical_uri=row["canonical_uri"], title=row["title"],
            page_start=row["page_start"], page_end=row["page_end"],
            line_start=row["line_start"], line_end=row["line_end"],
            git_commit=row["git_commit"], metadata=metadata,
        )

    def validate(self, answer: str, allowed_chunk_ids: set[UUID]) -> tuple[list[Citation], list[str]]:
        valid: list[Citation] = []
        invalid: list[str] = []
        for identifier_text in extract_citation_ids(answer):
            try:
                identifier = UUID(identifier_text[5:-1])
            except ValueError:
                invalid.append(identifier_text)
                continue
            resolved = self.resolve(identifier) if identifier in allowed_chunk_ids else None
            if resolved is None:
                invalid.append(identifier_text)
            else:
                valid.append(resolved)
        return valid, invalid

    @staticmethod
    def _display(row, source_type: SourceType) -> str:
        title = row["title"] or row["canonical_uri"]
        if source_type is SourceType.PAPER:
            page = _span("p.", row["page_start"], row["page_end"])
            return ", ".join(part for part in (title, page) if part)
        commit = f"@{row['git_commit'][:8]}" if row["git_commit"] else ""
        lines = _span("L", row["line_start"], row["line_end"])
        return f"{title}{commit}" + (f"#{lines}" if lines else "")


def _span(prefix: str, start: int | None, end: int | None) -> str:
    if start is None:
        return ""
    return f"{prefix} {start}" if end is None or end == start else f"{prefix} {start}-{end}"


def _file_uri_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    return Path(unquote(parsed.path))
