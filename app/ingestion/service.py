"""Phase-2 ingestion orchestration through normalized chunks and SQLite."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.db import ProvenanceRepository, sha256_bytes, sha256_text, validate_approved_path
from app.models import ChunkCreate, DocumentCreate, DocumentVersionCreate, SourceType, VersionStatus

from .registry import parse_file


def ingest_file(
    path: Path,
    *,
    dataset: str,
    settings: Settings,
    project: str | None = None,
    git_repository: str | None = None,
    git_commit: str | None = None,
) -> tuple[object, tuple[object, ...]]:
    """Parse and register one immutable version; vector indexing is Phase 3."""
    source = validate_approved_path(path, settings.approved_ingestion_roots)
    content_hash = sha256_bytes(source.read_bytes())
    normalized = parse_file(source)
    source_type = SourceType.PAPER if source.suffix.lower() == ".pdf" else (
        SourceType.REPO_FILE if git_repository else SourceType.NOTE
    )
    repository = ProvenanceRepository(settings.provenance_db_path)
    repository.initialize()
    with repository.transaction() as connection:
        document = repository.upsert_document(DocumentCreate(
            source_type=source_type, canonical_uri=source.as_uri(),
            content_hash=content_hash, dataset=dataset, project=project,
            title=normalized.title, mime_type=normalized.mime_type,
            language=normalized.language,
        ), connection)
        version = repository.create_version(DocumentVersionCreate(
            document_id=document.id, content_hash=content_hash,
            source_mtime=datetime.fromtimestamp(source.stat().st_mtime, UTC),
            git_repository=git_repository, git_commit=git_commit,
            parser_name=normalized.parser_name, parser_version=normalized.parser_version,
            index_version=settings.index_version,
        ), connection)
        chunks = tuple(repository.add_chunk(ChunkCreate(
            document_version_id=version.id, ordinal=ordinal,
            text_hash=sha256_text(chunk.text), text=chunk.text,
            token_count=chunk.token_count, chunk_type=chunk.chunk_type,
            heading_path=chunk.heading_path, page_start=chunk.page_start,
            page_end=chunk.page_end, line_start=chunk.line_start,
            line_end=chunk.line_end, symbol=chunk.symbol, language=chunk.language,
        ), connection) for ordinal, chunk in enumerate(normalized.chunks))
        if version.status is VersionStatus.PENDING:
            version = repository.set_version_status(version.id, VersionStatus.PARSED, connection=connection)
    return version, chunks
