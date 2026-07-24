"""Explicit, provenance-bearing note ingestion for the versioned API."""

from __future__ import annotations

from datetime import UTC, datetime

from app.config import Settings
from app.db import ProvenanceRepository, sha256_text
from app.models import ChunkCreate, DocumentCreate, DocumentVersionCreate, SourceType, VersionStatus


def register_memory(
    *, content: str, source_uri: str, dataset: str, settings: Settings,
    title: str | None = None, project: str | None = None,
):
    """Register one explicit note; indexing remains a separate verified step."""
    clean = content.strip()
    if not clean:
        raise ValueError("content cannot be empty")
    if not source_uri.strip():
        raise ValueError("source_uri cannot be empty")
    content_hash = sha256_text(clean)
    repository = ProvenanceRepository(settings.provenance_db_path)
    repository.initialize()
    with repository.transaction() as connection:
        document = repository.upsert_document(DocumentCreate(
            source_type=SourceType.NOTE, canonical_uri=source_uri.strip(),
            content_hash=content_hash, dataset=dataset, project=project,
            title=title, mime_type="text/plain", language="en",
        ), connection)
        version = repository.create_version(DocumentVersionCreate(
            document_id=document.id, content_hash=content_hash,
            source_mtime=datetime.now(UTC), parser_name="explicit-memory",
            parser_version="1", index_version=settings.index_version,
        ), connection)
        chunks = repository.list_version_chunks(version.id)
        if not chunks:
            chunks = (repository.add_chunk(ChunkCreate(
                document_version_id=version.id, ordinal=0, text_hash=content_hash,
                text=clean, token_count=max(1, len(clean.split())), chunk_type="note",
                language="en",
            ), connection),)
        if version.status is VersionStatus.PENDING:
            version = repository.set_version_status(
                version.id, VersionStatus.PARSED, connection=connection
            )
    return version, chunks
