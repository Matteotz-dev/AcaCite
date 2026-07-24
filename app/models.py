"""Shared domain and API types for provenance and later RAG phases."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceType(StrEnum):
    PAPER = "paper"
    REPO_FILE = "repo_file"
    NOTE = "note"
    WEB_CAPTURE = "web_capture"


class VersionStatus(StrEnum):
    PENDING = "pending"
    PARSED = "parsed"
    INDEXED = "indexed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentCreate(FrozenModel):
    source_type: SourceType
    canonical_uri: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    dataset: str = Field(min_length=1)
    title: str | None = None
    mime_type: str | None = None
    language: str | None = None
    project: str | None = None


class DocumentRecord(DocumentCreate):
    id: UUID
    current_version_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class DocumentVersionCreate(FrozenModel):
    document_id: UUID
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    source_mtime: datetime | None = None
    git_repository: str | None = None
    git_commit: str | None = None
    doi: str | None = None
    authors: tuple[str, ...] = ()
    publication_date: str | None = None
    raw_snapshot_path: Path | None = None


class DocumentVersionRecord(DocumentVersionCreate):
    id: UUID
    status: VersionStatus
    error: str | None = None
    created_at: datetime


class ChunkCreate(FrozenModel):
    document_version_id: UUID
    ordinal: int = Field(ge=0)
    text_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0)
    chunk_type: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    symbol: str | None = None
    language: str | None = None
    parent_chunk_id: UUID | None = None

    @model_validator(mode="after")
    def validate_source_spans(self) -> "ChunkCreate":
        for kind, start, end in (
            ("page", self.page_start, self.page_end),
            ("line", self.line_start, self.line_end),
        ):
            if end is not None and start is None:
                raise ValueError(f"{kind}_end requires {kind}_start")
            if start is not None and end is not None and end < start:
                raise ValueError(f"{kind}_end cannot precede {kind}_start")
        return self


class ChunkRecord(ChunkCreate):
    id: UUID
    qdrant_point_id: str | None = None
    cognee_ref: str | None = None
    created_at: datetime


class Citation(FrozenModel):
    citation_id: str
    chunk_id: UUID
    display: str
    source_type: SourceType
    canonical_uri: str
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    git_commit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
