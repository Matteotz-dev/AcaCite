"""Canonical SQLite provenance repository and stable identity helpers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from uuid import UUID, uuid4, uuid5

from app.models import (
    ChunkCreate,
    ChunkRecord,
    DocumentCreate,
    DocumentRecord,
    DocumentVersionCreate,
    DocumentVersionRecord,
    SourceType,
    VersionStatus,
)

from .migrations import migrate


IDENTITY_NAMESPACE = UUID("59b73bea-b9e1-52e5-93d3-76c6451e802d")


def utc_now() -> datetime:
    return datetime.now(UTC)


def sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_document_id(dataset: str, canonical_uri: str) -> UUID:
    return uuid5(IDENTITY_NAMESPACE, f"document\0{dataset}\0{canonical_uri}")


def stable_version_id(document_id: UUID, content_hash: str, index_version: str) -> UUID:
    return uuid5(document_id, f"version\0{content_hash}\0{index_version}")


def stable_chunk_id(
    document_version_id: UUID, ordinal: int, text_hash: str
) -> UUID:
    return uuid5(document_version_id, f"chunk\0{ordinal}\0{text_hash}")


def validate_approved_path(
    path: Path | str,
    approved_roots: Sequence[Path | str],
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a path and ensure symlinks cannot escape an approved root."""
    candidate = Path(path).expanduser().resolve(strict=must_exist)
    roots = tuple(Path(root).expanduser().resolve(strict=True) for root in approved_roots)
    if not roots:
        raise ValueError("at least one approved ingestion root is required")
    if not any(candidate == root or candidate.is_relative_to(root) for root in roots):
        raise ValueError(f"path is outside approved ingestion roots: {candidate}")
    if must_exist and not (candidate.is_file() or candidate.is_dir()):
        raise ValueError(f"path must be a regular file or directory: {candidate}")
    return candidate


class ProvenanceRepository:
    """Transaction-oriented repository; SQLite is the provenance authority."""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            with connection:
                migrate(connection)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_document(
        self, value: DocumentCreate, connection: sqlite3.Connection | None = None
    ) -> DocumentRecord:
        document_id = stable_document_id(value.dataset, value.canonical_uri)
        now = utc_now().isoformat()
        with self._connection(connection) as conn:
            conn.execute(
                """INSERT INTO documents(
                       id, source_type, canonical_uri, title, content_hash, mime_type,
                       language, dataset, project, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(canonical_uri, dataset) DO UPDATE SET
                       source_type=excluded.source_type, title=excluded.title,
                       mime_type=excluded.mime_type, language=excluded.language,
                       project=excluded.project,
                       updated_at=excluded.updated_at, deleted_at=NULL""",
                (
                    str(document_id), value.source_type.value, value.canonical_uri,
                    value.title, value.content_hash, value.mime_type, value.language,
                    value.dataset, value.project, now, now,
                ),
            )
            row = conn.execute("SELECT * FROM documents WHERE id=?", (str(document_id),)).fetchone()
        return self._document(row)

    def create_version(
        self,
        value: DocumentVersionCreate,
        connection: sqlite3.Connection | None = None,
    ) -> DocumentVersionRecord:
        version_id = stable_version_id(value.document_id, value.content_hash, value.index_version)
        with self._connection(connection) as conn:
            conn.execute(
                """INSERT INTO document_versions(
                       id, document_id, content_hash, source_mtime, git_repository,
                       git_commit, doi, authors_json, publication_date, parser_name,
                       parser_version, index_version, raw_snapshot_path, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                   ON CONFLICT(document_id, content_hash, index_version) DO NOTHING""",
                (
                    str(version_id), str(value.document_id), value.content_hash,
                    _iso(value.source_mtime), value.git_repository, value.git_commit,
                    value.doi, json.dumps(value.authors), value.publication_date,
                    value.parser_name, value.parser_version, value.index_version,
                    str(value.raw_snapshot_path) if value.raw_snapshot_path else None,
                    utc_now().isoformat(),
                ),
            )
            row = conn.execute("SELECT * FROM document_versions WHERE id=?", (str(version_id),)).fetchone()
        return self._version(row)

    def add_chunk(
        self, value: ChunkCreate, connection: sqlite3.Connection | None = None
    ) -> ChunkRecord:
        chunk_id = stable_chunk_id(value.document_version_id, value.ordinal, value.text_hash)
        with self._connection(connection) as conn:
            conn.execute(
                """INSERT INTO chunks(
                       id, document_version_id, ordinal, text_hash, text, token_count,
                       chunk_type, heading_path_json, page_start, page_end, line_start,
                       line_end, symbol, language, parent_chunk_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_version_id, ordinal) DO NOTHING""",
                (
                    str(chunk_id), str(value.document_version_id), value.ordinal,
                    value.text_hash, value.text, value.token_count, value.chunk_type,
                    json.dumps(value.heading_path), value.page_start, value.page_end,
                    value.line_start, value.line_end, value.symbol, value.language,
                    str(value.parent_chunk_id) if value.parent_chunk_id else None,
                    utc_now().isoformat(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM chunks WHERE document_version_id=? AND ordinal=?",
                (str(value.document_version_id), value.ordinal),
            ).fetchone()
            if row["id"] != str(chunk_id):
                raise ValueError("chunk ordinal already contains different content")
        return self._chunk(row)

    def add_chunks(
        self, values: Iterable[ChunkCreate], connection: sqlite3.Connection | None = None
    ) -> list[ChunkRecord]:
        if connection is not None:
            return [self.add_chunk(value, connection) for value in values]
        with self.transaction() as conn:
            return [self.add_chunk(value, conn) for value in values]

    def set_version_status(
        self,
        version_id: UUID,
        status: VersionStatus,
        error: str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> DocumentVersionRecord:
        with self._connection(connection) as conn:
            cursor = conn.execute(
                "UPDATE document_versions SET status=?, error=? WHERE id=?",
                (status.value, error, str(version_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown document version: {version_id}")
            row = conn.execute("SELECT * FROM document_versions WHERE id=?", (str(version_id),)).fetchone()
        return self._version(row)

    def mark_version_current(
        self, version_id: UUID, connection: sqlite3.Connection | None = None
    ) -> DocumentVersionRecord:
        """Atomically index the selected version and supersede its predecessor."""
        with self._connection(connection) as conn:
            row = conn.execute("SELECT * FROM document_versions WHERE id=?", (str(version_id),)).fetchone()
            if row is None:
                raise KeyError(f"unknown document version: {version_id}")
            document_id = row["document_id"]
            current = conn.execute(
                "SELECT current_version_id FROM documents WHERE id=?", (document_id,)
            ).fetchone()[0]
            if current and current != str(version_id):
                conn.execute(
                    "UPDATE document_versions SET status='superseded', error=NULL WHERE id=?",
                    (current,),
                )
            conn.execute(
                "UPDATE document_versions SET status='indexed', error=NULL WHERE id=?",
                (str(version_id),),
            )
            conn.execute(
                "UPDATE documents SET current_version_id=?, content_hash=?, updated_at=? WHERE id=?",
                (str(version_id), row["content_hash"], utc_now().isoformat(), document_id),
            )
            updated = conn.execute("SELECT * FROM document_versions WHERE id=?", (str(version_id),)).fetchone()
        return self._version(updated)

    def get_document(self, document_id: UUID) -> DocumentRecord | None:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (str(document_id),)).fetchone()
        return self._document(row) if row else None

    def get_version(self, version_id: UUID) -> DocumentVersionRecord | None:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM document_versions WHERE id=?", (str(version_id),)).fetchone()
        return self._version(row) if row else None

    def get_chunk(self, chunk_id: UUID) -> ChunkRecord | None:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id=?", (str(chunk_id),)).fetchone()
        return self._chunk(row) if row else None

    def list_version_chunks(self, version_id: UUID) -> tuple[ChunkRecord, ...]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_version_id=? ORDER BY ordinal",
                (str(version_id),),
            ).fetchall()
        return tuple(self._chunk(row) for row in rows)

    def current_version_ids(self) -> tuple[UUID, ...]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT current_version_id FROM documents "
                "WHERE current_version_id IS NOT NULL AND deleted_at IS NULL"
            ).fetchall()
        return tuple(UUID(row[0]) for row in rows)

    def find_document(self, canonical_uri: str, dataset: str) -> DocumentRecord | None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE canonical_uri=? AND dataset=?",
                (canonical_uri, dataset),
            ).fetchone()
        return self._document(row) if row else None

    def list_documents(
        self, *, dataset: str | None = None, project: str | None = None,
        uri_prefix: str | None = None, include_deleted: bool = False,
    ) -> tuple[DocumentRecord, ...]:
        clauses, values = [], []
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        for column, value in (("dataset", dataset), ("project", project)):
            if value is not None:
                clauses.append(f"{column}=?")
                values.append(value)
        if uri_prefix is not None:
            clauses.append("canonical_uri LIKE ?")
            values.append(uri_prefix + "%")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM documents" + where + " ORDER BY canonical_uri", values
            ).fetchall()
        return tuple(self._document(row) for row in rows)

    def mark_document_deleted(self, document_id: UUID) -> UUID | None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT current_version_id FROM documents WHERE id=?", (str(document_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown document: {document_id}")
            current = UUID(row[0]) if row[0] else None
            conn.execute(
                "UPDATE documents SET deleted_at=?, current_version_id=NULL, updated_at=? WHERE id=?",
                (utc_now().isoformat(), utc_now().isoformat(), str(document_id)),
            )
            if current:
                conn.execute(
                    "UPDATE document_versions SET status='superseded' WHERE id=?", (str(current),)
                )
        return current

    def create_ingestion_job(
        self, *, requested_uri: str, dataset: str, options: dict
    ) -> str:
        job_id, now = str(uuid4()), utc_now().isoformat()
        with self._connection(None) as conn:
            conn.execute(
                """INSERT INTO ingestion_jobs(id, requested_uri, dataset, options_json,
                   status, created_at) VALUES (?, ?, ?, ?, 'queued', ?)""",
                (job_id, requested_uri, dataset, json.dumps(options, sort_keys=True), now),
            )
        return job_id

    def update_ingestion_job(self, job_id: str, **values) -> None:
        allowed = {"status", "documents_seen", "documents_changed", "chunks_indexed",
                   "error", "started_at", "finished_at"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown ingestion job fields: {sorted(unknown)}")
        if not values:
            return
        with self._connection(None) as conn:
            cursor = conn.execute(
                "UPDATE ingestion_jobs SET " + ", ".join(f"{key}=?" for key in values)
                + " WHERE id=?", (*values.values(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown ingestion job: {job_id}")

    def record_ingestion_file(
        self, *, job_id: str, canonical_uri: str, action: str, status: str,
        version_id: UUID | None = None, chunks_indexed: int = 0,
        error: str | None = None,
    ) -> None:
        now = utc_now().isoformat()
        with self._connection(None) as conn:
            conn.execute(
                """INSERT INTO ingestion_job_files(
                       id, job_id, canonical_uri, action, status, document_version_id,
                       chunks_indexed, error, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id, canonical_uri) DO UPDATE SET
                       action=excluded.action, status=excluded.status,
                       document_version_id=excluded.document_version_id,
                       chunks_indexed=excluded.chunks_indexed, error=excluded.error,
                       finished_at=excluded.finished_at""",
                (str(uuid4()), job_id, canonical_uri, action, status,
                 str(version_id) if version_id else None, chunks_indexed, error, now, now),
            )

    def get_ingestion_job(self, job_id: str) -> dict | None:
        with closing(self.connect()) as conn:
            row = conn.execute("SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            files = conn.execute(
                "SELECT * FROM ingestion_job_files WHERE job_id=? ORDER BY canonical_uri", (job_id,)
            ).fetchall()
        result = dict(row)
        result["options"] = json.loads(result.pop("options_json"))
        result["files"] = [dict(item) for item in files]
        return result

    def list_ingestion_jobs(self, limit: int = 20) -> list[dict]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["options"] = json.loads(item.pop("options_json"))
            results.append(item)
        return results

    def set_chunk_point_ids(
        self,
        assignments: Iterable[tuple[UUID, str]],
        connection: sqlite3.Connection | None = None,
    ) -> None:
        with self._connection(connection) as conn:
            for chunk_id, point_id in assignments:
                cursor = conn.execute(
                    "UPDATE chunks SET qdrant_point_id=? WHERE id=?",
                    (point_id, str(chunk_id)),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown chunk: {chunk_id}")

    def get_memory_promotion(
        self, version_id: UUID, kind: str, payload_hash: str
    ) -> sqlite3.Row | None:
        with closing(self.connect()) as conn:
            return conn.execute(
                """SELECT * FROM memory_promotions
                   WHERE document_version_id=? AND kind=? AND payload_hash=?""",
                (str(version_id), kind, payload_hash),
            ).fetchone()

    def record_memory_promotion(
        self, *, document_id: UUID, version_id: UUID, kind: str,
        cognee_dataset: str, cognee_ref: str | None, payload_hash: str,
        connection: sqlite3.Connection | None = None,
    ) -> str:
        """Record a completed promotion; the unique key makes retries idempotent."""
        promotion_id = str(uuid4())
        with self._connection(connection) as conn:
            conn.execute(
                """INSERT INTO memory_promotions(
                       id, document_id, document_version_id, kind, cognee_dataset,
                       cognee_ref, payload_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_version_id, kind, payload_hash) DO NOTHING""",
                (promotion_id, str(document_id), str(version_id), kind,
                 cognee_dataset, cognee_ref, payload_hash, utc_now().isoformat()),
            )
            row = conn.execute(
                """SELECT id FROM memory_promotions
                   WHERE document_version_id=? AND kind=? AND payload_hash=?""",
                (str(version_id), kind, payload_hash),
            ).fetchone()
        return row["id"]

    def set_chunk_cognee_ref(
        self, chunk_id: UUID, cognee_ref: str,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        with self._connection(connection) as conn:
            cursor = conn.execute(
                "UPDATE chunks SET cognee_ref=? WHERE id=?", (cognee_ref, str(chunk_id))
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown chunk: {chunk_id}")

    def citation_row(self, chunk_id: UUID) -> sqlite3.Row | None:
        with closing(self.connect()) as conn:
            return conn.execute(
                """SELECT c.*, d.source_type, d.canonical_uri, d.title, d.dataset,
                          d.project, v.git_commit, v.doi, v.publication_date
                   FROM chunks c
                   JOIN document_versions v ON v.id=c.document_version_id
                   JOIN documents d ON d.id=v.document_id
                   WHERE c.id=?""",
                (str(chunk_id),),
            ).fetchone()

    def record_answer_trace(
        self, *, query: str, generator_model: str, filters: dict,
        retrieved_chunk_ids: Sequence[UUID], selected_chunk_ids: Sequence[UUID],
        citation_ids: Sequence[str], latency: dict,
    ) -> str:
        """Persist observable answer inputs/outputs, never model chain-of-thought."""
        trace_id = str(uuid4())
        with self._connection(None) as conn:
            conn.execute(
                """INSERT INTO answer_traces(
                       id, query, generator_model, filters_json, retrieved_chunk_ids,
                       selected_chunk_ids, citation_ids_json, latency_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trace_id, query, generator_model, json.dumps(filters, sort_keys=True),
                 json.dumps([str(value) for value in retrieved_chunk_ids]),
                 json.dumps([str(value) for value in selected_chunk_ids]),
                 json.dumps(list(citation_ids)), json.dumps(latency, sort_keys=True),
                 utc_now().isoformat()),
            )
        return trace_id

    @contextmanager
    def _connection(self, supplied: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
        if supplied is not None:
            yield supplied
            return
        with closing(self.connect()) as connection:
            with connection:
                yield connection

    @staticmethod
    def _document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=UUID(row["id"]), source_type=SourceType(row["source_type"]),
            canonical_uri=row["canonical_uri"], title=row["title"],
            content_hash=row["content_hash"], mime_type=row["mime_type"],
            language=row["language"], dataset=row["dataset"], project=row["project"],
            current_version_id=UUID(row["current_version_id"]) if row["current_version_id"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            deleted_at=datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
        )

    @staticmethod
    def _version(row: sqlite3.Row) -> DocumentVersionRecord:
        return DocumentVersionRecord(
            id=UUID(row["id"]), document_id=UUID(row["document_id"]),
            content_hash=row["content_hash"], source_mtime=_datetime(row["source_mtime"]),
            git_repository=row["git_repository"], git_commit=row["git_commit"], doi=row["doi"],
            authors=tuple(json.loads(row["authors_json"])), publication_date=row["publication_date"],
            parser_name=row["parser_name"], parser_version=row["parser_version"],
            index_version=row["index_version"],
            raw_snapshot_path=Path(row["raw_snapshot_path"]) if row["raw_snapshot_path"] else None,
            status=VersionStatus(row["status"]), error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _chunk(row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            id=UUID(row["id"]), document_version_id=UUID(row["document_version_id"]),
            ordinal=row["ordinal"], text_hash=row["text_hash"], text=row["text"],
            token_count=row["token_count"], chunk_type=row["chunk_type"],
            heading_path=tuple(json.loads(row["heading_path_json"])),
            page_start=row["page_start"], page_end=row["page_end"],
            line_start=row["line_start"], line_end=row["line_end"], symbol=row["symbol"],
            language=row["language"],
            parent_chunk_id=UUID(row["parent_chunk_id"]) if row["parent_chunk_id"] else None,
            qdrant_point_id=row["qdrant_point_id"], cognee_ref=row["cognee_ref"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
