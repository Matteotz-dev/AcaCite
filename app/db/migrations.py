"""Minimal, transactional SQLite schema migration runner."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


SCHEMA_VERSION = 2
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

MIGRATION_2 = """
CREATE TABLE ingestion_job_files (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
    canonical_uri TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('index', 'skip', 'delete')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'failed')),
    document_version_id TEXT REFERENCES document_versions(id),
    chunks_indexed INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE(job_id, canonical_uri)
);
CREATE INDEX idx_ingestion_job_files_job ON ingestion_job_files(job_id);
CREATE INDEX idx_ingestion_job_files_status ON ingestion_job_files(status);
"""


def migrate(connection: sqlite3.Connection) -> None:
    """Bring *connection* to the current schema version atomically."""
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    current = connection.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema {current} is newer than supported {SCHEMA_VERSION}"
        )
    migrations = {1: SCHEMA_PATH.read_text(encoding="utf-8"), 2: MIGRATION_2}
    for version in range(current + 1, SCHEMA_VERSION + 1):
        applied_at = datetime.now(UTC).isoformat()
        script = (
            "BEGIN IMMEDIATE;\n" + migrations[version]
            + f"\nINSERT INTO schema_migrations(version, applied_at) "
            f"VALUES ({version}, '{applied_at}');\nCOMMIT;"
        )
        try:
            connection.executescript(script)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
