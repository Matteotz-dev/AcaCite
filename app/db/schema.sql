PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (source_type IN ('paper', 'repo_file', 'note', 'web_capture')),
    canonical_uri TEXT NOT NULL,
    title TEXT,
    content_hash TEXT NOT NULL,
    mime_type TEXT,
    language TEXT,
    dataset TEXT NOT NULL,
    project TEXT,
    current_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(canonical_uri, dataset),
    FOREIGN KEY(current_version_id) REFERENCES document_versions(id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS document_versions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    source_mtime TEXT,
    git_repository TEXT,
    git_commit TEXT,
    doi TEXT,
    authors_json TEXT NOT NULL DEFAULT '[]',
    publication_date TEXT,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    index_version TEXT NOT NULL,
    raw_snapshot_path TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'parsed', 'indexed', 'failed', 'superseded')),
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(document_id, content_hash, index_version)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_version_id TEXT NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    text_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count > 0),
    chunk_type TEXT NOT NULL,
    heading_path_json TEXT NOT NULL DEFAULT '[]',
    page_start INTEGER,
    page_end INTEGER,
    line_start INTEGER,
    line_end INTEGER,
    symbol TEXT,
    language TEXT,
    parent_chunk_id TEXT REFERENCES chunks(id),
    qdrant_point_id TEXT,
    cognee_ref TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(document_version_id, ordinal),
    CHECK (page_start IS NULL OR page_start >= 1),
    CHECK (page_end IS NULL OR page_end >= page_start),
    CHECK (line_start IS NULL OR line_start >= 1),
    CHECK (line_end IS NULL OR line_end >= line_start)
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY,
    requested_uri TEXT NOT NULL,
    dataset TEXT NOT NULL,
    options_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'partial', 'failed')),
    documents_seen INTEGER NOT NULL DEFAULT 0,
    documents_changed INTEGER NOT NULL DEFAULT 0,
    chunks_indexed INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS memory_promotions (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version_id TEXT NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    cognee_dataset TEXT NOT NULL,
    cognee_ref TEXT,
    payload_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(document_version_id, kind, payload_hash)
);

CREATE TABLE IF NOT EXISTS answer_traces (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    generator_model TEXT,
    filters_json TEXT,
    retrieved_chunk_ids TEXT NOT NULL,
    selected_chunk_ids TEXT NOT NULL,
    citation_ids_json TEXT NOT NULL,
    latency_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_versions_document ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(document_version_id);
CREATE INDEX IF NOT EXISTS idx_documents_dataset ON documents(dataset);
