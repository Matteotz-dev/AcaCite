# Local Qdrant operations

The default deployment is persistent Qdrant Local Mode at `data/qdrant/`.
Dense and sparse embeddings run on CPU. This avoids another daemon and leaves
GPU memory to Ollama, but only one process may open the embedded store at a
time. Run ingestion in the API/worker process, or stop the API before using a
standalone indexing CLI.

Set `QDRANT_URL=http://127.0.0.1:6333` to use a Qdrant server when concurrent
process access is needed. `QDRANT_PATH` is ignored in server mode.

The `research_chunks_v1` collection contains named `dense` (384-dimensional,
cosine) and `sparse` vectors. Startup/indexing rejects an existing collection
whose dense dimensions or sparse vector name do not match configuration.
Changing the embedding model or dimensions requires a new collection and
`INDEX_VERSION`; it must never silently reuse the old vector space.

Index promotion is ordered as follows:

1. Embed and upsert every chunk in an immutable parsed version.
2. Count points filtered by `document_version_id`.
3. Record point IDs and atomically mark the version current in SQLite.
4. If any step fails, delete only that version's points and mark it failed.

The previous current version remains searchable until the replacement passes
the point-count gate. `.cognee_data/` is unrelated and must never be reset by
Qdrant maintenance.

## Cognee fusion and promotion

`/v1/search` queries Cognee through a timeout-bounded adapter. A timeout or
Cognee failure sets `trace.cognee_status` to `degraded`; dense/sparse Qdrant
retrieval continues normally. Tune `COGNEE_SEARCH_TIMEOUT_SECONDS` rather than
allowing graph retrieval to hold the whole request indefinitely.

Graph evidence is citation-grade only after its embedded `document_id` and
version/chunk identifiers resolve to the current SQLite provenance ledger.
Resolved evidence is attached to the matching Qdrant candidate and never
duplicates its text as another result. Unresolved durable memory is returned
only under `trace.memory_without_source` with `source_grade: false`.

Use `app.ingestion.promotion.promote_memory` for deliberate summaries, claims,
decisions, entity sets, and architecture notes. It adds a provenance marker,
hashes the entire payload, calls Cognee, and records completion in
`memory_promotions`. Repeating the same version/kind/payload is a no-op. It
does not cognify every raw Qdrant chunk. Promotion failures are not recorded as
successes and never modify the Qdrant current-version pointer.

## Grounded generation and citations

`POST /v1/answer` runs the same hybrid retrieval pipeline and then calls Ollama
at `OLLAMA_BASE_URL`. The request may select any installed Ollama model with
`model`; this choice is request-scoped and does not reindex data or change the
Cognee extraction model. Generation is non-streaming and bounded by
`OLLAMA_GENERATION_TIMEOUT_SECONDS`. A connection, timeout, HTTP, or malformed
response failure returns HTTP 503 with code `ollama_unavailable`.

The server assigns each packed chunk an opaque `[SRC:<chunk UUID>]` token before
generation. Returned tokens are accepted only if they match a supplied chunk
and resolve through SQLite. Invented or malformed tokens are returned under
`invalid_citations` and are never remapped. `warnings` contains a conservative
uncited-prose heuristic; it is an audit signal, not a proof of claim support.

`GET /v1/sources/{chunk_id}` resolves citation display and source metadata. A
local `open_path` is exposed only when the canonical file URI resolves inside
`APPROVED_INGESTION_ROOTS`; otherwise it is marked blocked. Successful and
deterministic no-evidence answers write an `answer_traces` row containing only
observable request/retrieval/citation/latency data. Prompts and hidden model
reasoning are not persisted.

## MCP transport

`cognee-research-mcp.service` exposes exactly six tools over MCP
Streamable HTTP at `127.0.0.1:8001/mcp`. It is a thin client of the versioned
API on `127.0.0.1:8000`; it owns no retrieval, citation, or ingestion state.
Consequently, API validation and SQLite source authorization remain the single
security boundary. See `docs/CONTINUE_MCP.md` for the verified Continue 2.0.0
configuration and SSH topology.

## Failure recovery and evaluation

`POST /v1/search` reports required Qdrant failures as HTTP 503 with code
`retrieval_unavailable`. Check `systemctl --user status`, available disk space,
collection configuration, and whether another process opened embedded Qdrant;
then restart the API. Do not delete the store as a recovery shortcut. Cognee
timeouts are nonfatal and appear in the search trace. Ollama is irrelevant to
search; restore the SSH port forward/server before retrying `/v1/answer`.

After recovery, run the full suite and offline acceptance evaluation described
in `docs/EVALUATION.md`. Back up `data/provenance.sqlite3`, `data/qdrant/`, and
`.cognee_data/` before migrations. Restore them as a matched snapshot while
services are stopped; mismatched SQLite/Qdrant versions must be reindexed from
the preserved sources rather than force-promoted.

## Corpus ingestion and synchronization

The operator CLI talks to the running API so the embedded Qdrant store always
has a single owner. Configure `APPROVED_INGESTION_ROOTS` narrowly, restart the
API after changing it, and then use:

```bash
rag ingest-repo /absolute/repo --dataset DATASET --project PROJECT
rag ingest-dir /absolute/papers --dataset DATASET --project PROJECT
rag sync /absolute/repo --dataset DATASET --project PROJECT
rag sync /absolute/papers --kind directory --dataset DATASET --project PROJECT
rag status
rag status --job JOB_ID
rag retry JOB_ID
```

Initial ingestion never interprets absent files as deletions. `sync` does:
after a successful discovery pass, indexed sources below that root which are
now absent are tombstoned and their current Qdrant points removed. Unchanged
content is skipped by hash. A failed replacement does not displace the prior
current version. Each file action has a durable complete/failed record; retry
creates a new job containing only failed paths and links it through
`options.retry_of`.

If a process interruption leaves a job marked `running`, rerun the same ingest
or sync command: content-addressed versions and point IDs make it idempotent.
For a failed/partial completed job, prefer `rag retry JOB_ID`. Inspect API logs
with `journalctl --user -u acacite-api.service -n 100`.
