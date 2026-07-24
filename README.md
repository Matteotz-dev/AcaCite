# AcaCite

AcaCite is a local, citation-oriented research memory system. It indexes
papers, notes, and source repositories into a provenance-tracked retrieval
store, then exposes evidence through HTTP and a retrieval-only MCP interface
for coding agents.

The core system includes typed local configuration, a transactional SQLite
provenance registry, stable document/version/chunk identities, approved-root
path protection, opaque citation resolution, deterministic text/Markdown/code
parsers, PDF ingestion, Git-aware repository discovery, persistent Qdrant
dense+sparse indexing, reciprocal-rank fusion, reranking, context packing,
optional Cognee promotion, and degraded-mode retrieval.

The MCP surface is intentionally retrieval-only. Agent clients such as Codex,
Claude, Continue, or other MCP consumers inspect the returned evidence and
synthesize answers themselves, without invoking a local answer model through
MCP.

## Why AcaCite?

AcaCite keeps citation provenance in a small local system while leaving answer
writing to the agent you already trust. The API owns ingestion, chunk identity,
safe source paths, retrieval, and optional local generation. The MCP server is a
thin retrieval transport: it returns evidence and source metadata, and it never
routes requests through the local answer model.

```text
papers/repos/notes -> SQLite provenance + Qdrant retrieval -> MCP evidence
                                                           -> Codex/Claude answer
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env
```

See [docs/SETUP.md](docs/SETUP.md) for the full local setup, service startup,
MCP client, ingestion, and query workflow.
See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a minimal example corpus
walkthrough.
See [docs/DEMO.md](docs/DEMO.md) for a public demo using only the included
example corpus.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the HTTP API versus
retrieval-only MCP split.
See [docs/CITATION_EXPANSION.md](docs/CITATION_EXPANSION.md) for pulling cited
papers and second-hop cited papers from an indexed bibliography.

Run the end-to-end test:

```bash
.venv/bin/python smoke_test.py
```

The smoke test uses an operating-system temporary Cognee root and refuses to
run its destructive reset if that isolation check fails. It never resets the
durable `.cognee_data/` store.

All generated state is ignored by Git. Qdrant/SQLite state belongs under
`data/`, and optional Cognee graph state belongs under `.cognee_data/`.

## HTTP API

Start the localhost-only service:

```bash
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `POST /v1/search`: hybrid Qdrant evidence search with provenance and a
  component-rank/latency trace plus optional Cognee graph evidence.
- `POST /v1/answer`: request-scoped Ollama generation grounded only in packed
  search evidence; returns resolved and invalid citations separately.
- `GET /v1/sources/{chunk_id}`: resolve an opaque chunk citation through the
  SQLite provenance authority, including a safe local open path when approved.
- `POST /v1/memory`: explicitly index a note with a stable user-supplied
  provenance URI and optional controlled Cognee promotion.
- `POST /v1/related`: find evidence related to an authorized indexed chunk.
- `GET /v1/health`: report index identity and component status without making
  Ollama availability a search prerequisite.
- `GET /docs`: interactive API documentation.

Legacy `/remember`, `/retrieve`, and `/answer` endpoints remain for local
compatibility. New clients should use `/v1/*`.

Changing or adding an Ollama model does not require re-ingestion. Agent clients
consume the same `/v1/search` results through MCP.

You can also install the API and MCP server as user services, for example:

```bash
systemctl --user status acacite-api.service
systemctl --user restart acacite-api.service
```

Example user service files are in [deploy/systemd](deploy/systemd).

The separate localhost-only MCP transport is
`acacite-mcp.service` at `http://127.0.0.1:8001/mcp`. Its tools are
retrieval-only; agent clients synthesize answers themselves from returned
evidence and do not invoke the local Ollama generator through MCP.

## Populate and synchronize the citation-grade corpus

The `rag` operator CLI calls the running API, so embedded Qdrant remains owned
by one process. Paths must be inside `APPROVED_INGESTION_ROOTS`.

```bash
./rag ingest-repo /path/to/research-repo \
  --dataset research-code --project example-project
./rag ingest-dir /path/to/papers \
  --dataset papers --project example-project
./rag status
./rag status --job JOB_ID
./rag retry JOB_ID
./rag sync /path/to/research-repo \
  --dataset research-code --project example-project
./rag doctor
```

`ingest-repo` respects Git and `.gitignore`; `ingest-dir` recursively selects
supported source formats and ignores hidden directories. Repeated ingestion
skips unchanged content. `sync` additionally tombstones missing sources and
deletes only their current Qdrant points. Every file operation and error is
recorded under its job, and retry creates a linked job containing only prior
failures. Use `--kind directory` when synchronizing a non-Git paper tree.
