# Architecture

AcaCite has two model-facing surfaces with different responsibilities.

## Retrieval-Only MCP

The MCP server is for agent clients such as Codex, Claude, Continue, or other
MCP consumers. It exposes evidence retrieval and source-resolution tools only:

```text
research_search
research_remember
research_open_source
research_related
research_status
```

The MCP server does not expose an answer-generation tool and does not call a
local LLM. The intended flow is:

1. The agent calls `research_search`.
2. AcaCite returns cited local evidence chunks.
3. The agent reads those chunks and synthesizes the answer with its own model.

This keeps local retrieval independent from local generation.

## HTTP API

The HTTP API exposes ingestion, search, source resolution, status, and optional
local answer generation:

```text
POST /v1/search
POST /v1/memory
POST /v1/related
GET  /v1/sources/{chunk_id}
GET  /v1/health
POST /v1/answer
```

`POST /v1/answer` is intentionally not part of MCP. It calls the configured
local Ollama generator after retrieval and citation packing. Use it only when
you explicitly want a local model to write the answer.

## Storage

AcaCite separates durable source provenance from vector retrieval:

- SQLite stores documents, versions, chunks, source metadata, ingestion jobs,
  and opaque citation IDs.
- Qdrant stores dense and sparse retrieval points keyed by chunk ID.
- Cognee promotion is optional and controlled by explicit payload hashes.

Generated state belongs under ignored local directories such as `data/` and
`.cognee_data/`; it should not be committed to source control.
