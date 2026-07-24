# Quickstart

This walkthrough ingests the committed example corpus and runs a retrieval
query.

## 1. Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` and include this repository in `APPROVED_INGESTION_ROOTS`.

## 2. Start The API

```bash
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

## 3. Ingest Examples

```bash
.venv/bin/acacite ingest-dir examples/corpus \
  --dataset examples --project quickstart
```

## 4. Search

```bash
curl -s http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does the MCP server do?","dataset":"examples"}'
```

The response contains evidence chunks with `citation_id`, source path, heading,
page/line metadata where available, and relevance scores.

## 5. MCP

Start the MCP server:

```bash
.venv/bin/acacite-mcp
```

Point your MCP client at:

```text
http://127.0.0.1:8001/mcp
```
