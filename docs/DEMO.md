# Public Demo

This repository includes a small non-private corpus in `examples/corpus/`.
It is intended to verify a fresh install without exposing personal papers,
passwords, generated vector stores, or local model state.

Start the API:

```bash
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

In another terminal, ingest the demo corpus:

```bash
.venv/bin/acacite ingest-dir examples/corpus \
  --dataset examples --project public-demo
```

Check the install:

```bash
.venv/bin/acacite doctor
.venv/bin/acacite status
```

Query evidence:

```bash
curl -s http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does AcaCite preserve for citations?","dataset":"examples"}'
```

Start MCP:

```bash
.venv/bin/acacite-mcp
```

Point an MCP client at:

```text
http://127.0.0.1:8001/mcp
```

The demo path exercises retrieval and source provenance only. The optional
HTTP `/v1/answer` endpoint requires Ollama and is not used by MCP.
