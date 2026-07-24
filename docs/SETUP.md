# Setup

This guide starts AcaCite on localhost, ingests a small corpus, and exposes the
retrieval-only MCP server to an agent client.

## Install

```bash
git clone https://github.com/Matteotz-dev/AcaCite.git
cd AcaCite
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` before ingesting real files. In particular, set
`APPROVED_INGESTION_ROOTS` to a narrow JSON list of directories AcaCite is
allowed to read.

## Start The API

```bash
.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/v1/health
```

## Start The MCP Server

```bash
.venv/bin/acacite-mcp
```

The MCP endpoint is:

```text
http://127.0.0.1:8001/mcp
```

Available MCP tools are retrieval-only:

```text
research_search
research_remember
research_open_source
research_related
research_expand_citations
research_status
```

There is intentionally no MCP answer-generation tool. Agent clients should call
`research_search`, inspect the returned evidence, and write the final answer
with their own model.

## Ingest Content

```bash
.venv/bin/acacite ingest-dir examples/corpus \
  --dataset examples --project quickstart
```

For repositories:

```bash
.venv/bin/acacite ingest-repo /path/to/research-repo \
  --dataset research-code --project my-project
```

For paper trees:

```bash
.venv/bin/acacite ingest-dir /path/to/papers \
  --dataset papers --project my-project
```

Check jobs:

```bash
.venv/bin/acacite status
.venv/bin/acacite status --job JOB_ID
.venv/bin/acacite doctor
```

## Query Evidence

```bash
curl -s http://127.0.0.1:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does AcaCite preserve for citations?","dataset":"examples"}'
```

## Optional Answer Endpoint

`POST /v1/answer` is a direct HTTP endpoint that calls the configured local
Ollama generator. It is separate from MCP and is not used by the retrieval-only
agent workflow.

## User Services

Example `systemd --user` service files live in `deploy/systemd/`.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/acacite-api.service ~/.config/systemd/user/
cp deploy/systemd/acacite-mcp.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now acacite-api.service
systemctl --user enable --now acacite-mcp.service
```

The examples assume the repository is checked out at `~/AcaCite`. Edit
`WorkingDirectory`, `EnvironmentFile`, and `ExecStart` if your path differs.

## Public Exposure

AcaCite is designed for localhost use. Do not expose ports `8000` or `8001` to
the public internet. If you must bind outside localhost, set `ACACITE_API_TOKEN`
and put the service behind normal network access controls.
