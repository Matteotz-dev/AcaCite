# Citation Expansion

AcaCite can expand an indexed paper bibliography into a first-hop and second-hop
citation manifest. This is useful when you want to pull the papers cited by a
source paper, plus the papers cited by those cited papers.

The expander uses:

- the local AcaCite SQLite provenance database for seed bibliography chunks,
- Crossref to resolve raw references into DOI/title metadata,
- OpenAlex to fetch referenced-work links and open-access locations,
- optional open-access PDF download.

It does not download paywalled PDFs. PDF download only happens when OpenAlex
reports a public PDF URL and the response looks like a PDF.

## Run

```bash
acacite-expand-citations \
  --database data/provenance.sqlite3 \
  --paper-title "My Source Paper Title" \
  --output reports/citation-expansion/source-paper \
  --depth 2 \
  --download-oa-pdfs
```

The same workflow is available through the main CLI:

```bash
acacite expand-citations \
  --paper-title "My Source Paper Title" \
  --output reports/citation-expansion/source-paper \
  --depth 2 \
  --download-oa-pdfs
```

Outputs:

```text
reports/citation-expansion/source-paper.jsonl
reports/citation-expansion/source-paper.md
reports/citation-expansion/pdfs/*.pdf
```

Use `--paper-title` more than once when the seed corpus has multiple source
papers:

```bash
acacite-expand-citations \
  --database data/provenance.sqlite3 \
  --paper-title "First Paper" \
  --paper-title "Second Paper" \
  --output reports/citation-expansion/project \
  --depth 2 \
  --download-oa-pdfs
```

## Ingest Downloaded PDFs

After reviewing the downloaded files:

```bash
acacite ingest-dir reports/citation-expansion/pdfs \
  --dataset citation-expansion --project my-project
```

Or ask the main CLI to submit the downloaded PDF directory to the running API:

```bash
acacite expand-citations \
  --paper-title "My Source Paper Title" \
  --output reports/citation-expansion/source-paper \
  --depth 2 \
  --download-oa-pdfs \
  --ingest-downloaded \
  --dataset citation-expansion --project my-project
```

MCP clients can call `research_expand_citations` for the same metadata/PDF
discovery path. That MCP tool is retrieval-only and does not invoke Ollama.

Keep the output directory under an approved ingestion root or update
`APPROVED_INGESTION_ROOTS` before ingesting.

## Review First

For a metadata-only run:

```bash
acacite-expand-citations \
  --database data/provenance.sqlite3 \
  --paper-title "My Source Paper Title" \
  --output reports/citation-expansion/source-paper \
  --depth 2
```

The JSONL manifest includes each work's depth, DOI, title, OpenAlex ID,
referenced works, OA URL, PDF URL, and local PDF path if downloaded.
