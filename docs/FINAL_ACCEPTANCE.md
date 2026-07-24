# Final acceptance evidence

## Automated gates

- The v1 gold set has exactly 60 schema-validated cases in the required category
  distribution; five ablation strategies produce 300 deterministic records.
- Hybrid+rerank beats dense-only and sparse-only Recall@5 in the fixed fixture.
- Fixture citation precision/validity and correct-version rate are 1.00;
  abstention accuracy is 1.00. Metric unit tests inject wrong versions,
  invented citations, bad ranks, and unsupported claims to ensure failures are
  counted rather than hidden.
- Integration tests cover PDF and Git/code ingestion, unchanged and changed
  re-ingestion, dense+sparse contribution, reranker CUDA-to-CPU fallback,
  Cognee degradation, Qdrant failure normalization, Ollama degradation,
  invented citation rejection, and generator-independent evidence.
- MCP exposes exactly six tools and switching Qwen/Devstral does not mutate
  current indexed versions.

## Operational gates

- API and MCP services bind only to `127.0.0.1`; Continue reaches MCP on the
  remote VS Code host.
- SQLite remains the citation/version authority. Qdrant points cannot create a
  valid citation without a current SQLite mapping, and unmapped Cognee memory
  is explicitly non-source-grade.
- Search does not require Ollama. Cognee failures return a degraded trace;
  Qdrant failures return HTTP 503 `retrieval_unavailable`; Ollama failures on
  answer return HTTP 503 `ollama_unavailable`.
- Tests and fixture generation use isolated stores. Durable `.cognee_data/` is
  never reset or deleted, and no corpus contents are sent externally.

## Remaining real-corpus evidence

The production collection was intentionally left empty during implementation.
Consequently, the software release and deterministic acceptance gates are
complete, while empirical relevance, live generator latency/VRAM, and manual
unsupported-claim review must be recorded after the user chooses and ingests a
reviewed paper/repository corpus. Use `docs/EVALUATION.md`; do not describe the
synthetic fixture as real retrieval quality.
