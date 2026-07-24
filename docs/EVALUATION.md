# Evaluation and tuning

Phase 8 adds a versioned, offline-first evaluation contract. The committed
`evaluation/gold/v1.jsonl` contains 60 fixed cases: 15 exact-code, 15 paper,
10 cross-source, 10 graph/multi-hop, and 10 insufficient-evidence cases. The
IDs in this first fixture are deterministic synthetic provenance IDs so CI can
verify metric and regression behavior without accessing a private corpus.
They are not evidence that retrieval quality on a future corpus is perfect.

Run the reproducible five-way fixture ablation:

```bash
.venv/bin/python scripts/build_eval_fixture.py
.venv/bin/python scripts/evaluate.py \
  --gold evaluation/gold/v1.jsonl \
  --records evaluation/fixtures/v1_ablation_records.jsonl \
  --output evaluation/reports/v1_offline_ablation.json
```

The evaluator reports Recall@5/10, MRR, nDCG@10, current-version correctness,
citation precision/validity, required-fact coverage, unsupported-claim rate,
abstention accuracy, p50/p95 component latency, end-to-end latency, and peak
CPU/GPU memory. Generation fields are optional, so routine regression tests do
not require Ollama.

For a corpus-quality run, copy the schema to a new version, replace synthetic
IDs with reviewed current SQLite IDs, record each strategy against the same
frozen cases, and keep the resulting report. Never tune and score on different
question sets under the same version.

## Recommended defaults

Keep the present production defaults until a real-corpus report supports a
change:

| Setting | Default | Rationale / next sweep |
|---|---:|---|
| Dense candidates | 40 | Sweep 20/40/80; retain smallest Recall@10 plateau. |
| Sparse candidates | 40 | Same sweep, with exact-symbol subset reported separately. |
| Fusion | RRF, k=60 | Stable across incomparable dense/sparse score scales. |
| Rerank candidates | 30 | Sweep 15/30/60 and compare nDCG@10 against p95 latency. |
| Graph candidates | 10 | Graph remains additive and provenance-gated. |
| Graph timeout | 3 s | A timeout degrades to document retrieval, never blocks citations. |
| Final context | 10 chunks | At most three chunks per document limits dominance. |
| Context budget | 12,000 tokens | Sweep 6k/9k/12k; avoid consuming Ollama's full context. |
| Chunking | structural | Preserve page, heading, symbol, and line boundaries before size tuning. |

The fixture report selects hybrid plus reranking: Recall@5 is 1.00 versus 0.68
dense-only and 0.50 sparse-only, while citation precision, citation validity,
and current-version correctness are 1.00. These numbers validate the evaluator
and acceptance wiring only. Graph fusion ties the selected path on the
synthetic set, so there is no evidence to pay its latency on every query;
retain its timeout-bounded optional behavior.

## Resource baseline

On 2026-07-22, with an empty production index and no model loaded, systemd
reported API RSS 293.6 MiB (295.4 MiB peak), MCP RSS 45.0 MiB (45.4 MiB peak),
and the RTX 5090 reported 34 MiB of 32,607 MiB used. Offline fixture p95 was
16.705 ms retrieval and 5 ms reranking. Synthetic timings are regression
fixtures, not hardware benchmarks; record live p50/p95 and `nvidia-smi` peaks
after the reviewed corpus is populated and for each chosen generator.

Acceptance thresholds for the reviewed corpus are citation precision >= 0.95,
citation validity = 1.00, current-version correctness >= 0.95, and abstention
accuracy >= 0.90. A release must either show hybrid+rerank Recall@5 beating
both single retrievers or document and adopt the simpler winner.
