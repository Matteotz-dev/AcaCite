#!/usr/bin/env python3
"""Create the immutable v1 acceptance fixture and five deterministic ablations."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evaluation/gold/v1.jsonl"
RECORDS = ROOT / "evaluation/fixtures/v1_ablation_records.jsonl"

QUERIES = {
    "exact_code": [
        "Where is {name} defined and what does it return?",
        "Which configuration sets {name}, and what is its value?",
        "Find the call site that updates {name}.",
    ],
    "paper_fact": [
        "What assumption does the paper make about {name}?",
        "Which equation defines {name}, and under what conditions?",
        "What dataset is used to validate {name}?",
    ],
    "cross_source": [
        "Compare the implementation of {name} with the paper's stated method.",
        "Does the repository configuration match the published value for {name}?",
    ],
    "graph_multihop": [
        "How does {name} connect the cited method, implementation, and experiment?",
        "Trace the evidence chain from {name} to the reported result.",
    ],
    "insufficient_evidence": [
        "What result does the unavailable future version report for {name}?",
        "Give the undocumented private setting for {name}.",
    ],
}

COUNTS = {"exact_code": 15, "paper_fact": 15, "cross_source": 10,
          "graph_multihop": 10, "insufficient_evidence": 10}


def uid(kind: str, number: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"research-rag-eval-v1:{kind}:{number}"))


def main() -> None:
    cases, records = [], []
    sequence = 0
    for category, count in COUNTS.items():
        for number in range(1, count + 1):
            sequence += 1
            prefix = {"exact_code": "code", "paper_fact": "paper",
                      "cross_source": "cross", "graph_multihop": "graph",
                      "insufficient_evidence": "abstain"}[category]
            case_id = f"{prefix}-{number:03d}"
            name = f"fixture_{prefix}_{number:02d}"
            abstain = category == "insufficient_evidence"
            chunk = uid("chunk", sequence)
            document = uid("document", sequence)
            version = uid("version", sequence)
            case = {
                "schema_version": "1.0", "id": case_id,
                "query": QUERIES[category][(number - 1) % len(QUERIES[category])].format(name=name),
                "category": category,
                "filters": {"project": "fixture", "dataset": "evaluation-v1"},
                "relevant_document_ids": [] if abstain else [document],
                "relevant_chunk_ids": [] if abstain else [chunk],
                "relevant_version_ids": [] if abstain else [version],
                "graded_relevance": {} if abstain else {chunk: 3},
                "required_facts": [] if abstain else [name],
                "must_abstain": abstain,
                "notes": "Synthetic, source-shaped regression case; replace IDs with live corpus IDs for quality tuning.",
            }
            cases.append(case)
            distractors = [{"chunk_id": uid("distractor", sequence * 10 + rank),
                            "document_id": uid("distractor-doc", sequence * 10 + rank),
                            "document_version_id": uid("distractor-version", sequence * 10 + rank),
                            "score": 1 - rank / 20} for rank in range(1, 11)]
            relevant_hit = {"chunk_id": chunk, "document_id": document,
                            "document_version_id": version, "score": .99}
            for strategy in ("dense", "sparse", "hybrid", "hybrid_rerank", "hybrid_graph"):
                if abstain:
                    hits = distractors[:3]
                elif strategy == "dense":
                    hits = ([relevant_hit] + distractors[:9]) if sequence % 3 else distractors
                elif strategy == "sparse":
                    hits = ([relevant_hit] + distractors[:9]) if sequence % 2 else distractors
                elif strategy == "hybrid":
                    hits = distractors[:2] + [relevant_hit] + distractors[2:9]
                elif strategy == "hybrid_rerank":
                    hits = [relevant_hit] + distractors[:9]
                else:
                    hits = [relevant_hit] + distractors[:9]
                records.append({
                    "schema_version": "1.0", "case_id": case_id, "strategy": strategy,
                    "hits": hits, "answer": None if strategy in ("dense", "sparse", "hybrid")
                    else ("Insufficient evidence." if abstain else f"The evidence establishes {name}."),
                    "cited_chunk_ids": [] if abstain or strategy in ("dense", "sparse", "hybrid") else [chunk],
                    "invalid_citation_ids": [],
                    "abstained": abstain if strategy in ("hybrid_rerank", "hybrid_graph") else None,
                    "supported_claims": 0 if strategy in ("dense", "sparse", "hybrid") else 1,
                    "unsupported_claims": 0,
                    "retrieval_ms": 8 + sequence / 10 + {"dense": 1, "sparse": 1,
                        "hybrid": 3, "hybrid_rerank": 3, "hybrid_graph": 3}[strategy],
                    "rerank_ms": 5 if strategy in ("hybrid_rerank", "hybrid_graph") else 0,
                    "generation_ms": 20 if strategy in ("hybrid_rerank", "hybrid_graph") else None,
                    "peak_cpu_ram_mb": 256, "peak_gpu_vram_mb": 0,
                })
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    GOLD.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cases), encoding="utf-8")
    RECORDS.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    print(f"wrote {len(cases)} cases and {len(records)} records")


if __name__ == "__main__":
    main()
