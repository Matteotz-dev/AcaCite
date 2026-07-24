"""Pure offline metrics; no embedding, graph, Qdrant, or generator is required."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Iterable

from .schema import GoldCase, RetrievalRecord


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _ndcg(case: GoldCase, ids: list[str], k: int = 10) -> float:
    grades = case.graded_relevance or {item: 1 for item in case.relevant_chunk_ids}
    dcg = sum((2 ** grades.get(item, 0) - 1) / math.log2(rank + 1)
              for rank, item in enumerate(ids[:k], 1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def evaluate_records(cases: Iterable[GoldCase], records: Iterable[RetrievalRecord]) -> dict:
    gold = {case.id: case for case in cases}
    grouped: dict[str, list[RetrievalRecord]] = defaultdict(list)
    for record in records:
        if record.case_id not in gold:
            raise ValueError(f"record references unknown case {record.case_id}")
        grouped[record.strategy].append(record)
    report = {"schema_version": "1.0", "gold_cases": len(gold), "strategies": {}}
    for strategy, runs in sorted(grouped.items()):
        recall5, recall10, reciprocal, ndcg, versions = [], [], [], [], []
        citation_precision, citation_validity, fact_coverage, abstention = [], [], [], []
        unsupported, retrieval, reranking, end_to_end, cpu, gpu = [], [], [], [], [], []
        for run in runs:
            case = gold[run.case_id]
            ids = [hit.chunk_id for hit in run.hits]
            relevant = set(case.relevant_chunk_ids)
            if relevant:
                recall5.append(len(relevant.intersection(ids[:5])) / len(relevant))
                recall10.append(len(relevant.intersection(ids[:10])) / len(relevant))
                ranks = [ids.index(item) + 1 for item in relevant if item in ids]
                reciprocal.append(1 / min(ranks) if ranks else 0.0)
                ndcg.append(_ndcg(case, ids))
                relevant_hits = [hit for hit in run.hits if hit.chunk_id in relevant]
                versions.append(mean(hit.document_version_id in case.relevant_version_ids
                                     for hit in relevant_hits) if relevant_hits else 0.0)
            cited = run.cited_chunk_ids
            if cited or run.invalid_citation_ids:
                citation_precision.append(sum(item in relevant for item in cited) /
                                          max(1, len(cited) + len(run.invalid_citation_ids)))
                citation_validity.append(len(cited) /
                                         max(1, len(cited) + len(run.invalid_citation_ids)))
            if case.required_facts and run.answer is not None:
                normalized = run.answer.casefold()
                fact_coverage.append(mean(fact.casefold() in normalized
                                          for fact in case.required_facts))
            if run.abstained is not None:
                abstention.append(run.abstained == case.must_abstain)
            claims = (run.supported_claims or 0) + (run.unsupported_claims or 0)
            if claims:
                unsupported.append((run.unsupported_claims or 0) / claims)
            retrieval.append(run.retrieval_ms)
            reranking.append(run.rerank_ms)
            if run.generation_ms is not None:
                end_to_end.append(run.retrieval_ms + run.rerank_ms + run.generation_ms)
            if run.peak_cpu_ram_mb is not None: cpu.append(run.peak_cpu_ram_mb)
            if run.peak_gpu_vram_mb is not None: gpu.append(run.peak_gpu_vram_mb)
        avg = lambda values: mean(values) if values else None
        report["strategies"][strategy] = {
            "cases": len(runs), "recall_at_5": avg(recall5), "recall_at_10": avg(recall10),
            "mrr": avg(reciprocal), "ndcg_at_10": avg(ndcg),
            "correct_version_rate": avg(versions), "citation_precision": avg(citation_precision),
            "citation_validity": avg(citation_validity), "required_fact_coverage": avg(fact_coverage),
            "unsupported_claim_rate": avg(unsupported), "abstention_accuracy": avg(abstention),
            "retrieval_ms": {"p50": _percentile(retrieval, .5), "p95": _percentile(retrieval, .95)},
            "rerank_ms": {"p50": _percentile(reranking, .5), "p95": _percentile(reranking, .95)},
            "end_to_end_ms": {"p50": _percentile(end_to_end, .5), "p95": _percentile(end_to_end, .95)},
            "peak_cpu_ram_mb": max(cpu) if cpu else None, "peak_gpu_vram_mb": max(gpu) if gpu else None,
        }
    return report
