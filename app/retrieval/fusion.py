"""Rank-safe reciprocal-rank fusion and chunk deduplication."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from .qdrant_store import VectorHit
from .cognee_adapter import GraphEvidence


@dataclass
class Candidate:
    chunk_id: UUID
    payload: dict
    fused_score: float = 0.0
    component_ranks: dict[str, int] = field(default_factory=dict)
    component_scores: dict[str, float] = field(default_factory=dict)
    reranker_score: float | None = None


def reciprocal_rank_fusion(results: dict[str, list[VectorHit]], *, k: int = 60) -> list[Candidate]:
    merged: dict[UUID, Candidate] = {}
    for component, hits in results.items():
        for rank, hit in enumerate(hits, 1):
            candidate = merged.setdefault(hit.chunk_id, Candidate(hit.chunk_id, hit.payload))
            candidate.fused_score += 1.0 / (k + rank)
            candidate.component_ranks[component] = rank
            candidate.component_scores[component] = hit.score
    return sorted(merged.values(), key=lambda item: (-item.fused_score, str(item.chunk_id)))


def merge_graph_evidence(
    candidates: list[Candidate], evidence: list[GraphEvidence], *, k: int = 60
) -> tuple[list[Candidate], list[GraphEvidence]]:
    """Attach source-resolved graph evidence; never create citation candidates from memory."""
    by_chunk = {item.chunk_id: item for item in candidates}
    unsupported: list[GraphEvidence] = []
    for rank, graph in enumerate(evidence, 1):
        candidate = by_chunk.get(graph.chunk_id) if graph.chunk_id else None
        if candidate is None and graph.document_id is not None:
            candidate = next((item for item in candidates if
                item.payload.get("document_id") == str(graph.document_id) and
                (graph.document_version_id is None or
                 item.payload.get("document_version_id") == str(graph.document_version_id))), None)
        if candidate is None or not graph.source_grade:
            unsupported.append(graph)
            continue
        candidate.fused_score += 1.0 / (k + rank)
        candidate.component_ranks["cognee"] = rank
        candidate.component_scores["cognee"] = graph.score
        candidate.payload.setdefault("graph_evidence", []).append({
            "text": graph.text, "score": graph.score, "cognee_ref": graph.cognee_ref,
            "source_grade": True,
        })
    return sorted(candidates, key=lambda item: (-item.fused_score, str(item.chunk_id))), unsupported
