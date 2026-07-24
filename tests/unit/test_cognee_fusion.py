from __future__ import annotations

import asyncio
from uuid import uuid4

from app.retrieval.cognee_adapter import CogneeAdapter, GraphEvidence, normalize_evidence
from app.retrieval.fusion import Candidate, merge_graph_evidence


def test_normalizes_embedded_provenance_marker():
    document_id, version_id, chunk_id = uuid4(), uuid4(), uuid4()
    evidence = normalize_evidence([{
        "text": (f"[RAG_PROVENANCE document_id={document_id} version_id={version_id} "
                 f"chunk_id={chunk_id}]\nA supported claim"),
        "score": 0.8, "id": "memory-1",
    }])[0]
    assert evidence.document_id == document_id
    assert evidence.document_version_id == version_id
    assert evidence.chunk_id == chunk_id
    assert evidence.source_grade is True


def test_graph_evidence_merges_without_duplicate_candidate():
    document_id, version_id, chunk_id = uuid4(), uuid4(), uuid4()
    candidate = Candidate(chunk_id, {
        "document_id": str(document_id), "document_version_id": str(version_id), "text": "source"
    })
    graph = GraphEvidence("relationship", 0.7, document_id, version_id, chunk_id,
                          "memory-1", True)
    merged, unsupported = merge_graph_evidence([candidate], [graph])
    assert len(merged) == 1
    assert unsupported == []
    assert merged[0].component_ranks["cognee"] == 1
    assert merged[0].payload["graph_evidence"][0]["source_grade"] is True


def test_unmapped_memory_is_labeled_not_promoted_to_candidate():
    memory = GraphEvidence("personal research preference", cognee_ref="memory-2")
    merged, unsupported = merge_graph_evidence([], [memory])
    assert merged == []
    assert unsupported == [memory]


def test_adapter_returns_degraded_on_timeout():
    class Slow(CogneeAdapter):
        async def _search_async(self, query, *, datasets, limit):
            await asyncio.sleep(0.1)
            return []

    result = Slow(timeout_seconds=0.01).search("query", datasets=None, limit=3)
    assert result.status == "degraded"
    assert "TimeoutError" in result.error
