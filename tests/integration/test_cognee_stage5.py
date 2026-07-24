from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.db import ProvenanceRepository
from app.ingestion.promotion import CogneePromoter, promote_memory
from app.ingestion.service import ingest_file
from app.retrieval.cognee_adapter import CogneeSearchResult, GraphEvidence
from app.retrieval.service import RetrievalService
from tests.integration.test_hybrid_index import FixtureDense, FixtureSparse
from app.ingestion.indexing import index_version
from app.retrieval.qdrant_store import QdrantStore
from qdrant_client import QdrantClient


class FixturePromoter(CogneePromoter):
    def __init__(self):
        self.calls = 0

    async def promote(self, payload: str, dataset: str) -> str:
        self.calls += 1
        assert "RAG_PROVENANCE" in payload
        return "fixture-cognee-ref"


def _rig(tmp_path: Path):
    settings = Settings(
        rag_data_root=tmp_path / "data", provenance_db_path=tmp_path / "provenance.sqlite3",
        qdrant_path=tmp_path / "qdrant", qdrant_collection="stage5", dense_embedding_dimensions=4,
        approved_ingestion_roots=(tmp_path,), dense_candidates=10, sparse_candidates=10,
        rerank_candidates=10,
    )
    repository = ProvenanceRepository(settings.provenance_db_path)
    store = QdrantStore(QdrantClient(":memory:"), settings.qdrant_collection, 4)
    return settings, repository, store


def test_controlled_promotion_is_payload_hash_idempotent(tmp_path: Path):
    settings, repository, _ = _rig(tmp_path)
    note = tmp_path / "paper.md"
    note.write_text("# Abstract\nA curated claim.", encoding="utf-8")
    version, chunks = ingest_file(note, dataset="papers", settings=settings)
    promoter = FixturePromoter()
    first = asyncio.run(promote_memory(
        repository=repository, version_id=version.id, kind="claim", text="A curated claim.",
        cognee_dataset="rag_papers", chunk_id=chunks[0].id, promoter=promoter,
    ))
    second = asyncio.run(promote_memory(
        repository=repository, version_id=version.id, kind="claim", text="A curated claim.",
        cognee_dataset="rag_papers", chunk_id=chunks[0].id, promoter=promoter,
    ))
    assert promoter.calls == 1
    assert second.already_promoted is True
    assert first.promotion_id == second.promotion_id
    assert repository.get_chunk(chunks[0].id).cognee_ref == "fixture-cognee-ref"


def test_qdrant_search_survives_cognee_degraded(tmp_path: Path):
    settings, repository, store = _rig(tmp_path)
    note = tmp_path / "source.md"
    note.write_text("# Solver\nfluxCorrector_X9 is citation-grade evidence.", encoding="utf-8")
    version, chunks = ingest_file(note, dataset="code", settings=settings)
    dense, sparse = FixtureDense(), FixtureSparse()
    index_version(version.id, repository=repository, store=store, dense=dense, sparse=sparse)

    class Unavailable:
        def search(self, query, *, datasets, limit):
            return CogneeSearchResult(status="degraded", error="connection refused")

    response = RetrievalService(
        settings=settings, repository=repository, store=store, dense=dense, sparse=sparse,
        graph=Unavailable(),
    ).search("fluxCorrector_X9")
    assert response.results[0].chunk_id == chunks[0].id
    assert response.trace["cognee_status"] == "degraded"
    assert response.trace["cognee_error"] == "connection refused"


def test_source_graph_evidence_merges_and_unmapped_memory_stays_uncited(tmp_path: Path):
    settings, repository, store = _rig(tmp_path)
    note = tmp_path / "source.md"
    note.write_text("# Method\nA solver relationship is documented here.", encoding="utf-8")
    version, chunks = ingest_file(note, dataset="papers", settings=settings)
    dense, sparse = FixtureDense(), FixtureSparse()
    index_version(version.id, repository=repository, store=store, dense=dense, sparse=sparse)

    class Graph:
        def search(self, query, *, datasets, limit):
            return CogneeSearchResult(evidence=(
                GraphEvidence("supported relation", 0.9, version.document_id, version.id,
                              chunks[0].id, "supported-ref", True),
                GraphEvidence("unattributed memory", 0.8, cognee_ref="memory-ref"),
            ))

    response = RetrievalService(
        settings=settings, repository=repository, store=store, dense=dense, sparse=sparse,
        graph=Graph(),
    ).search("solver relationship")
    assert len(response.results) == 1
    assert response.results[0].component_ranks["cognee"] == 1
    assert response.results[0].payload["graph_evidence"][0]["cognee_ref"] == "supported-ref"
    assert response.trace["memory_without_source"] == [{
        "text": "unattributed memory", "cognee_ref": "memory-ref", "source_grade": False,
    }]
