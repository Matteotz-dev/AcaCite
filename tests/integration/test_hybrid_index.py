from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.db import ProvenanceRepository
from app.ingestion.indexing import index_version
from app.ingestion.service import ingest_file
from app.models import VersionStatus
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.service import RetrievalService, SearchFilters
from app.retrieval.sparse import SparseEmbedding


class FixtureDense:
    model_name = "fixture-semantic"
    dimensions = 4

    @staticmethod
    def _one(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(any(word in lowered for word in ("car", "automobile", "vehicle"))),
            float(any(word in lowered for word in ("solver", "algorithm", "method"))),
            float("turbulence" in lowered), 1.0,
        ]

    def embed_documents(self, texts): return [self._one(text) for text in texts]
    def embed_query(self, text): return self._one(text)


class FixtureSparse:
    model_name = "fixture-bm25"

    @staticmethod
    def _one(text: str) -> SparseEmbedding:
        tokens = sorted(set(re.findall(r"[A-Za-z0-9_]+", text.lower())))
        indices = [int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) for token in tokens]
        return SparseEmbedding(indices, [1.0] * len(indices))

    def embed_documents(self, texts): return [self._one(text) for text in texts]
    def embed_query(self, text): return self._one(text)


@pytest.fixture
def rig(tmp_path: Path):
    settings = Settings(
        rag_data_root=tmp_path / "data", provenance_db_path=tmp_path / "provenance.sqlite3",
        qdrant_path=tmp_path / "qdrant", qdrant_collection="fixture_chunks_v1",
        dense_embedding_dimensions=4, approved_ingestion_roots=(tmp_path,),
        dense_candidates=10, sparse_candidates=10, rerank_candidates=10,
    )
    repository = ProvenanceRepository(settings.provenance_db_path)
    store = QdrantStore(QdrantClient(":memory:"), settings.qdrant_collection, 4)
    return settings, repository, store, FixtureDense(), FixtureSparse()


def _write_note(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_hybrid_exact_semantic_filters_and_idempotency(rig, tmp_path: Path):
    settings, repository, store, dense, sparse = rig
    exact = tmp_path / "exact.md"
    semantic = tmp_path / "semantic.md"
    _write_note(exact, "# Implementation\nThe fluxCorrector_X9 symbol stabilizes the solver.")
    _write_note(semantic, "# Transport\nAn automobile carries passengers on a highway.")

    first, first_chunks = ingest_file(exact, dataset="code", project="les", settings=settings)
    second, _ = ingest_file(semantic, dataset="papers", project="transport", settings=settings)
    index_version(first.id, repository=repository, store=store, dense=dense, sparse=sparse)
    index_version(second.id, repository=repository, store=store, dense=dense, sparse=sparse)

    service = RetrievalService(settings=settings, repository=repository, store=store,
                               dense=dense, sparse=sparse)
    exact_result = service.search("fluxCorrector_X9")
    assert exact_result.results[0].chunk_id == first_chunks[0].id
    assert "sparse" in exact_result.results[0].component_ranks
    semantic_result = service.search("vehicle", SearchFilters(dataset="papers"))
    assert semantic_result.results[0].payload["dataset"] == "papers"
    assert "dense" in semantic_result.results[0].component_ranks
    assert semantic_result.trace["dense_count"] >= 1

    unchanged, unchanged_chunks = ingest_file(exact, dataset="code", project="les", settings=settings)
    index_version(unchanged.id, repository=repository, store=store, dense=dense, sparse=sparse)
    assert unchanged.id == first.id
    assert [chunk.id for chunk in unchanged_chunks] == [chunk.id for chunk in first_chunks]
    assert store.count_version(first.id) == len(first_chunks)


def test_failed_reindex_cleans_only_new_version_and_keeps_previous_current(rig, tmp_path: Path):
    settings, repository, store, dense, sparse = rig
    note = tmp_path / "note.md"
    _write_note(note, "# V1\nworking solver")
    old, _ = ingest_file(note, dataset="notes", settings=settings)
    index_version(old.id, repository=repository, store=store, dense=dense, sparse=sparse)

    _write_note(note, "# V2\nchanged solver")
    new, _ = ingest_file(note, dataset="notes", settings=settings)

    class BrokenDense(FixtureDense):
        def embed_documents(self, texts): raise RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        index_version(new.id, repository=repository, store=store,
                      dense=BrokenDense(), sparse=sparse)
    assert repository.get_version(new.id).status is VersionStatus.FAILED
    assert repository.get_document(old.document_id).current_version_id == old.id
    assert store.count_version(old.id) > 0
    assert store.count_version(new.id) == 0


def test_collection_dimension_mismatch_fails_fast():
    client = QdrantClient(":memory:")
    wrong = QdrantStore(client, "mismatch", 3)
    wrong.ensure_collection()
    with pytest.raises(ValueError, match="expected 4"):
        QdrantStore(client, "mismatch", 4).ensure_collection()
