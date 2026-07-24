from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app import api
from app.config import Settings
from app.db import ProvenanceRepository, sha256_text
from app.models import ChunkCreate, DocumentCreate, DocumentVersionCreate, SourceType
from app.retrieval.service import RetrievalService, RetrievalUnavailable
from tests.integration.test_hybrid_index import FixtureDense, FixtureSparse


class BrokenStore:
    def query_dense(self, *args, **kwargs):
        raise ConnectionError("connection refused")


def _repository_with_current_version(tmp_path: Path):
    repository = ProvenanceRepository(tmp_path / "p.sqlite3")
    repository.initialize()
    document = repository.upsert_document(DocumentCreate(
        source_type=SourceType.NOTE, canonical_uri="memory://degraded",
        content_hash=sha256_text("evidence"), dataset="test",
    ))
    version = repository.create_version(DocumentVersionCreate(
        document_id=document.id, content_hash=sha256_text("evidence"),
        parser_name="test", parser_version="1", index_version="v1",
    ))
    repository.add_chunk(ChunkCreate(
        document_version_id=version.id, ordinal=0, text_hash=sha256_text("evidence"),
        text="evidence", token_count=1, chunk_type="text",
    ))
    repository.mark_version_current(version.id)
    return repository


def test_qdrant_dependency_failure_is_normalized(tmp_path):
    settings = Settings(rag_data_root=tmp_path, provenance_db_path=tmp_path / "p.sqlite3",
                        dense_embedding_dimensions=4, dense_candidates=10,
                        sparse_candidates=10, rerank_candidates=10)
    service = RetrievalService(settings=settings,
        repository=_repository_with_current_version(tmp_path), store=BrokenStore(),
        dense=FixtureDense(), sparse=FixtureSparse())
    with pytest.raises(RetrievalUnavailable, match="dense Qdrant search failed"):
        service.search("evidence")


def test_search_api_reports_retrieval_dependency_as_503(monkeypatch):
    class Offline:
        def search(self, *args, **kwargs):
            raise RetrievalUnavailable("dense Qdrant search failed: connection refused")
    monkeypatch.setattr(api, "_rag_retrieval_service", lambda: Offline())
    response = TestClient(api.app).post("/v1/search", json={"query": "evidence"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "retrieval_unavailable"
