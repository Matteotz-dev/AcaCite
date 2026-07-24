from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from qdrant_client import QdrantClient

from app.config import Settings
from app.db import ProvenanceRepository
from app.ingestion import operator
from app.ingestion.operator import IngestionDependencies, ingest_directory, ingest_repo, retry_job
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.sparse import SparseEmbedding


class Dense:
    model_name = "fixture"
    dimensions = 4
    def embed_documents(self, texts): return [[1.0, 0.0, 0.0, 1.0] for _ in texts]
    def embed_query(self, text): return [1.0, 0.0, 0.0, 1.0]


class Sparse:
    model_name = "fixture"
    def _one(self, text):
        tokens = sorted(set(re.findall(r"\w+", text.lower())))
        return SparseEmbedding(
            [int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) for value in tokens],
            [1.0] * len(tokens),
        )
    def embed_documents(self, texts): return [self._one(text) for text in texts]
    def embed_query(self, text): return self._one(text)


def dependencies(tmp_path: Path) -> IngestionDependencies:
    settings = Settings(
        rag_data_root=tmp_path / "data", provenance_db_path=tmp_path / "data/db.sqlite3",
        qdrant_path=tmp_path / "data/qdrant", qdrant_collection="bulk_fixture",
        dense_embedding_dimensions=4, approved_ingestion_roots=(tmp_path,),
    )
    repository = ProvenanceRepository(settings.provenance_db_path)
    repository.initialize()
    return IngestionDependencies(
        settings, repository, QdrantStore(QdrantClient(":memory:"), "bulk_fixture", 4),
        Dense(), Sparse(),
    )


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_repo_ingest_is_idempotent_then_syncs_change_and_delete(tmp_path: Path):
    deps = dependencies(tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    first = root / "first.py"
    second = root / "second.md"
    first.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    second.write_text("# Second\n\nStable evidence.\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "initial")

    initial = ingest_repo(root, dataset="code", project="demo", deps=deps)
    assert initial["status"] == "complete"
    assert initial["documents_changed"] == 2
    unchanged = ingest_repo(root, dataset="code", project="demo", deps=deps)
    assert unchanged["documents_changed"] == 0
    assert {item["action"] for item in unchanged["files"]} == {"skip"}

    first.write_text("def alpha():\n    return 2\n", encoding="utf-8")
    second.unlink()
    synced = ingest_repo(
        root, dataset="code", project="demo", deps=deps, delete_missing=True
    )
    assert synced["status"] == "complete"
    assert synced["documents_changed"] == 2
    assert {item["action"] for item in synced["files"]} == {"index", "delete"}
    active = deps.repository.list_documents(dataset="code", project="demo")
    assert len(active) == 1
    assert active[0].canonical_uri == first.as_uri()


def test_directory_failure_is_recorded_and_retryable(tmp_path: Path, monkeypatch):
    deps = dependencies(tmp_path)
    root = tmp_path / "papers"
    root.mkdir()
    source = root / "paper.md"
    source.write_text("# Evidence\n\nA reviewed claim.\n", encoding="utf-8")
    real_ingest = operator.ingest_file

    def fail(*args, **kwargs):
        raise RuntimeError("fixture parser failure")

    monkeypatch.setattr(operator, "ingest_file", fail)
    failed = ingest_directory(root, dataset="papers", project=None, deps=deps)
    assert failed["status"] == "failed"
    assert failed["files"][0]["status"] == "failed"
    assert "fixture parser failure" in failed["files"][0]["error"]

    monkeypatch.setattr(operator, "ingest_file", real_ingest)
    retried = retry_job(failed["id"], deps=deps)
    assert retried["status"] == "complete"
    assert retried["options"]["retry_of"] == failed["id"]
    assert retried["documents_changed"] == 1
