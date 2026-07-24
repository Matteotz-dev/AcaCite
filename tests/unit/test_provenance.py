from pathlib import Path

import pytest

from app.db import (
    ProvenanceRepository,
    sha256_text,
    stable_chunk_id,
    stable_document_id,
    validate_approved_path,
)
from app.db.migrations import SCHEMA_VERSION
from app.models import ChunkCreate, DocumentCreate, DocumentVersionCreate, SourceType, VersionStatus


def document_value(uri: str, content: str = "first") -> DocumentCreate:
    return DocumentCreate(
        source_type=SourceType.NOTE,
        canonical_uri=uri,
        content_hash=sha256_text(content),
        dataset="test",
        title="Test note",
    )


def version_value(document_id, content: str) -> DocumentVersionCreate:
    return DocumentVersionCreate(
        document_id=document_id,
        content_hash=sha256_text(content),
        parser_name="unit",
        parser_version="1",
        index_version="v1",
    )


@pytest.fixture
def repository(tmp_path: Path) -> ProvenanceRepository:
    repo = ProvenanceRepository(tmp_path / "registry.sqlite3")
    repo.initialize()
    return repo


def test_stable_hashes_and_ids():
    assert sha256_text("same") == sha256_text("same")
    first = stable_document_id("dataset", "file:///one")
    assert first == stable_document_id("dataset", "file:///one")
    assert first != stable_document_id("other", "file:///one")
    assert stable_chunk_id(first, 0, sha256_text("x")) == stable_chunk_id(
        first, 0, sha256_text("x")
    )


def test_idempotent_document_version_and_chunk(repository: ProvenanceRepository):
    doc = repository.upsert_document(document_value("note:test"))
    assert repository.upsert_document(document_value("note:test")).id == doc.id
    version = repository.create_version(version_value(doc.id, "first"))
    assert repository.create_version(version_value(doc.id, "first")).id == version.id
    chunk_value = ChunkCreate(
        document_version_id=version.id,
        ordinal=0,
        text_hash=sha256_text("hello"),
        text="hello",
        token_count=1,
        chunk_type="paragraph",
    )
    chunk = repository.add_chunk(chunk_value)
    assert repository.add_chunk(chunk_value).id == chunk.id


def test_changed_chunk_at_existing_ordinal_is_rejected(repository: ProvenanceRepository):
    doc = repository.upsert_document(document_value("note:test"))
    version = repository.create_version(version_value(doc.id, "first"))
    base = dict(
        document_version_id=version.id, ordinal=0, token_count=1, chunk_type="paragraph"
    )
    repository.add_chunk(ChunkCreate(**base, text="one", text_hash=sha256_text("one")))
    with pytest.raises(ValueError, match="different content"):
        repository.add_chunk(ChunkCreate(**base, text="two", text_hash=sha256_text("two")))


def test_current_version_transition_preserves_history(repository: ProvenanceRepository):
    doc = repository.upsert_document(document_value("note:test"))
    first = repository.create_version(version_value(doc.id, "first"))
    repository.mark_version_current(first.id)
    second = repository.create_version(version_value(doc.id, "second"))
    repository.mark_version_current(second.id)

    assert repository.get_document(doc.id).current_version_id == second.id
    assert repository.get_version(first.id).status is VersionStatus.SUPERSEDED
    assert repository.get_version(second.id).status is VersionStatus.INDEXED


def test_changed_upsert_does_not_claim_unindexed_hash_is_current(
    repository: ProvenanceRepository,
):
    original = repository.upsert_document(document_value("note:test", "first"))
    first = repository.create_version(version_value(original.id, "first"))
    repository.mark_version_current(first.id)

    changed = repository.upsert_document(document_value("note:test", "failed-change"))

    assert changed.current_version_id == first.id
    assert changed.content_hash == sha256_text("first")


def test_transaction_rolls_back(repository: ProvenanceRepository):
    value = document_value("note:rollback")
    with pytest.raises(RuntimeError):
        with repository.transaction() as connection:
            document = repository.upsert_document(value, connection)
            repository.create_version(version_value(document.id, "first"), connection)
            raise RuntimeError("stop")
    assert repository.get_document(stable_document_id("test", "note:rollback")) is None


def test_approved_root_blocks_escape_and_symlinks(tmp_path: Path):
    approved = tmp_path / "approved"
    approved.mkdir()
    inside = approved / "paper.txt"
    inside.write_text("paper", encoding="utf-8")
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    link = approved / "escape"
    link.symlink_to(outside)

    assert validate_approved_path(inside, (approved,)) == inside.resolve()
    with pytest.raises(ValueError, match="outside approved"):
        validate_approved_path(outside, (approved,))
    with pytest.raises(ValueError, match="outside approved"):
        validate_approved_path(link, (approved,))


def test_initialize_is_idempotent_and_records_schema_version(
    repository: ProvenanceRepository,
):
    repository.initialize()
    with repository.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone()[0] == SCHEMA_VERSION
