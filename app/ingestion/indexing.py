"""Transaction-like promotion of parsed SQLite versions into Qdrant."""

from __future__ import annotations

from uuid import UUID

from app.db import ProvenanceRepository
from app.models import VersionStatus
from app.retrieval.embeddings import DenseEmbedder
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.sparse import SparseEmbedder


def index_version(
    version_id: UUID, *, repository: ProvenanceRepository, store: QdrantStore,
    dense: DenseEmbedder, sparse: SparseEmbedder,
):
    version = repository.get_version(version_id)
    if version is None:
        raise KeyError(f"unknown document version: {version_id}")
    document = repository.get_document(version.document_id)
    assert document is not None
    chunks = repository.list_version_chunks(version_id)
    if version.status is VersionStatus.INDEXED and store.count_version(version_id) == len(chunks):
        return version
    try:
        store.ensure_collection()
        texts = [_embedding_text(document.title, chunk) for chunk in chunks]
        store.upsert_version(document, version, chunks, dense.embed_documents(texts), sparse.embed_documents(texts))
        if store.count_version(version_id) != len(chunks):
            raise RuntimeError("Qdrant point-count verification failed")
        with repository.transaction() as connection:
            repository.set_chunk_point_ids(((chunk.id, str(chunk.id)) for chunk in chunks), connection)
            return repository.mark_version_current(version_id, connection)
    except Exception as exc:
        try:
            store.delete_version(version_id)
        finally:
            repository.set_version_status(version_id, VersionStatus.FAILED, str(exc))
        raise


def _embedding_text(title: str | None, chunk) -> str:
    context = [value for value in (title, " > ".join(chunk.heading_path), chunk.symbol) if value]
    return f"{' | '.join(context)}\n{chunk.text}" if context else chunk.text
