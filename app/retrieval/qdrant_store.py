"""Citation-grade dense+sparse Qdrant storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.models import ChunkRecord, DocumentRecord, DocumentVersionRecord
from .sparse import SparseEmbedding

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


@dataclass(frozen=True)
class VectorHit:
    chunk_id: UUID
    score: float
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dimensions: int):
        self.client = client
        self.collection = collection
        self.dimensions = dimensions

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantStore":
        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url)
        else:
            assert settings.qdrant_path is not None
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(settings.qdrant_path))
        return cls(client, settings.qdrant_collection, settings.dense_embedding_dimensions)

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={DENSE_VECTOR: models.VectorParams(
                    size=self.dimensions, distance=models.Distance.COSINE
                )},
                sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
            )
            return
        params = self.client.get_collection(self.collection).config.params
        dense = params.vectors.get(DENSE_VECTOR) if isinstance(params.vectors, dict) else None
        if dense is None or dense.size != self.dimensions:
            actual = dense.size if dense else None
            raise ValueError(
                f"collection {self.collection!r} dense dimension is {actual}; expected {self.dimensions}"
            )
        if not params.sparse_vectors or SPARSE_VECTOR not in params.sparse_vectors:
            raise ValueError(f"collection {self.collection!r} lacks sparse vector {SPARSE_VECTOR!r}")

    def upsert_version(
        self,
        document: DocumentRecord,
        version: DocumentVersionRecord,
        chunks: Sequence[ChunkRecord],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseEmbedding],
    ) -> None:
        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError("chunk and embedding counts differ")
        points = []
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            if len(dense) != self.dimensions:
                raise ValueError(f"dense vector dimension {len(dense)} != {self.dimensions}")
            payload = {
                "chunk_id": str(chunk.id), "document_id": str(document.id),
                "document_version_id": str(version.id), "index_version": version.index_version,
                "dataset": document.dataset, "project": document.project,
                "source_type": document.source_type.value, "canonical_uri": document.canonical_uri,
                "title": document.title, "language": chunk.language or document.language,
                "git_repository": version.git_repository, "git_commit": version.git_commit,
                "ordinal": chunk.ordinal, "text": chunk.text, "text_hash": chunk.text_hash,
                "chunk_type": chunk.chunk_type, "heading_path": list(chunk.heading_path),
                "page_start": chunk.page_start, "page_end": chunk.page_end,
                "line_start": chunk.line_start, "line_end": chunk.line_end,
                "symbol": chunk.symbol,
            }
            points.append(models.PointStruct(
                id=str(chunk.id), payload=payload,
                vector={
                    DENSE_VECTOR: list(dense),
                    SPARSE_VECTOR: models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
            ))
        if points:
            self.client.upsert(self.collection, points=points, wait=True)

    def count_version(self, version_id: UUID) -> int:
        return self.client.count(
            self.collection, count_filter=self._version_filter(version_id), exact=True
        ).count

    def delete_version(self, version_id: UUID) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=self._version_filter(version_id)),
            wait=True,
        )

    def query_dense(self, vector: Sequence[float], *, limit: int, query_filter=None) -> list[VectorHit]:
        points = self.client.query_points(
            self.collection, query=list(vector), using=DENSE_VECTOR,
            query_filter=query_filter, limit=limit, with_payload=True,
        ).points
        return self._hits(points)

    def query_sparse(self, vector: SparseEmbedding, *, limit: int, query_filter=None) -> list[VectorHit]:
        points = self.client.query_points(
            self.collection,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=SPARSE_VECTOR, query_filter=query_filter, limit=limit, with_payload=True,
        ).points
        return self._hits(points)

    @staticmethod
    def metadata_filter(**values: Any) -> models.Filter | None:
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in values.items() if value is not None
        ]
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _version_filter(version_id: UUID) -> models.Filter:
        return QdrantStore.metadata_filter(document_version_id=str(version_id))  # type: ignore[return-value]

    @staticmethod
    def _hits(points) -> list[VectorHit]:
        return [VectorHit(UUID(point.payload["chunk_id"]), point.score, point.payload) for point in points]
