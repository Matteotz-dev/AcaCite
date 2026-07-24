"""Phase-4 hybrid search orchestration with traceable component rankings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from qdrant_client import models

from app.config import Settings
from app.db import ProvenanceRepository
from .context import pack_context
from .cognee_adapter import CogneeSearchResult, GraphRetriever
from .embeddings import DenseEmbedder
from .fusion import Candidate, merge_graph_evidence, reciprocal_rank_fusion
from .qdrant_store import QdrantStore
from .reranker import BypassReranker, Reranker
from .sparse import SparseEmbedder


@dataclass(frozen=True)
class SearchFilters:
    dataset: str | None = None
    project: str | None = None
    source_type: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class SearchResponse:
    results: list[Candidate]
    trace: dict[str, Any]


class RetrievalUnavailable(RuntimeError):
    """A required raw-evidence dependency could not serve the search."""


def looks_lexical(query: str) -> bool:
    return bool(re.search(r"[/_]|\b[A-Za-z]+[A-Z][A-Za-z]+\b|['\"`].+?['\"`]", query))


class RetrievalService:
    def __init__(self, *, settings: Settings, repository: ProvenanceRepository,
                 store: QdrantStore, dense: DenseEmbedder, sparse: SparseEmbedder,
                 reranker: Reranker | None = None, graph: GraphRetriever | None = None):
        self.settings, self.repository, self.store = settings, repository, store
        self.dense, self.sparse = dense, sparse
        self.reranker = reranker or BypassReranker()
        self.graph = graph

    def search(self, query: str, filters: SearchFilters | None = None) -> SearchResponse:
        started = perf_counter()
        filters = filters or SearchFilters()
        current = [str(value) for value in self.repository.current_version_ids()]
        if not current:
            return SearchResponse([], {"total_ms": 0.0, "reason": "empty_index"})
        must = [models.FieldCondition(
            key="document_version_id", match=models.MatchAny(any=current)
        )]
        for key in ("dataset", "project", "source_type", "language"):
            value = getattr(filters, key)
            if value is not None:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        query_filter = models.Filter(must=must)
        dense_started = perf_counter()
        try:
            dense_hits = self.store.query_dense(
                self.dense.embed_query(query), limit=self.settings.dense_candidates,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise RetrievalUnavailable(f"dense Qdrant search failed: {exc}") from exc
        dense_ms = (perf_counter() - dense_started) * 1000
        sparse_started = perf_counter()
        try:
            sparse_hits = self.store.query_sparse(
                self.sparse.embed_query(query), limit=self.settings.sparse_candidates,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise RetrievalUnavailable(f"sparse Qdrant search failed: {exc}") from exc
        sparse_ms = (perf_counter() - sparse_started) * 1000
        fused = reciprocal_rank_fusion({"dense": dense_hits, "sparse": sparse_hits})
        graph_started = perf_counter()
        graph_result = CogneeSearchResult(status="disabled")
        unsupported = []
        if self.graph is not None and self.settings.cognee_graph_enabled:
            datasets = [filters.dataset] if filters.dataset else None
            graph_result = self.graph.search(
                query, datasets=datasets, limit=self.settings.graph_candidates
            )
            valid = []
            current_set = set(current)
            for item in graph_result.evidence:
                chunk = self.repository.get_chunk(item.chunk_id) if item.chunk_id else None
                if chunk is not None and str(chunk.document_version_id) in current_set:
                    version = self.repository.get_version(chunk.document_version_id)
                    if version and item.document_id == version.document_id and (
                        item.document_version_id is None or
                        item.document_version_id == chunk.document_version_id
                    ):
                        valid.append(item)
                    else:
                        unsupported.append(item)
                elif item.document_version_id and str(item.document_version_id) in current_set:
                    document = self.repository.get_document(item.document_id) if item.document_id else None
                    if document and document.current_version_id == item.document_version_id:
                        valid.append(item)
                    else:
                        unsupported.append(item)
                else:
                    unsupported.append(item)
            fused, unmerged = merge_graph_evidence(fused, valid)
            unsupported.extend(unmerged)
        graph_ms = (perf_counter() - graph_started) * 1000
        rerank_pool = fused[:self.settings.rerank_candidates]
        rerank_started = perf_counter()
        ranked = self.reranker.rerank(query, rerank_pool)
        rerank_ms = (perf_counter() - rerank_started) * 1000
        results = pack_context(
            ranked, max_chunks=self.settings.final_context_chunks,
            token_budget=self.settings.answer_context_tokens,
        )
        return SearchResponse(results, {
            "dense_ms": dense_ms, "sparse_ms": sparse_ms, "graph_ms": graph_ms,
            "rerank_ms": rerank_ms,
            "total_ms": (perf_counter() - started) * 1000,
            "dense_count": len(dense_hits), "sparse_count": len(sparse_hits),
            "fused_count": len(fused), "lexical_query": looks_lexical(query),
            "reranker": self.reranker.model_name,
            "cognee_status": graph_result.status, "cognee_error": graph_result.error,
            "cognee_count": len(graph_result.evidence),
            "memory_without_source": [
                {"text": item.text, "cognee_ref": item.cognee_ref,
                 "source_grade": False} for item in unsupported
            ],
        })
