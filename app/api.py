"""FastAPI composition for the local shared-memory service."""

from dataclasses import asdict
from functools import lru_cache
import subprocess
from typing import Any
from uuid import UUID

import shared_memory  # Configures durable storage before other Cognee imports.
import cognee
from cognee import SearchType
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="AcaCite",
    description="Model-independent retrieval and ingestion for local research corpora.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _api_token() -> str | None:
    from app.config import get_settings

    token = get_settings().acacite_api_token
    return token.strip() if token and token.strip() else None


def _authorized_api_token(headers, expected: str) -> bool:
    auth = headers.get("Authorization", "")
    bearer = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    supplied = bearer or headers.get("X-AcaCite-Token", "")
    return supplied == expected


@app.middleware("http")
async def optional_token_auth(request: Request, call_next):
    token = _api_token()
    if not token or request.url.path in {"/docs", "/openapi.json", "/redoc"}:
        return await call_next(request)
    if not _authorized_api_token(request.headers, token):
        return JSONResponse(
            status_code=401,
            content={"detail": {"code": "unauthorized", "message": "AcaCite API token required"}},
        )
    return await call_next(request)


class RememberRequest(BaseModel):
    content: str = Field(description="Text, file path, or URL to ingest")
    dataset: str = "main_dataset"


class RetrieveRequest(BaseModel):
    query: str
    datasets: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=50)


class AnswerRequest(BaseModel):
    query: str
    datasets: list[str] | None = None


class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    dataset: str | None = None
    project: str | None = None
    source_type: str | None = None
    language: str | None = None


class RAGAnswerRequest(RAGSearchRequest):
    model: str | None = Field(default=None, min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)


class RAGMemoryRequest(BaseModel):
    content: str = Field(min_length=1)
    source_uri: str = Field(min_length=1, description="Stable user-supplied provenance URI")
    dataset: str = Field(min_length=1)
    title: str | None = None
    project: str | None = None
    promote_to_cognee: bool = False
    promotion_kind: str = "decision"
    cognee_dataset: str | None = None


class RAGRelatedRequest(BaseModel):
    chunk_id: UUID
    limit: int = Field(default=5, ge=1, le=20)


class IngestionRequest(BaseModel):
    path: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    project: str | None = None
    delete_missing: bool = False


def serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    return value


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "memory_root": str(shared_memory.SHARED_ROOT),
        "extraction_model": "qwen3.6:27b",
    }


@app.post("/remember")
async def remember(request: RememberRequest) -> Any:
    result = await cognee.remember(request.content, dataset_name=request.dataset)
    return serialize(result)


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest) -> dict[str, Any]:
    """Return source chunks without binding completion to a particular LLM."""
    results = await cognee.search(
        request.query,
        query_type=SearchType.CHUNKS,
        datasets=request.datasets,
        top_k=request.limit,
    )
    return {"query": request.query, "results": serialize(results)}


@app.post("/answer")
async def answer(request: AnswerRequest) -> Any:
    """Optional Cognee-generated answer using the configured extraction LLM."""
    return serialize(
        await cognee.recall(
            query_text=request.query,
            datasets=request.datasets,
        )
    )


@lru_cache(maxsize=1)
def _rag_retrieval_service():
    """Construct runtime dependencies lazily; importing the API mutates no RAG store."""
    from app.config import get_settings
    from app.db import ProvenanceRepository
    from app.retrieval.embeddings import FastEmbedDense
    from app.retrieval.cognee_adapter import CogneeAdapter
    from app.retrieval.qdrant_store import QdrantStore
    from app.retrieval.reranker import CrossEncoderReranker
    from app.retrieval.service import RetrievalService
    from app.retrieval.sparse import FastEmbedSparse

    settings = get_settings()
    repository = ProvenanceRepository(settings.provenance_db_path)
    repository.initialize()
    return RetrievalService(
        settings=settings, repository=repository, store=QdrantStore.from_settings(settings),
        dense=FastEmbedDense(settings.dense_embedding_model, settings.dense_embedding_dimensions),
        sparse=FastEmbedSparse(settings.sparse_embedding_model),
        reranker=CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device),
        graph=CogneeAdapter(timeout_seconds=settings.cognee_search_timeout_seconds),
    )


@lru_cache(maxsize=1)
def _rag_answer_service():
    """Keep generation optional and independent from search/index construction."""
    from app.config import get_settings
    from app.generation.ollama import OllamaAdapter
    from app.generation.service import AnswerService

    settings = get_settings()
    retrieval = _rag_retrieval_service()
    return AnswerService(
        retrieval=retrieval, repository=retrieval.repository,
        generator=OllamaAdapter(
            settings.ollama_base_url, settings.ollama_generation_timeout_seconds
        ),
        approved_roots=settings.approved_ingestion_roots,
    )


@app.post("/v1/search")
def rag_search(request: RAGSearchRequest) -> dict[str, Any]:
    from app.retrieval.service import RetrievalUnavailable, SearchFilters

    try:
        response = _rag_retrieval_service().search(request.query, SearchFilters(
            dataset=request.dataset, project=request.project,
            source_type=request.source_type, language=request.language,
        ))
    except RetrievalUnavailable as exc:
        raise HTTPException(status_code=503, detail={
            "code": "retrieval_unavailable", "message": str(exc),
        }) from exc
    return {"query": request.query, "results": [asdict(item) for item in response.results],
            "trace": response.trace}


@app.post("/v1/answer")
def rag_answer(request: RAGAnswerRequest) -> dict[str, Any]:
    from app.config import get_settings
    from app.generation.ollama import OllamaUnavailable
    from app.retrieval.service import RetrievalUnavailable, SearchFilters

    settings = get_settings()
    try:
        return _rag_answer_service().answer(
            query=request.query,
            filters=SearchFilters(
                dataset=request.dataset, project=request.project,
                source_type=request.source_type, language=request.language,
            ),
            model=request.model or settings.default_generator_model,
            temperature=(settings.default_generation_temperature
                         if request.temperature is None else request.temperature),
        )
    except OllamaUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "ollama_unavailable", "message": str(exc)},
        ) from exc
    except RetrievalUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "retrieval_unavailable", "message": str(exc)},
        ) from exc


@app.get("/v1/sources/{chunk_id}")
def rag_source(chunk_id: UUID) -> dict[str, Any]:
    from app.config import get_settings
    from app.generation.citations import CitationResolver

    retrieval = _rag_retrieval_service()
    citation = CitationResolver(
        retrieval.repository, get_settings().approved_ingestion_roots
    ).resolve(chunk_id)
    if citation is None:
        raise HTTPException(status_code=404, detail="source chunk not found")
    anchor = retrieval.repository.get_chunk(chunk_id)
    context = []
    if anchor is not None:
        chunks = retrieval.repository.list_version_chunks(anchor.document_version_id)
        neighbors = [
            chunk for chunk in chunks
            if abs(chunk.ordinal - anchor.ordinal) <= 5
        ]
        context = [{
            "chunk_id": str(chunk.id),
            "ordinal": chunk.ordinal,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "heading": list(chunk.heading_path),
            "text": chunk.text,
        } for chunk in neighbors]
    result = citation.model_dump(mode="json")
    result["context"] = context
    open_path = result.get("metadata", {}).get("open_path")
    if open_path and str(open_path).lower().endswith(".pdf") and citation.page_start:
        first_page = max(1, citation.page_start)
        last_page = max(citation.page_end or first_page, first_page) + 2
        try:
            extracted = subprocess.run(
                [
                    "pdftotext", "-f", str(first_page), "-l", str(last_page),
                    "-layout", str(open_path), "-",
                ],
                check=True, capture_output=True, text=True, timeout=15,
            )
            result["layout_text"] = extracted.stdout[:30_000]
            result["layout_pages"] = [first_page, last_page]
        except (OSError, subprocess.SubprocessError):
            result["layout_text"] = None
    return result


@app.post("/v1/memory")
async def rag_memory(request: RAGMemoryRequest) -> dict[str, Any]:
    """Deliberately ingest a provenance-bearing note and optionally promote it."""
    from app.config import get_settings
    from app.ingestion.indexing import index_version
    from app.ingestion.memory import register_memory
    from app.ingestion.promotion import promote_memory

    settings = get_settings()
    retrieval = _rag_retrieval_service()
    try:
        version, chunks = register_memory(
            content=request.content, source_uri=request.source_uri,
            dataset=request.dataset, settings=settings, title=request.title,
            project=request.project,
        )
        indexed = index_version(
            version.id, repository=retrieval.repository, store=retrieval.store,
            dense=retrieval.dense, sparse=retrieval.sparse,
        )
        promotion = None
        if request.promote_to_cognee:
            promotion = await promote_memory(
                repository=retrieval.repository, version_id=indexed.id,
                kind=request.promotion_kind, text=request.content,
                cognee_dataset=request.cognee_dataset or request.dataset,
                chunk_id=chunks[0].id,
                timeout_seconds=settings.cognee_promotion_timeout_seconds,
            )
        return {
            "document_version_id": str(indexed.id),
            "chunk_ids": [str(chunk.id) for chunk in chunks],
            "status": indexed.status.value,
            "promotion": asdict(promotion) if promotion else None,
        }
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/related")
def rag_related(request: RAGRelatedRequest) -> dict[str, Any]:
    """Find indexed evidence related to an existing SQLite-authorized chunk."""
    retrieval = _rag_retrieval_service()
    chunk = retrieval.repository.get_chunk(request.chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="source chunk not found")
    response = retrieval.search(chunk.text)
    results = [item for item in response.results if item.chunk_id != request.chunk_id]
    return {
        "chunk_id": str(request.chunk_id),
        "results": [asdict(item) for item in results[:request.limit]],
        "trace": response.trace,
    }


def _ingestion_dependencies():
    from app.config import get_settings
    from app.ingestion.operator import IngestionDependencies

    retrieval = _rag_retrieval_service()
    return IngestionDependencies(
        settings=get_settings(), repository=retrieval.repository, store=retrieval.store,
        dense=retrieval.dense, sparse=retrieval.sparse,
    )


@app.post("/v1/ingestion/repo")
def rag_ingest_repo(request: IngestionRequest) -> dict[str, Any]:
    from pathlib import Path
    from app.ingestion.operator import ingest_repo

    try:
        return ingest_repo(
            Path(request.path), dataset=request.dataset, project=request.project,
            delete_missing=request.delete_missing, deps=_ingestion_dependencies(),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/ingestion/directory")
def rag_ingest_directory(request: IngestionRequest) -> dict[str, Any]:
    from pathlib import Path
    from app.ingestion.operator import ingest_directory

    try:
        return ingest_directory(
            Path(request.path), dataset=request.dataset, project=request.project,
            delete_missing=request.delete_missing, deps=_ingestion_dependencies(),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/ingestion/status")
def rag_ingestion_status(limit: int = 20) -> dict[str, Any]:
    from app.ingestion.operator import corpus_status

    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    return corpus_status(_rag_retrieval_service().repository, limit=limit)


@app.get("/v1/ingestion/jobs/{job_id}")
def rag_ingestion_job(job_id: str) -> dict[str, Any]:
    result = _rag_retrieval_service().repository.get_ingestion_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="ingestion job not found")
    return result


@app.post("/v1/ingestion/jobs/{job_id}/retry")
def rag_retry_ingestion(job_id: str) -> dict[str, Any]:
    from app.ingestion.operator import retry_job

    try:
        return retry_job(job_id, deps=_ingestion_dependencies())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/health")
def rag_health() -> dict[str, Any]:
    """Report component identity without making generation a health prerequisite."""
    from app.config import get_settings

    settings = get_settings()
    retrieval = _rag_retrieval_service()
    current_versions = retrieval.repository.current_version_ids()
    collection_exists = retrieval.store.client.collection_exists(retrieval.store.collection)
    point_count = 0
    if collection_exists:
        point_count = retrieval.store.client.count(retrieval.store.collection, exact=True).count
    return {
        "status": "ok",
        "sqlite": "ok",
        "qdrant": "ok",
        "qdrant_collection": retrieval.store.collection,
        "collection_points": point_count,
        "current_versions": len(current_versions),
        "index_version": settings.index_version,
        "ollama": {"required_for_search": False, "base_url": settings.ollama_base_url},
        "default_generator_model": settings.default_generator_model,
        "cognee_graph_enabled": settings.cognee_graph_enabled,
    }
