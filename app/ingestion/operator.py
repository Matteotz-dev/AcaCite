"""Observable bulk ingestion and incremental synchronization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from app.config import Settings
from app.db import ProvenanceRepository, sha256_bytes, utc_now, validate_approved_path
from app.retrieval.embeddings import DenseEmbedder
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.sparse import SparseEmbedder

from .indexing import index_version
from .registry import parser_for
from .repository import OPENFOAM_NAMES, SUPPORTED_SUFFIXES, discover_repository
from .service import ingest_file


@dataclass(frozen=True)
class IngestionDependencies:
    settings: Settings
    repository: ProvenanceRepository
    store: QdrantStore
    dense: DenseEmbedder
    sparse: SparseEmbedder


def discover_directory(path: Path, approved_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    root = validate_approved_path(path, approved_roots)
    if not root.is_dir():
        raise ValueError(f"directory ingestion requires a directory: {root}")
    files = []
    for candidate in root.rglob("*"):
        if any(part.startswith(".") for part in candidate.relative_to(root).parts):
            continue
        if candidate.is_file() and (
            candidate.suffix.lower() in SUPPORTED_SUFFIXES or candidate.name in OPENFOAM_NAMES
        ):
            files.append(validate_approved_path(candidate, approved_roots))
    return tuple(sorted(files))


def ingest_repo(
    path: Path, *, dataset: str, project: str | None, deps: IngestionDependencies,
    delete_missing: bool = False,
) -> dict:
    snapshot = discover_repository(path, deps.settings.approved_ingestion_roots)
    options = {
        "mode": "repo", "project": project, "delete_missing": delete_missing,
        "git_repository": snapshot.root.as_uri(), "git_commit": snapshot.commit,
        "branch": snapshot.branch, "dirty": snapshot.dirty,
    }
    return _run(
        root=snapshot.root, paths=snapshot.files, dataset=dataset, project=project,
        git_repository=snapshot.root.as_uri(), git_commit=snapshot.commit,
        delete_missing=delete_missing, options=options, deps=deps,
    )


def ingest_directory(
    path: Path, *, dataset: str, project: str | None, deps: IngestionDependencies,
    delete_missing: bool = False,
) -> dict:
    root = validate_approved_path(path, deps.settings.approved_ingestion_roots)
    options = {"mode": "directory", "project": project, "delete_missing": delete_missing}
    return _run(
        root=root, paths=discover_directory(root, deps.settings.approved_ingestion_roots),
        dataset=dataset, project=project, git_repository=None, git_commit=None,
        delete_missing=delete_missing, options=options, deps=deps,
    )


def retry_job(job_id: str, *, deps: IngestionDependencies) -> dict:
    prior = deps.repository.get_ingestion_job(job_id)
    if prior is None:
        raise KeyError(f"unknown ingestion job: {job_id}")
    failed = [item for item in prior["files"] if item["status"] == "failed"]
    if not failed:
        raise ValueError(f"ingestion job {job_id} has no failed files")
    options = prior["options"]
    paths = tuple(_file_uri_path(item["canonical_uri"]) for item in failed)
    root = _file_uri_path(prior["requested_uri"])
    return _run(
        root=root, paths=paths, dataset=prior["dataset"], project=options.get("project"),
        git_repository=options.get("git_repository"), git_commit=options.get("git_commit"),
        delete_missing=False, options={**options, "retry_of": job_id, "delete_missing": False},
        deps=deps,
    )


def _run(
    *, root: Path, paths: Iterable[Path], dataset: str, project: str | None,
    git_repository: str | None, git_commit: str | None, delete_missing: bool,
    options: dict, deps: IngestionDependencies,
) -> dict:
    paths = tuple(paths)
    repository = deps.repository
    job_id = repository.create_ingestion_job(
        requested_uri=root.as_uri(), dataset=dataset, options=options
    )
    repository.update_ingestion_job(job_id, status="running", started_at=utc_now().isoformat())
    changed = chunks_indexed = failures = 0
    seen = {path.as_uri() for path in paths}
    for path in paths:
        uri = path.as_uri()
        try:
            parser_for(path)  # Fail early with a useful per-file record.
            content_hash = sha256_bytes(path.read_bytes())
            existing = repository.find_document(uri, dataset)
            if existing and existing.current_version_id and existing.content_hash == content_hash:
                repository.record_ingestion_file(
                    job_id=job_id, canonical_uri=uri, action="skip", status="complete"
                )
                continue
            version, chunks = ingest_file(
                path, dataset=dataset, settings=deps.settings, project=project,
                git_repository=git_repository, git_commit=git_commit,
            )
            indexed = index_version(
                version.id, repository=repository, store=deps.store,
                dense=deps.dense, sparse=deps.sparse,
            )
            changed += 1
            chunks_indexed += len(chunks)
            repository.record_ingestion_file(
                job_id=job_id, canonical_uri=uri, action="index", status="complete",
                version_id=indexed.id, chunks_indexed=len(chunks),
            )
        except Exception as exc:
            failures += 1
            repository.record_ingestion_file(
                job_id=job_id, canonical_uri=uri, action="index", status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    if delete_missing:
        prefix = root.as_uri().rstrip("/") + "/"
        for document in repository.list_documents(
            dataset=dataset, project=project, uri_prefix=prefix
        ):
            if document.canonical_uri in seen:
                continue
            try:
                if document.current_version_id:
                    deps.store.delete_version(document.current_version_id)
                repository.mark_document_deleted(document.id)
                changed += 1
                repository.record_ingestion_file(
                    job_id=job_id, canonical_uri=document.canonical_uri,
                    action="delete", status="complete",
                )
            except Exception as exc:
                failures += 1
                repository.record_ingestion_file(
                    job_id=job_id, canonical_uri=document.canonical_uri,
                    action="delete", status="failed", error=f"{type(exc).__name__}: {exc}",
                )

    status = "partial" if failures and (changed or len(paths) > failures) else (
        "failed" if failures else "complete"
    )
    repository.update_ingestion_job(
        job_id, status=status, documents_seen=len(paths), documents_changed=changed,
        chunks_indexed=chunks_indexed,
        error=f"{failures} file operation(s) failed" if failures else None,
        finished_at=utc_now().isoformat(),
    )
    return repository.get_ingestion_job(job_id)  # type: ignore[return-value]


def corpus_status(repository: ProvenanceRepository, *, limit: int = 20) -> dict:
    documents = repository.list_documents()
    current = [document for document in documents if document.current_version_id]
    return {
        "documents": len(documents),
        "current_versions": len(current),
        "datasets": sorted({document.dataset for document in documents}),
        "projects": sorted({document.project for document in documents if document.project}),
        "recent_jobs": repository.list_ingestion_jobs(limit),
    }


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"expected a file URI, got {uri!r}")
    return Path(unquote(parsed.path))
