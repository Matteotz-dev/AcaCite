"""Timeout-bounded Cognee access and normalized graph evidence."""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID

PROVENANCE_RE = re.compile(
    r"\[RAG_PROVENANCE\s+document_id=(?P<document>[0-9a-f-]{36})"
    r"(?:\s+version_id=(?P<version>[0-9a-f-]{36}))?"
    r"(?:\s+chunk_id=(?P<chunk>[0-9a-f-]{36}))?\]", re.IGNORECASE,
)


@dataclass(frozen=True)
class GraphEvidence:
    text: str
    score: float = 0.0
    document_id: UUID | None = None
    document_version_id: UUID | None = None
    chunk_id: UUID | None = None
    cognee_ref: str | None = None
    source_grade: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CogneeSearchResult:
    evidence: tuple[GraphEvidence, ...] = ()
    status: str = "ok"
    error: str | None = None


class GraphRetriever(Protocol):
    def search(self, query: str, *, datasets: list[str] | None, limit: int) -> CogneeSearchResult: ...


def _run_with_timeout(factory: Callable[[], Awaitable[Any]], timeout: float) -> Any:
    output: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def runner() -> None:
        try:
            output.put((True, asyncio.run(factory())))
        except BaseException as exc:
            output.put((False, exc))

    thread = threading.Thread(target=runner, daemon=True, name="cognee-timeout-call")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"Cognee call exceeded {timeout:g}s")
    ok, value = output.get_nowait()
    if not ok:
        raise value
    return value


class CogneeAdapter:
    """Lazy gateway; importing this module does not initialize Cognee."""

    def __init__(self, *, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, datasets: list[str] | None, limit: int) -> CogneeSearchResult:
        if limit <= 0:
            return CogneeSearchResult(status="disabled")
        try:
            raw = _run_with_timeout(
                lambda: self._search_async(query, datasets=datasets, limit=limit),
                self.timeout_seconds,
            )
            return CogneeSearchResult(tuple(normalize_evidence(raw)[:limit]))
        except Exception as exc:
            return CogneeSearchResult(status="degraded", error=f"{type(exc).__name__}: {exc}")

    async def _search_async(self, query: str, *, datasets: list[str] | None, limit: int) -> Any:
        import shared_memory  # noqa: F401 - configure durable store only on a call
        import cognee
        from cognee.modules.search.types import SearchType
        return await cognee.search(
            query, query_type=SearchType.CHUNKS, datasets=datasets, top_k=limit
        )


def normalize_evidence(raw: Any) -> list[GraphEvidence]:
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    normalized: list[GraphEvidence] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        data = value if isinstance(value, dict) else {}
        text = _first_text(data) if data else str(value)
        marker = PROVENANCE_RE.search(text)
        document_id = _uuid(data.get("document_id"))
        version_id = _uuid(data.get("document_version_id") or data.get("version_id"))
        chunk_id = _uuid(data.get("chunk_id"))
        if marker:
            document_id = document_id or _uuid(marker.group("document"))
            version_id = version_id or _uuid(marker.group("version"))
            chunk_id = chunk_id or _uuid(marker.group("chunk"))
        normalized.append(GraphEvidence(
            text=text, score=float(data.get("score", 1.0 / (index + 1))),
            document_id=document_id, document_version_id=version_id, chunk_id=chunk_id,
            cognee_ref=_first(data, "id", "cognee_ref", "uuid"),
            source_grade=document_id is not None and (chunk_id is not None or version_id is not None),
            metadata={key: val for key, val in data.items() if key not in {"text", "content"}},
        ))
    return normalized


def _first_text(data: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk", "result"):
        if isinstance(data.get(key), str):
            return data[key]
    return str(data)


def _first(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if data.get(key) is not None:
            return str(data[key])
    return None


def _uuid(value: Any) -> UUID | None:
    try:
        return UUID(str(value)) if value else None
    except (ValueError, TypeError, AttributeError):
        return None
