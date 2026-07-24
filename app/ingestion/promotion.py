"""Controlled, idempotent promotion of curated source knowledge into Cognee."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable
from uuid import UUID

from app.db import ProvenanceRepository
from app.db.provenance import sha256_text

ALLOWED_KINDS = frozenset({"summary", "claim", "decision", "entity_set", "architecture"})


@dataclass(frozen=True)
class PromotionResult:
    promotion_id: str
    payload_hash: str
    cognee_ref: str | None
    already_promoted: bool


class CogneePromoter:
    def __init__(self, add: Callable[..., Awaitable[object]], cognify: Callable[..., Awaitable[object]]):
        self._add, self._cognify = add, cognify

    @classmethod
    def runtime(cls) -> "CogneePromoter":
        import shared_memory  # noqa: F401
        import cognee
        return cls(cognee.add, cognee.cognify)

    async def promote(self, payload: str, dataset: str) -> str | None:
        added = await self._add(payload, dataset_name=dataset)
        await self._cognify([dataset])
        return _extract_ref(added)


async def promote_memory(
    *, repository: ProvenanceRepository, version_id: UUID, kind: str,
    text: str, cognee_dataset: str, chunk_id: UUID | None = None,
    promoter: CogneePromoter | None = None, timeout_seconds: float = 120.0,
) -> PromotionResult:
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unsupported promotion kind: {kind}")
    clean = text.strip()
    if not clean:
        raise ValueError("promotion text cannot be empty")
    version = repository.get_version(version_id)
    if version is None:
        raise KeyError(f"unknown document version: {version_id}")
    marker = f"[RAG_PROVENANCE document_id={version.document_id} version_id={version.id}"
    if chunk_id is not None:
        chunk = repository.get_chunk(chunk_id)
        if chunk is None or chunk.document_version_id != version_id:
            raise ValueError("chunk does not belong to promoted document version")
        marker += f" chunk_id={chunk_id}"
    payload = f"{marker}]\nkind: {kind}\n{clean}"
    payload_hash = sha256_text(payload)
    existing = repository.get_memory_promotion(version_id, kind, payload_hash)
    if existing:
        return PromotionResult(existing["id"], payload_hash, existing["cognee_ref"], True)
    gateway = promoter or CogneePromoter.runtime()
    cognee_ref = await asyncio.wait_for(gateway.promote(payload, cognee_dataset), timeout_seconds)
    promotion_id = repository.record_memory_promotion(
        document_id=version.document_id, version_id=version.id, kind=kind,
        cognee_dataset=cognee_dataset, cognee_ref=cognee_ref, payload_hash=payload_hash,
    )
    if chunk_id is not None and cognee_ref:
        repository.set_chunk_cognee_ref(chunk_id, cognee_ref)
    return PromotionResult(promotion_id, payload_hash, cognee_ref, False)


def _extract_ref(value: object) -> str | None:
    if isinstance(value, dict):
        for key in ("id", "uuid", "data_id", "dataset_id"):
            if value.get(key) is not None:
                return str(value[key])
    if isinstance(value, (list, tuple)) and value:
        return _extract_ref(value[0])
    return None
