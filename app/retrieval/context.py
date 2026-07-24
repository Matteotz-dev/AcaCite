"""Token-budgeted evidence packing."""

from __future__ import annotations

from collections.abc import Sequence
from .fusion import Candidate


def pack_context(candidates: Sequence[Candidate], *, max_chunks: int, token_budget: int) -> list[Candidate]:
    selected, used, per_document = [], 0, {}
    for candidate in candidates:
        tokens = max(1, len(candidate.payload.get("text", "")) // 4)
        document_id = candidate.payload.get("document_id")
        if used + tokens > token_budget or per_document.get(document_id, 0) >= 3:
            continue
        selected.append(candidate)
        used += tokens
        per_document[document_id] = per_document.get(document_id, 0) + 1
        if len(selected) >= max_chunks:
            break
    return selected
