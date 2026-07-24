"""Cross-encoder reranking with CUDA OOM recovery and CPU fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .fusion import Candidate


class Reranker(Protocol):
    model_name: str
    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]: ...


class BypassReranker:
    model_name = "bypass"
    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        return list(candidates)


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str = "cuda", batch_size: int = 16):
        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        self._model = None

    def _load(self, device: str):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=device)

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        pairs = [(query, item.payload["text"]) for item in candidates]
        device, batch = self.device, self.batch_size
        while True:
            try:
                if self._model is None:
                    self._load(device)
                scores = self._model.predict(pairs, batch_size=batch)
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if batch > 1:
                    batch = max(1, batch // 2)
                    continue
                if device != "cpu":
                    device, self._model = "cpu", None
                    continue
                raise
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.reranker_score = float(score)
        return sorted(candidates, key=lambda item: (-float(item.reranker_score), -item.fused_score))
