"""CPU sparse/BM25 embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


class SparseEmbedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[SparseEmbedding]: ...
    def embed_query(self, text: str) -> SparseEmbedding: ...


class FastEmbedSparse:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _embed(self, texts: Sequence[str]) -> list[SparseEmbedding]:
        if self._model is None:
            from fastembed import SparseTextEmbedding
            self._model = SparseTextEmbedding(model_name=self.model_name)
        return [
            SparseEmbedding(vector.indices.tolist(), vector.values.tolist())
            for vector in self._model.embed(list(texts))
        ]

    def embed_documents(self, texts: Sequence[str]) -> list[SparseEmbedding]:
        return self._embed(texts)

    def embed_query(self, text: str) -> SparseEmbedding:
        return self._embed([text])[0]
