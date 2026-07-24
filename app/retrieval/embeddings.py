"""CPU embedding adapters, independent of the answer-generating model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol


class DenseEmbedder(Protocol):
    dimensions: int
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedDense:
    def __init__(self, model_name: str, dimensions: int, *, threads: int | None = None):
        self.model_name = model_name
        self.dimensions = dimensions
        self._threads = threads
        self._model = None

    def _embed(self, texts: Iterable[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name, threads=self._threads)
        values = [vector.tolist() for vector in self._model.embed(list(texts))]
        if any(len(vector) != self.dimensions for vector in values):
            raise ValueError(
                f"dense model {self.model_name!r} did not produce {self.dimensions} dimensions"
            )
        return values

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
