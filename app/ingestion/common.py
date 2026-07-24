"""Parser-neutral document representation used before persistence/indexing."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


def estimate_tokens(text: str) -> int:
    """Stable dependency-free estimate used until the embedding tokenizer exists."""
    return max(1, (len(text) + 3) // 4)


class NormalizedChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    chunk_type: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    symbol: str | None = None
    language: str | None = None

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.text)

    @model_validator(mode="after")
    def validate_spans(self) -> "NormalizedChunk":
        for kind, start, end in (
            ("page", self.page_start, self.page_end),
            ("line", self.line_start, self.line_end),
        ):
            if end is not None and start is None:
                raise ValueError(f"{kind}_end requires {kind}_start")
            if start is not None and end is not None and end < start:
                raise ValueError(f"{kind}_end cannot precede {kind}_start")
        return self


class NormalizedDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: Path
    title: str | None = None
    mime_type: str | None = None
    language: str | None = None
    parser_name: str
    parser_version: str
    chunks: tuple[NormalizedChunk, ...]
    metadata: dict[str, str | bool | int | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_chunks(self) -> "NormalizedDocument":
        if not self.chunks:
            raise ValueError("a parsed document must contain at least one chunk")
        return self
