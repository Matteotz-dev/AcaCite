"""Strict schemas for reproducible gold sets and recorded retrieval runs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Category = Literal[
    "exact_code", "paper_fact", "cross_source", "graph_multihop",
    "insufficient_evidence",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldCase(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]+$")
    query: str = Field(min_length=3)
    category: Category
    filters: dict[str, str] = Field(default_factory=dict)
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_version_ids: list[str] = Field(default_factory=list)
    graded_relevance: dict[str, int] = Field(default_factory=dict)
    required_facts: list[str] = Field(default_factory=list)
    must_abstain: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def validate_answerability(self) -> "GoldCase":
        if self.must_abstain and (self.relevant_chunk_ids or self.required_facts):
            raise ValueError("abstention cases cannot declare evidence or required facts")
        if not self.must_abstain and not self.relevant_chunk_ids:
            raise ValueError("answerable cases require at least one relevant chunk")
        unknown = set(self.graded_relevance) - set(self.relevant_chunk_ids)
        if unknown:
            raise ValueError(f"graded relevance contains unknown chunks: {sorted(unknown)}")
        return self


class RetrievalHit(StrictModel):
    chunk_id: str
    document_id: str
    document_version_id: str
    score: float = 0.0


class RetrievalRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    strategy: Literal["dense", "sparse", "hybrid", "hybrid_rerank", "hybrid_graph"]
    hits: list[RetrievalHit] = Field(default_factory=list)
    answer: str | None = None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    invalid_citation_ids: list[str] = Field(default_factory=list)
    abstained: bool | None = None
    supported_claims: int | None = Field(default=None, ge=0)
    unsupported_claims: int | None = Field(default=None, ge=0)
    retrieval_ms: float = Field(default=0.0, ge=0)
    rerank_ms: float = Field(default=0.0, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    peak_cpu_ram_mb: float | None = Field(default=None, ge=0)
    peak_gpu_vram_mb: float | None = Field(default=None, ge=0)
