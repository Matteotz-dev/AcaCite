"""Grounded answer orchestration and citation validation."""

from __future__ import annotations

import re
from dataclasses import asdict
from time import perf_counter
from typing import Protocol
from uuid import UUID

from app.db import ProvenanceRepository
from app.retrieval.service import RetrievalService, SearchFilters
from .citations import CitationResolver, citation_id, extract_citation_ids
from .ollama import OllamaResult


SYSTEM_PROMPT = """You answer only from the supplied evidence.
Every factual claim supported by a source must include one or more exact [SRC:uuid] tokens
copied from that evidence. Never invent or alter a source ID. If evidence is insufficient,
say so plainly. Label inferences as inferences. Describe source disagreements explicitly.
Do not cite general memory or knowledge outside the evidence."""


class Generator(Protocol):
    def generate(self, *, model: str, system: str, prompt: str,
                 temperature: float) -> OllamaResult: ...


def format_evidence(results) -> str:
    sections = []
    for item in results:
        payload = item.payload
        label = payload.get("title") or payload.get("canonical_uri") or "source"
        sections.append(
            f"{citation_id(item.chunk_id)}\nSource: {label}\nEvidence:\n{payload.get('text', '')}"
        )
    return "\n\n---\n\n".join(sections)


def has_uncited_claim_warning(answer: str, valid_ids: set[str]) -> bool:
    """Conservative signal, not a claim-level proof: flag substantive uncited prose."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    substantive = [value for value in sentences if len(value.split()) >= 8]
    return any(not any(identifier in sentence for identifier in valid_ids)
               for sentence in substantive)


class AnswerService:
    def __init__(self, *, retrieval: RetrievalService, repository: ProvenanceRepository,
                 generator: Generator, approved_roots=()):
        self.retrieval = retrieval
        self.repository = repository
        self.generator = generator
        self.resolver = CitationResolver(repository, tuple(approved_roots))

    def answer(self, *, query: str, filters: SearchFilters, model: str,
               temperature: float) -> dict:
        started = perf_counter()
        search_started = perf_counter()
        search = self.retrieval.search(query, filters)
        search_ms = (perf_counter() - search_started) * 1000
        selected = search.results
        allowed = {item.chunk_id for item in selected}
        if not selected:
            answer = "I have insufficient indexed evidence to answer this question."
            generated_model = model
            generation_ms = 0.0
        else:
            generation_started = perf_counter()
            result = self.generator.generate(
                model=model, system=SYSTEM_PROMPT,
                prompt=f"Question:\n{query}\n\nEvidence:\n{format_evidence(selected)}",
                temperature=temperature,
            )
            generation_ms = (perf_counter() - generation_started) * 1000
            answer, generated_model = result.text, result.model
        valid, invalid = self.resolver.validate(answer, allowed)
        valid_ids = {item.citation_id for item in valid}
        latency = {"search_ms": search_ms, "generation_ms": generation_ms,
                   "total_ms": (perf_counter() - started) * 1000}
        filter_values = {key: value for key, value in asdict(filters).items() if value is not None}
        trace_id = self.repository.record_answer_trace(
            query=query, generator_model=generated_model, filters=filter_values,
            retrieved_chunk_ids=[item.chunk_id for item in search.results],
            selected_chunk_ids=[item.chunk_id for item in selected],
            citation_ids=extract_citation_ids(answer), latency=latency,
        )
        return {
            "query": query, "answer": answer, "model": generated_model,
            "citations": [item.model_dump(mode="json") for item in valid],
            "invalid_citations": invalid,
            "warnings": (["substantive prose may contain uncited claims"]
                         if selected and has_uncited_claim_warning(answer, valid_ids) else []),
            "selected_chunk_ids": [str(item.chunk_id) for item in selected],
            "trace_id": trace_id, "latency": latency,
        }
