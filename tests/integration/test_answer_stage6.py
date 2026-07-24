import json
from pathlib import Path
from uuid import uuid4

from app.db import ProvenanceRepository, sha256_text
from app.generation.ollama import OllamaResult, OllamaUnavailable
from app.generation.service import AnswerService
from app.models import ChunkCreate, DocumentCreate, DocumentVersionCreate, SourceType
from app.retrieval.fusion import Candidate
from app.retrieval.service import SearchFilters, SearchResponse


class FakeRetrieval:
    def __init__(self, results):
        self.results = results

    def search(self, query, filters):
        return SearchResponse(self.results, {"total_ms": 1})


class FakeGenerator:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return OllamaResult(self.text, kwargs["model"])


def fixture_source(tmp_path: Path):
    repository = ProvenanceRepository(tmp_path / "p.sqlite3")
    repository.initialize()
    document = repository.upsert_document(DocumentCreate(
        source_type=SourceType.NOTE, canonical_uri=(tmp_path / "note.md").as_uri(),
        title="Note", content_hash=sha256_text("evidence"), dataset="notes",
    ))
    version = repository.create_version(DocumentVersionCreate(
        document_id=document.id, content_hash=sha256_text("evidence"), parser_name="test",
        parser_version="1", index_version="v1",
    ))
    chunk = repository.add_chunk(ChunkCreate(
        document_version_id=version.id, ordinal=0, text_hash=sha256_text("evidence"),
        text="The measured value is 42.", token_count=6, chunk_type="text", line_start=1,
    ))
    repository.mark_version_current(version.id)
    return repository, chunk


def test_answer_validates_citations_and_persists_observable_trace(tmp_path):
    repository, chunk = fixture_source(tmp_path)
    invented = uuid4()
    generator = FakeGenerator(
        f"The value is 42 [SRC:{chunk.id}]. Unsupported [SRC:{invented}]."
    )
    candidate = Candidate(chunk.id, {"text": chunk.text, "title": "Note"})
    service = AnswerService(
        retrieval=FakeRetrieval([candidate]), repository=repository,
        generator=generator, approved_roots=(tmp_path,),
    )
    response = service.answer(
        query="value?", filters=SearchFilters(dataset="notes"),
        model="qwen3-coder:30b", temperature=0,
    )
    assert [item["chunk_id"] for item in response["citations"]] == [str(chunk.id)]
    assert response["invalid_citations"] == [f"[SRC:{invented}]"]
    assert generator.calls[0]["model"] == "qwen3-coder:30b"
    with repository.connect() as connection:
        trace = connection.execute("SELECT * FROM answer_traces").fetchone()
    assert json.loads(trace["selected_chunk_ids"]) == [str(chunk.id)]
    assert json.loads(trace["citation_ids_json"]) == [f"[SRC:{chunk.id}]", f"[SRC:{invented}]"]
    assert "prompt" not in trace.keys()


def test_empty_evidence_returns_insufficient_answer_without_calling_generator(tmp_path):
    repository = ProvenanceRepository(tmp_path / "p.sqlite3")
    repository.initialize()
    generator = FakeGenerator("must not be used")
    response = AnswerService(
        retrieval=FakeRetrieval([]), repository=repository, generator=generator
    ).answer(query="unknown", filters=SearchFilters(), model="qwen", temperature=0)
    assert "insufficient" in response["answer"]
    assert generator.calls == []
    assert response["warnings"] == []


def test_search_is_generator_independent_when_generator_is_unavailable(tmp_path):
    repository, chunk = fixture_source(tmp_path)
    candidate = Candidate(chunk.id, {"text": chunk.text})

    class Offline:
        def generate(self, **kwargs):
            raise OllamaUnavailable("offline")

    retrieval = FakeRetrieval([candidate])
    assert retrieval.search("value", SearchFilters()).results == [candidate]
    service = AnswerService(retrieval=retrieval, repository=repository, generator=Offline())
    try:
        service.answer(query="value", filters=SearchFilters(), model="qwen", temperature=0)
    except OllamaUnavailable as exc:
        assert "offline" in str(exc)
    else:
        raise AssertionError("answer should expose generator dependency failure")


def test_switching_generator_models_reuses_identical_evidence_without_ingestion(tmp_path):
    repository, chunk = fixture_source(tmp_path)
    candidate = Candidate(chunk.id, {"text": chunk.text})
    generator = FakeGenerator(f"The value is 42 [SRC:{chunk.id}].")
    retrieval = FakeRetrieval([candidate])
    service = AnswerService(
        retrieval=retrieval, repository=repository, generator=generator,
        approved_roots=(tmp_path,),
    )
    before = repository.current_version_ids()
    first = service.answer(
        query="value", filters=SearchFilters(), model="qwen3-coder:30b", temperature=0
    )
    second = service.answer(
        query="value", filters=SearchFilters(), model="devstral-small-2:24b", temperature=0
    )
    assert first["selected_chunk_ids"] == second["selected_chunk_ids"] == [str(chunk.id)]
    assert [call["model"] for call in generator.calls] == [
        "qwen3-coder:30b", "devstral-small-2:24b"
    ]
    assert repository.current_version_ids() == before


def test_v1_answer_reports_clear_503_for_missing_ollama(monkeypatch):
    from fastapi.testclient import TestClient
    from app import api

    class OfflineAnswerService:
        def answer(self, **kwargs):
            raise OllamaUnavailable("Ollama is unavailable: connection refused")

    monkeypatch.setattr(api, "_rag_answer_service", lambda: OfflineAnswerService())
    response = TestClient(api.app).post("/v1/answer", json={"query": "what?"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "ollama_unavailable"
