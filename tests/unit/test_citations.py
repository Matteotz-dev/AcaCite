from pathlib import Path
from uuid import uuid4

from app.db import ProvenanceRepository, sha256_text
from app.generation.citations import CitationResolver, citation_id, extract_citation_ids
from app.models import ChunkCreate, DocumentCreate, DocumentVersionCreate, SourceType


def indexed_paper(tmp_path: Path):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"fixture")
    repository = ProvenanceRepository(tmp_path / "provenance.sqlite3")
    repository.initialize()
    document = repository.upsert_document(
        DocumentCreate(
            source_type=SourceType.PAPER,
            canonical_uri=source.as_uri(),
            title="Gradient Models",
            content_hash=sha256_text("paper"),
            dataset="papers",
        )
    )
    version = repository.create_version(
        DocumentVersionCreate(
            document_id=document.id,
            content_hash=sha256_text("paper"),
            parser_name="fixture",
            parser_version="1",
            index_version="v1",
            doi="10.0/test",
        )
    )
    chunk = repository.add_chunk(
        ChunkCreate(
            document_version_id=version.id,
            ordinal=0,
            text_hash=sha256_text("gradient"),
            text="gradient",
            token_count=1,
            chunk_type="section",
            heading_path=("Methods",),
            page_start=4,
            page_end=5,
        )
    )
    repository.mark_version_current(version.id)
    return repository, chunk, source


def test_citation_resolution_and_safe_open_path(tmp_path: Path):
    repository, chunk, source = indexed_paper(tmp_path)
    citation = CitationResolver(repository, (tmp_path,)).resolve(chunk.id)
    assert citation.citation_id == citation_id(chunk.id)
    assert citation.display == "Gradient Models, p. 4-5"
    assert citation.metadata["open_path"] == str(source)
    assert citation.metadata["heading_path"] == ("Methods",)


def test_parser_deduplicates_and_validator_rejects_unknown_ids(tmp_path: Path):
    repository, chunk, _ = indexed_paper(tmp_path)
    unknown = uuid4()
    answer = f"Known {citation_id(chunk.id)} twice {citation_id(chunk.id)} fake {citation_id(unknown)}"
    assert extract_citation_ids(answer) == (citation_id(chunk.id), citation_id(unknown))
    valid, invalid = CitationResolver(repository).validate(answer, {chunk.id})
    assert [item.chunk_id for item in valid] == [chunk.id]
    assert invalid == [citation_id(unknown)]


def test_validator_reports_malformed_and_truncated_citations(tmp_path: Path):
    repository, chunk, _ = indexed_paper(tmp_path)
    answer = "Malformed [SRC:not-a-uuid] and truncated [SRC:1234]"

    valid, invalid = CitationResolver(repository).validate(answer, {chunk.id})

    assert valid == []
    assert invalid == ["[SRC:not-a-uuid]", "[SRC:1234]"]


def test_resolver_does_not_open_source_outside_approved_root(tmp_path: Path):
    repository, chunk, _ = indexed_paper(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    citation = CitationResolver(repository, (other_root,)).resolve(chunk.id)
    assert citation.metadata["open_path_blocked"] is True
    assert "open_path" not in citation.metadata
