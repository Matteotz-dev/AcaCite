from pathlib import Path

from app.config import Settings
from app.db import ProvenanceRepository
from app.ingestion.service import ingest_file
from app.models import VersionStatus


def test_ingest_file_is_idempotent_and_stops_at_parsed(tmp_path: Path):
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nStable evidence.\n", encoding="utf-8")
    settings = Settings(
        rag_data_root=tmp_path / "data",
        provenance_db_path=tmp_path / "data" / "provenance.sqlite3",
        approved_ingestion_roots=(tmp_path,),
    )
    first_version, first_chunks = ingest_file(source, dataset="fixture", settings=settings)
    second_version, second_chunks = ingest_file(source, dataset="fixture", settings=settings)

    assert first_version.id == second_version.id
    assert first_version.status is VersionStatus.PARSED
    assert [chunk.id for chunk in first_chunks] == [chunk.id for chunk in second_chunks]
    assert ProvenanceRepository(settings.provenance_db_path).get_document(
        first_version.document_id
    ).current_version_id is None
