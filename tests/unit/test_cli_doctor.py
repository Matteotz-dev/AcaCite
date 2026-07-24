from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from cli.rag import doctor_report


def test_doctor_offline_reports_local_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    database = data_root / "provenance.sqlite3"
    data_root.mkdir()
    database.write_text("", encoding="utf-8")
    monkeypatch.setenv("RAG_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(database))
    monkeypatch.setenv("APPROVED_INGESTION_ROOTS", f'["{tmp_path}"]')
    get_settings.cache_clear()

    try:
        report = doctor_report("http://127.0.0.1:9", offline=True)
    finally:
        get_settings.cache_clear()

    assert report["status"] == "ok"
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["api_health"]["status"] == "skipped"
    assert checks["provenance_db"]["status"] == "ok"
