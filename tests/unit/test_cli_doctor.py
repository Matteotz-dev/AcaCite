from __future__ import annotations

from pathlib import Path
import sqlite3

from app.config import get_settings
from cli.rag import doctor_report


def _write_minimal_database(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                deleted_at TEXT
            );
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY
            );
            """
        )


def test_doctor_offline_reports_local_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    database = data_root / "provenance.sqlite3"
    data_root.mkdir()
    _write_minimal_database(database)
    monkeypatch.setenv("RAG_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(database))
    monkeypatch.setenv("APPROVED_INGESTION_ROOTS", f'["{tmp_path}"]')
    monkeypatch.delenv("ACACITE_API_TOKEN", raising=False)
    get_settings.cache_clear()

    try:
        report = doctor_report("http://127.0.0.1:9", offline=True)
    finally:
        get_settings.cache_clear()

    assert report["status"] == "ok"
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["api_health"]["status"] == "skipped"
    assert checks["mcp_health"]["status"] == "skipped"
    assert checks["provenance_db"]["status"] == "ok"
    assert checks["sqlite_integrity"]["status"] == "ok"


def test_doctor_warns_when_configured_token_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    database = data_root / "provenance.sqlite3"
    data_root.mkdir()
    _write_minimal_database(database)
    monkeypatch.setenv("RAG_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(database))
    monkeypatch.setenv("APPROVED_INGESTION_ROOTS", f'["{tmp_path}"]')
    monkeypatch.delenv("ACACITE_API_TOKEN", raising=False)
    monkeypatch.setenv("acacite_api_token", "configured")
    get_settings.cache_clear()

    try:
        report = doctor_report("http://127.0.0.1:9", offline=True)
    finally:
        get_settings.cache_clear()

    checks = {item["name"]: item for item in report["checks"]}
    assert report["status"] == "warn"
    assert checks["api_token"]["status"] == "warn"
