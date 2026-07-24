import sqlite3
from pathlib import Path

import pytest

from app.db import migrations


def test_failed_migration_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken_schema = tmp_path / "broken.sql"
    broken_schema.write_text(
        "CREATE TABLE should_rollback (id INTEGER);\nTHIS IS NOT SQL;",
        encoding="utf-8",
    )
    monkeypatch.setattr(migrations, "SCHEMA_PATH", broken_schema)
    connection = sqlite3.connect(tmp_path / "broken.sqlite3")

    with pytest.raises(sqlite3.DatabaseError):
        migrations.migrate(connection)

    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "should_rollback" not in tables
    assert connection.execute(
        "SELECT COUNT(*) FROM schema_migrations"
    ).fetchone()[0] == 0
    connection.close()


def test_newer_schema_is_rejected(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "newer.sqlite3")
    connection.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (migrations.SCHEMA_VERSION + 1, "now"),
    )

    with pytest.raises(RuntimeError, match="newer than supported"):
        migrations.migrate(connection)
    connection.close()
