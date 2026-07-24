from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_are_local_and_side_effect_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RAG_DATA_ROOT", "PROVENANCE_DB_PATH", "QDRANT_PATH", "APPROVED_INGESTION_ROOTS"):
        monkeypatch.delenv(name, raising=False)
    data_root = tmp_path / "data"
    settings = Settings(_env_file=None, rag_data_root=data_root)

    assert settings.rag_data_root == data_root.resolve()
    assert settings.provenance_db_path == data_root.resolve() / "provenance.sqlite3"
    assert settings.rag_api_host == "127.0.0.1"
    assert settings.mcp_host == "127.0.0.1"
    assert not data_root.exists()


def test_uppercase_environment_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_VERSION", "test-v2")
    monkeypatch.setenv("ANSWER_CONTEXT_TOKENS", "4096")

    settings = Settings(_env_file=None)

    assert settings.index_version == "test-v2"
    assert settings.answer_context_tokens == 4096


def test_context_budget_has_hard_ceiling() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, answer_context_tokens=20_001)


def test_rerank_pool_cannot_exceed_retrieval_pool() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            dense_candidates=2,
            sparse_candidates=2,
            rerank_candidates=5,
        )
