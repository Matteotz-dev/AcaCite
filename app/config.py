"""Typed configuration for the local research RAG service.

Importing this module is side-effect free: runtime directories and external
connections are created by service startup, not while settings are parsed.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Validated environment contract shared by API and worker processes."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Environment variables in the deployment contract are uppercase.
        case_sensitive=False,
    )

    rag_data_root: Path = Field(default=PROJECT_ROOT / "data")
    provenance_db_path: Path | None = None
    qdrant_url: str | None = None
    qdrant_path: Path | None = None
    qdrant_collection: str = "research_chunks_v1"
    index_version: str = "v1"

    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    dense_embedding_dimensions: int = Field(default=384, gt=0)
    sparse_embedding_model: str = "Qdrant/bm25"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_device: str = "cpu"
    reranker_device: str = "cuda"

    dense_candidates: int = Field(default=40, ge=1, le=1000)
    sparse_candidates: int = Field(default=40, ge=1, le=1000)
    graph_candidates: int = Field(default=10, ge=0, le=1000)
    cognee_search_timeout_seconds: float = Field(default=3.0, gt=0, le=60)
    cognee_promotion_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    cognee_graph_enabled: bool = True
    rerank_candidates: int = Field(default=30, ge=1, le=1000)
    final_context_chunks: int = Field(default=10, ge=1, le=100)
    answer_context_tokens: int = Field(default=12_000, ge=512, le=20_000)

    ollama_base_url: str = "http://127.0.0.1:11434"
    default_generator_model: str = "qwen3.6:27b"
    ollama_generation_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    default_generation_temperature: float = Field(default=0.0, ge=0, le=2)
    rag_api_host: str = "127.0.0.1"
    rag_api_port: int = Field(default=8000, ge=1, le=65535)
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8001, ge=1, le=65535)

    approved_ingestion_roots: tuple[Path, ...] = (PROJECT_ROOT,)

    @model_validator(mode="after")
    def resolve_local_paths(self) -> "Settings":
        self.rag_data_root = self.rag_data_root.expanduser().resolve()
        if self.provenance_db_path is None:
            self.provenance_db_path = self.rag_data_root / "provenance.sqlite3"
        else:
            self.provenance_db_path = self.provenance_db_path.expanduser().resolve()
        if self.qdrant_path is None:
            self.qdrant_path = self.rag_data_root / "qdrant"
        else:
            self.qdrant_path = self.qdrant_path.expanduser().resolve()
        if self.qdrant_url is not None:
            self.qdrant_url = self.qdrant_url.strip() or None
        self.ollama_base_url = self.ollama_base_url.rstrip("/")
        self.approved_ingestion_roots = tuple(
            root.expanduser().resolve() for root in self.approved_ingestion_roots
        )
        if self.rerank_candidates > self.dense_candidates + self.sparse_candidates:
            raise ValueError(
                "RERANK_CANDIDATES cannot exceed the dense+sparse candidate pool"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable-in-practice settings instance per process."""

    return Settings()
