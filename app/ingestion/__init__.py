"""Source ingestion adapters (implemented beginning in Phase 2)."""
"""Local document discovery and normalized parsing."""

from .common import NormalizedChunk, NormalizedDocument, estimate_tokens
from .registry import parse_file, parser_for
from .repository import RepositorySnapshot, discover_repository

__all__ = [
    "NormalizedChunk", "NormalizedDocument", "RepositorySnapshot",
    "discover_repository", "estimate_tokens", "parse_file", "parser_for",
]
