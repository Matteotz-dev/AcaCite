"""Compatibility entrypoint for the local shared-memory API."""

from app.api import SearchType, app, cognee, shared_memory

__all__ = ["SearchType", "app", "cognee", "shared_memory"]
