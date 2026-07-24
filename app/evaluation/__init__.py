"""Versioned, offline-first evaluation support for the research RAG."""

from .metrics import evaluate_records
from .schema import GoldCase, RetrievalRecord

__all__ = ["GoldCase", "RetrievalRecord", "evaluate_records"]
