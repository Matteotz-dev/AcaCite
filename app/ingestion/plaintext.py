"""Deterministic line-aware parser for plain text and reStructuredText."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from .common import NormalizedChunk, NormalizedDocument


PARSER_VERSION = "1"


def _paragraphs(text: str) -> list[tuple[str, int, int]]:
    lines = text.splitlines()
    result: list[tuple[str, int, int]] = []
    start: int | None = None
    buffer: list[str] = []
    for number, line in enumerate(lines, 1):
        if line.strip():
            if start is None:
                start = number
            buffer.append(line.rstrip())
        elif buffer:
            result.append(("\n".join(buffer).strip(), start or number, number - 1))
            start, buffer = None, []
    if buffer:
        result.append(("\n".join(buffer).strip(), start or 1, len(lines)))
    return result


def parse(path: Path) -> NormalizedDocument:
    text = path.read_text(encoding="utf-8")
    chunks = tuple(
        NormalizedChunk(
            text=value, chunk_type="paragraph", line_start=start, line_end=end,
            language="text",
        )
        for value, start, end in _paragraphs(text)
        if value
    )
    return NormalizedDocument(
        source_path=path.resolve(), title=path.stem,
        mime_type=mimetypes.guess_type(path)[0] or "text/plain", language="text",
        parser_name="plaintext", parser_version=PARSER_VERSION, chunks=chunks,
    )
