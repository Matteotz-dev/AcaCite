"""Heading-aware Markdown parser with exact source line spans."""

from __future__ import annotations

import re
from pathlib import Path

from .common import NormalizedChunk, NormalizedDocument


PARSER_VERSION = "1"
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse(path: Path) -> NormalizedDocument:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[str] = []
    chunks: list[NormalizedChunk] = []
    section_start = 1
    section_lines: list[str] = []

    def flush(end: int) -> None:
        text = "\n".join(section_lines).strip()
        if text:
            chunks.append(NormalizedChunk(
                text=text, chunk_type="section", heading_path=tuple(headings),
                line_start=section_start, line_end=end, language="markdown",
            ))

    title: str | None = None
    for number, line in enumerate(lines, 1):
        match = HEADING.match(line)
        if not match:
            section_lines.append(line.rstrip())
            continue
        flush(number - 1)
        level, name = len(match.group(1)), match.group(2).strip()
        title = title or name
        headings[:] = headings[: level - 1]
        headings.append(name)
        section_start = number
        section_lines = [line.rstrip()]
    flush(len(lines))
    return NormalizedDocument(
        source_path=path.resolve(), title=title or path.stem,
        mime_type="text/markdown", language="markdown", parser_name="markdown",
        parser_version=PARSER_VERSION, chunks=tuple(chunks),
    )
