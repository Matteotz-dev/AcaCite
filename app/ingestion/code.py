"""Deterministic first-pass code and OpenFOAM dictionary chunking."""

from __future__ import annotations

import re
from pathlib import Path

from .common import NormalizedChunk, NormalizedDocument


PARSER_VERSION = "1"
LANGUAGES = {
    ".py": "python", ".c": "c", ".h": "cpp", ".cc": "cpp", ".cpp": "cpp",
    ".cxx": "cpp", ".hpp": "cpp", ".sh": "shell", ".bash": "shell",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
}
PY_SYMBOL = re.compile(r"^(?:async\s+def|def|class)\s+([A-Za-z_]\w*)\b")
CPP_SYMBOL = re.compile(
    r"^(?!\s*(?:if|for|while|switch|catch)\b).*?([~A-Za-z_]\w*(?:::\w+)*)\s*\([^;]*\)\s*(?:const\s*)?(?:\{|$)"
)
FOAM_KEY = re.compile(r"^\s*([A-Za-z_]\w*)\s*$")


def _brace_end(lines: list[str], start: int) -> int:
    depth = 0
    seen = False
    for index in range(start, len(lines)):
        clean = re.sub(r'"(?:\\.|[^"\\])*"|//.*$', "", lines[index])
        opens, closes = clean.count("{"), clean.count("}")
        if opens:
            seen = True
        depth += opens - closes
        if seen and depth <= 0:
            return index
    return max(start, len(lines) - 1)


def _symbol_starts(lines: list[str], language: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    if language == "python":
        for index, line in enumerate(lines):
            match = PY_SYMBOL.match(line)
            if match and len(line) - len(line.lstrip()) == 0:
                found.append((index, match.group(1)))
    elif language in {"c", "cpp"}:
        for index, line in enumerate(lines):
            match = CPP_SYMBOL.match(line.strip())
            if match:
                found.append((index, match.group(1)))
    return found


def _openfoam_chunks(lines: list[str]) -> list[NormalizedChunk]:
    chunks: list[NormalizedChunk] = []
    for index in range(len(lines) - 1):
        match = FOAM_KEY.match(lines[index])
        if match and lines[index + 1].lstrip().startswith("{"):
            end = _brace_end(lines, index + 1)
            text = "\n".join(lines[index : end + 1]).strip()
            chunks.append(NormalizedChunk(
                text=text, chunk_type="config", line_start=index + 1,
                line_end=end + 1, symbol=match.group(1), language="openfoam",
            ))
    return chunks


def parse(path: Path) -> NormalizedDocument:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    language = LANGUAGES.get(path.suffix.lower(), "openfoam")
    chunks: list[NormalizedChunk] = []
    starts = _symbol_starts(lines, language)
    for ordinal, (start, symbol) in enumerate(starts):
        if language == "python":
            end = (starts[ordinal + 1][0] - 1) if ordinal + 1 < len(starts) else len(lines) - 1
        else:
            end = _brace_end(lines, start)
        value = "\n".join(lines[start : end + 1]).strip()
        if value:
            chunks.append(NormalizedChunk(
                text=value, chunk_type="code_symbol", line_start=start + 1,
                line_end=end + 1, symbol=symbol, language=language,
            ))
    if language == "openfoam" or (not chunks and "FoamFile" in text):
        chunks = _openfoam_chunks(lines)
        language = "openfoam"
    if not chunks and text.strip():
        chunks.append(NormalizedChunk(
            text=text.strip(), chunk_type="config" if language in {"json", "yaml", "toml"} else "code",
            line_start=1, line_end=max(1, len(lines)), language=language,
        ))
    return NormalizedDocument(
        source_path=path.resolve(), title=path.name, mime_type="text/plain",
        language=language, parser_name="code", parser_version=PARSER_VERSION,
        chunks=tuple(chunks),
    )
