"""Explicit parser dispatch for supported first-release formats."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import code, markdown, pdf, plaintext
from .common import NormalizedDocument
from .repository import OPENFOAM_NAMES


Parser = Callable[[Path], NormalizedDocument]
PARSERS: dict[str, Parser] = {
    ".pdf": pdf.parse,
    ".md": markdown.parse,
    ".markdown": markdown.parse,
    ".txt": plaintext.parse,
    ".rst": plaintext.parse,
}
CODE_SUFFIXES = {
    ".py", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".sh", ".bash",
    ".json", ".yaml", ".yml", ".toml",
}


def parser_for(path: Path) -> Parser:
    if path.name in OPENFOAM_NAMES or path.suffix.lower() in CODE_SUFFIXES:
        return code.parse
    try:
        return PARSERS[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported ingestion format: {path}") from exc


def parse_file(path: Path) -> NormalizedDocument:
    return parser_for(path)(path)
