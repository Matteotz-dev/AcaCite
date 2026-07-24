#!/usr/bin/env python3
"""Fail if files intended for publication contain private or generated artifacts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


BLOCKED_NAMES = {
    ".env",
    "provenance.sqlite3",
}
BLOCKED_PARTS = {
    ".cognee_data",
    ".pytest_cache",
    ".venv",
    "backups",
    "__pycache__",
}
BLOCKED_SUFFIXES = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".lance",
    ".pdf",
    ".pt",
    ".bin",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".pyc",
    ".pyo",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".css",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_CONTENT_SCAN = {
    Path("scripts/check_public_tree.py"),
}
SECRET_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"/home/mugliotti3\b",
        r"/usr/local/home/mugliotti3\b",
        r"mugliotti3@gatech\.edu",
        r"github_pat_[A-Za-z0-9_]+",
        r"ghp_[A-Za-z0-9_]+",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"OPENAI_API_KEY\s*=\s*['\"]?sk-",
    )
]


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line) for line in output.splitlines() if line.strip()]


def main() -> int:
    errors: list[str] = []
    for path in tracked_files():
        if path.name in BLOCKED_NAMES:
            errors.append(f"blocked file name: {path}")
        if any(part in BLOCKED_PARTS for part in path.parts):
            errors.append(f"blocked path component: {path}")
        if path.suffix in BLOCKED_SUFFIXES:
            errors.append(f"blocked generated/binary suffix: {path}")
        if path in SKIP_CONTENT_SCAN or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"blocked private pattern {pattern.pattern!r}: {path}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("public tree check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
