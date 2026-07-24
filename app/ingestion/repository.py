"""Git-aware, gitignore-respecting repository discovery."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.db import validate_approved_path


SUPPORTED_SUFFIXES = {
    ".md", ".markdown", ".rst", ".txt", ".pdf", ".py", ".c", ".h", ".cc",
    ".cpp", ".cxx", ".hpp", ".sh", ".bash", ".json", ".yaml", ".yml", ".toml",
}
OPENFOAM_NAMES = {"controlDict", "fvSchemes", "fvSolution", "momentumTransport", "transportProperties"}


class RepositorySnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root: Path
    commit: str | None
    branch: str | None
    dirty: bool
    files: tuple[Path, ...]


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=check, capture_output=True, text=True,
    )
    return result.stdout.strip()


def discover_repository(path: Path, approved_roots: tuple[Path, ...]) -> RepositorySnapshot:
    root = validate_approved_path(path, approved_roots)
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    validate_approved_path(top, approved_roots)
    raw = subprocess.run(
        ["git", "-C", str(top), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True, capture_output=True,
    ).stdout
    files: list[Path] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        relative = Path(item.decode("utf-8", errors="surrogateescape"))
        # A tracked working-tree deletion remains in ``git ls-files --cached``;
        # leave it out of the snapshot so sync can tombstone its prior record.
        if not (top / relative).exists():
            continue
        candidate = validate_approved_path(top / relative, approved_roots)
        if candidate.is_file() and (candidate.suffix.lower() in SUPPORTED_SUFFIXES or candidate.name in OPENFOAM_NAMES):
            files.append(candidate)
    branch = _git(top, "symbolic-ref", "--short", "-q", "HEAD", check=False) or None
    commit = _git(top, "rev-parse", "HEAD", check=False) or None
    dirty = bool(_git(top, "status", "--porcelain", "--untracked-files=normal", check=False))
    return RepositorySnapshot(root=top, commit=commit, branch=branch, dirty=dirty, files=tuple(sorted(files)))
