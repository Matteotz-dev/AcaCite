"""JSONL loading with duplicate detection and actionable line errors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def load_jsonl(path: Path, model: type[T], *, identity: str) -> list[T]:
    rows, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = model.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        key = getattr(row, identity)
        if key in seen:
            raise ValueError(f"{path}:{number}: duplicate {identity} {key}")
        seen.add(key)
        rows.append(row)
    return rows
