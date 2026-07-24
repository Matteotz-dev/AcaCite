#!/usr/bin/env python3
"""Evaluate recorded retrieval/generation JSONL without live model dependencies."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.io import load_jsonl
from app.evaluation.metrics import evaluate_records
from app.evaluation.schema import GoldCase, RetrievalRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = load_jsonl(args.gold, GoldCase, identity="id")
    # A case can have one record per strategy, so use the compound key manually.
    records = []
    for line in args.records.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            records.append(RetrievalRecord.model_validate_json(line))
    report = evaluate_records(cases, records)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
