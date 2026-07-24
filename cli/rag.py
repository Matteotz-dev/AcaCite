"""Operator CLI for repository, directory, status, retry, and sync workflows."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"


def main() -> int:
    parser = argparse.ArgumentParser(prog="acacite")
    parser.add_argument("--base-url", default=BASE_URL)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest-repo", "ingest-dir", "sync"):
        command = commands.add_parser(name)
        command.add_argument("path")
        command.add_argument("--dataset", required=True)
        command.add_argument("--project")
        if name == "sync":
            command.add_argument("--kind", choices=("repo", "directory"), default="repo")
    status = commands.add_parser("status")
    status.add_argument("--job")
    status.add_argument("--limit", type=int, default=20)
    retry = commands.add_parser("retry")
    retry.add_argument("job_id")
    args = parser.parse_args()
    try:
        if args.command == "status":
            path = f"/v1/ingestion/jobs/{args.job}" if args.job else f"/v1/ingestion/status?limit={args.limit}"
            result = request(args.base_url, "GET", path)
        elif args.command == "retry":
            result = request(args.base_url, "POST", f"/v1/ingestion/jobs/{args.job_id}/retry", {})
        else:
            kind = args.kind if args.command == "sync" else (
                "repo" if args.command == "ingest-repo" else "directory"
            )
            result = request(args.base_url, "POST", f"/v1/ingestion/{kind}", {
                "path": str(Path(args.path).expanduser()), "dataset": args.dataset,
                "project": args.project, "delete_missing": args.command == "sync",
            })
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (HTTPError, URLError) as exc:
        detail = exc.read().decode() if isinstance(exc, HTTPError) else str(exc)
        parser.exit(1, f"rag: {detail}\n")


def request(base_url: str, method: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("ACACITE_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        base_url.rstrip("/") + path, data=data, method=method,
        headers=headers,
    )
    with urlopen(request, timeout=3600) as response:
        return json.load(response)


if __name__ == "__main__":
    raise SystemExit(main())
