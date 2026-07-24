"""Operator CLI for repository, directory, status, retry, citation, and diagnostics workflows."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import json
import sqlite3
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
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--offline", action="store_true", help="Skip the live API health check.")
    doctor.add_argument("--mcp-url", default="http://127.0.0.1:8001/mcp")
    expand = commands.add_parser("expand-citations")
    expand.add_argument("--database", type=Path)
    expand.add_argument("--paper-title", action="append", required=True)
    expand.add_argument("--output", type=Path, required=True)
    expand.add_argument("--depth", type=int, default=2, choices=(1, 2))
    expand.add_argument("--download-oa-pdfs", action="store_true")
    expand.add_argument("--pdf-dir", type=Path)
    expand.add_argument("--timeout", type=float, default=20.0)
    expand.add_argument("--polite-sleep", type=float, default=0.2)
    expand.add_argument("--max-pdf-mb", type=int, default=80)
    expand.add_argument("--ingest-downloaded", action="store_true")
    expand.add_argument("--dataset")
    expand.add_argument("--project")
    args = parser.parse_args()
    try:
        if args.command == "status":
            path = f"/v1/ingestion/jobs/{args.job}" if args.job else f"/v1/ingestion/status?limit={args.limit}"
            result = request(args.base_url, "GET", path)
        elif args.command == "retry":
            result = request(args.base_url, "POST", f"/v1/ingestion/jobs/{args.job_id}/retry", {})
        elif args.command == "doctor":
            result = doctor_report(args.base_url, mcp_url=args.mcp_url, offline=args.offline)
        elif args.command == "expand-citations":
            result = expand_citations(args, args.base_url)
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


def doctor_report(
    base_url: str, *, offline: bool, mcp_url: str = "http://127.0.0.1:8001/mcp"
) -> dict:
    from app.config import get_settings

    settings = get_settings()
    checks = []

    def add(name: str, status: str, detail: str | None = None) -> None:
        checks.append({"name": name, "status": status, **({"detail": detail} if detail else {})})

    try:
        version = importlib.metadata.version("acacite")
    except importlib.metadata.PackageNotFoundError:
        version = "editable-or-uninstalled"
    add("package", "ok", version)
    add(
        "data_root",
        "ok" if settings.rag_data_root.exists() and os.access(settings.rag_data_root, os.W_OK) else "warn",
        str(settings.rag_data_root),
    )
    add(
        "provenance_db", "ok" if settings.provenance_db_path and settings.provenance_db_path.exists() else "warn",
        str(settings.provenance_db_path),
    )
    if settings.provenance_db_path and settings.provenance_db_path.exists():
        try:
            with sqlite3.connect(settings.provenance_db_path) as connection:
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                documents = connection.execute(
                    "SELECT count(*) FROM documents WHERE deleted_at IS NULL"
                ).fetchone()[0]
                chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            add("sqlite_integrity", "ok" if integrity == "ok" else "fail", str(integrity))
            add("indexed_corpus", "ok", f"documents={documents}, chunks={chunks}")
        except sqlite3.DatabaseError as exc:
            add("sqlite_integrity", "fail", str(exc))
    if settings.qdrant_url:
        add("qdrant_mode", "ok", f"server:{settings.qdrant_url}")
    else:
        add("qdrant_mode", "ok" if settings.qdrant_path and settings.qdrant_path.exists() else "warn",
            f"local:{settings.qdrant_path}")
    roots = [str(root) for root in settings.approved_ingestion_roots]
    missing_roots = [root for root in settings.approved_ingestion_roots if not root.exists()]
    add("approved_ingestion_roots", "warn" if missing_roots else "ok", json.dumps(roots))
    token_enabled = bool(settings.acacite_api_token and settings.acacite_api_token.strip())
    token_available = bool(os.environ.get("ACACITE_API_TOKEN", "").strip())
    if token_enabled:
        add(
            "api_token",
            "ok" if token_available else "warn",
            "ACACITE_API_TOKEN is configured; API clients and MCP must forward it",
        )
    else:
        add("api_token", "ok", "disabled")
    if offline:
        add("api_health", "skipped", "offline")
        add("mcp_health", "skipped", "offline")
    else:
        try:
            health = request(base_url, "GET", "/v1/health")
            add("api_health", "ok", health.get("status", "reachable"))
        except (HTTPError, URLError) as exc:
            detail = exc.read().decode() if isinstance(exc, HTTPError) else str(exc)
            add("api_health", "fail", detail)
        try:
            request(mcp_url, "POST", "", {
                "jsonrpc": "2.0", "id": "doctor", "method": "tools/list", "params": {},
            })
            add("mcp_health", "ok", "reachable")
        except (HTTPError, URLError, ValueError) as exc:
            detail = exc.read().decode() if isinstance(exc, HTTPError) else str(exc)
            add("mcp_health", "warn", detail)
    overall = "fail" if any(item["status"] == "fail" for item in checks) else (
        "warn" if any(item["status"] == "warn" for item in checks) else "ok"
    )
    return {"status": overall, "checks": checks}


def expand_citations(args, base_url: str) -> dict:
    from app.config import get_settings
    from scripts.expand_citations import (
        download_pdf, expand, extract_seed_references, write_manifest,
    )
    import time

    settings = get_settings()
    database = args.database or settings.provenance_db_path
    if database is None:
        raise SystemExit("acacite: no provenance database configured\n")
    seeds = extract_seed_references(database, args.paper_title)
    works = expand(seeds, args.depth, args.timeout, args.polite_sleep)
    pdf_dir = args.pdf_dir or args.output.parent / "pdfs"
    pdf_attempts = 0
    if args.download_oa_pdfs:
        for work in works.values():
            if work.pdf_url:
                pdf_attempts += 1
            download_pdf(work, pdf_dir, args.timeout, args.max_pdf_mb * 1024 * 1024)
            time.sleep(args.polite_sleep)
    write_manifest(works, args.output)
    resolved = sum(1 for work in works.values() if work.doi or work.openalex_id)
    unresolved = [work.raw_reference for work in works.values() if not (work.doi or work.openalex_id)]
    result = {
        "seed_references": len(seeds),
        "works": len(works),
        "resolved_works": resolved,
        "unresolved_works": len(unresolved),
        "manifest_jsonl": str(args.output.with_suffix(".jsonl")),
        "manifest_markdown": str(args.output.with_suffix(".md")),
        "pdf_dir": str(pdf_dir),
        "pdf_attempts": pdf_attempts,
        "downloaded_pdfs": sum(1 for work in works.values() if work.pdf_path),
        "unresolved_sample": unresolved[:10],
    }
    if args.ingest_downloaded:
        if not args.download_oa_pdfs:
            raise SystemExit("acacite: --ingest-downloaded requires --download-oa-pdfs\n")
        if not args.dataset:
            raise SystemExit("acacite: --ingest-downloaded requires --dataset\n")
        result["ingestion_job"] = request(base_url, "POST", "/v1/ingestion/directory", {
            "path": str(pdf_dir.expanduser()), "dataset": args.dataset,
            "project": args.project, "delete_missing": False,
        })
    return result


if __name__ == "__main__":
    raise SystemExit(main())
