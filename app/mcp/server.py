"""Retrieval-only Continue-facing tools over the stable local HTTP API."""

from __future__ import annotations

import re
import time
from typing import Any
from pathlib import Path

from app.config import get_settings
from .client import ResearchAPIClient


def _compact_search_response(response: dict[str, Any]) -> dict[str, Any]:
    """Keep model-facing evidence useful without flooding the chat context."""
    compact_results = []
    for item in response.get("results", []):
        payload = item.get("payload", {})
        compact_results.append({
            "citation_id": item.get("chunk_id") or payload.get("chunk_id"),
            "title": payload.get("title"),
            "source": payload.get("canonical_uri"),
            "page_start": payload.get("page_start"),
            "page_end": payload.get("page_end"),
            "heading": payload.get("heading_path", []),
            "evidence": payload.get("text", ""),
            "score": item.get("reranker_score", item.get("fused_score")),
        })
    trace = response.get("trace", {})
    return {
        "query": response.get("query"),
        "results": compact_results,
        "retrieval_ms": trace.get("total_ms"),
        "result_count": len(compact_results),
    }


def _filters(
    dataset: str | None, project: str | None,
    source_type: str | None, language: str | None,
) -> dict[str, Any]:
    return {key: value for key, value in {
        "dataset": dataset, "project": project,
        "source_type": source_type, "language": language,
    }.items() if value is not None}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())[:80].strip("-") or "citation-expansion"


def create_mcp(client: ResearchAPIClient | None = None, server_factory=None):
    settings = get_settings()
    api = client or ResearchAPIClient()
    if server_factory is None:
        from mcp.server.fastmcp import FastMCP

        server_factory = FastMCP
    server = server_factory(
        "acacite",
        instructions=(
            "Use research_search when you will inspect and reason over evidence yourself. "
            "This MCP surface is retrieval-only and never invokes a local answer model. "
            "Citations and source paths are authorized by the research API."
        ),
        host=settings.mcp_host, port=settings.mcp_port,
        streamable_http_path="/mcp", stateless_http=True, json_response=True,
    )

    @server.tool()
    def research_search(
        query: str, dataset: str | None = None, project: str | None = None,
        source_type: str | None = None, language: str | None = None,
    ) -> dict[str, Any]:
        """Search indexed evidence for agent inspection; does not call an answer model."""
        response = api.post("/v1/search", {
            "query": query, **_filters(dataset, project, source_type, language)
        })
        return _compact_search_response(response)

    @server.tool()
    def research_remember(
        content: str, source_uri: str, dataset: str,
        title: str | None = None, project: str | None = None,
        promote_to_cognee: bool = False, promotion_kind: str = "decision",
        cognee_dataset: str | None = None,
    ) -> dict[str, Any]:
        """Explicitly index a provenance-bearing note; Cognee promotion is opt-in."""
        return api.post("/v1/memory", {
            "content": content, "source_uri": source_uri, "dataset": dataset,
            "title": title, "project": project,
            "promote_to_cognee": promote_to_cognee,
            "promotion_kind": promotion_kind, "cognee_dataset": cognee_dataset,
        })

    @server.tool()
    def research_open_source(chunk_id: str) -> dict[str, Any]:
        """Resolve a citation UUID to safe source-opening metadata; the client performs opening."""
        return api.get(f"/v1/sources/{chunk_id}")

    @server.tool()
    def research_related(chunk_id: str, limit: int = 5) -> dict[str, Any]:
        """Find indexed evidence related to an already authorized source chunk."""
        return api.post("/v1/related", {"chunk_id": chunk_id, "limit": limit})

    @server.tool()
    def research_expand_citations(
        paper_title: str, depth: int = 2, download_oa_pdfs: bool = False,
        output_dir: str = "reports/citation-expansion/mcp",
    ) -> dict[str, Any]:
        """Resolve first/second-hop bibliography papers without invoking an answer model."""
        from scripts.expand_citations import (
            download_pdf, expand, extract_seed_references, write_manifest,
        )

        depth = max(1, min(2, int(depth)))
        output_root = Path(output_dir).expanduser()
        output = output_root / _slug(paper_title)
        seeds = extract_seed_references(settings.provenance_db_path, [paper_title])
        works = expand(seeds, depth, timeout=20.0, polite_sleep=0.2)
        pdf_dir = output_root / "pdfs"
        if download_oa_pdfs:
            for work in works.values():
                download_pdf(work, pdf_dir, timeout=20.0, max_bytes=80 * 1024 * 1024)
                time.sleep(0.2)
        write_manifest(works, output)
        resolved = [
            {
                "depth": work.depth,
                "title": work.title,
                "doi": work.doi,
                "year": work.year,
                "venue": work.venue,
                "pdf_path": work.pdf_path,
                "status": work.status,
            }
            for work in sorted(works.values(), key=lambda item: (item.depth, item.title or item.key))
        ]
        return {
            "paper_title": paper_title,
            "works": len(resolved),
            "downloaded_pdfs": sum(1 for item in resolved if item["pdf_path"]),
            "manifest_jsonl": str(output.with_suffix(".jsonl")),
            "manifest_markdown": str(output.with_suffix(".md")),
            "results": resolved[:50],
        }

    @server.tool()
    def research_status() -> dict[str, Any]:
        """Show local research services, index identity, and corpus counts."""
        return api.get("/v1/health")

    return server


_mcp = None


def get_mcp():
    global _mcp
    if _mcp is None:
        _mcp = create_mcp()
    return _mcp


def main() -> None:
    get_mcp().run(transport="streamable-http")


if __name__ == "__main__":
    main()
