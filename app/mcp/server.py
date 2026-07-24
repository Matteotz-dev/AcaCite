"""Retrieval-only Continue-facing tools over the stable local HTTP API."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

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


def create_mcp(client: ResearchAPIClient | None = None) -> FastMCP:
    settings = get_settings()
    api = client or ResearchAPIClient()
    server = FastMCP(
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
    def research_status() -> dict[str, Any]:
        """Show local research services, index identity, and corpus counts."""
        return api.get("/v1/health")

    return server


mcp = create_mcp()


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
