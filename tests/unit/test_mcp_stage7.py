from __future__ import annotations

import asyncio

import pytest

from app.mcp.client import ResearchAPIClient, ResearchAPIError
from app.mcp.server import _compact_search_response, create_mcp


class FixtureClient:
    def __init__(self):
        self.calls = []

    def get(self, path):
        self.calls.append(("GET", path, None))
        return {"path": path}

    def post(self, path, payload):
        self.calls.append(("POST", path, payload))
        return {"path": path, "payload": payload}


def test_mcp_exposes_retrieval_only_tools():
    tools = asyncio.run(create_mcp(FixtureClient()).list_tools())
    assert {tool.name for tool in tools} == {
        "research_search", "research_remember", "research_open_source",
        "research_related", "research_status",
    }
    schemas = {tool.name: tool.inputSchema for tool in tools}
    assert "query" in schemas["research_search"]["required"]
    assert "research_answer" not in schemas
    assert set(schemas["research_remember"]["required"]) == {
        "content", "source_uri", "dataset"
    }


def test_search_response_drops_index_metadata_but_keeps_citable_evidence():
    compact = _compact_search_response({
        "query": "gradient model",
        "results": [{
            "chunk_id": "chunk-1",
            "payload": {
                "title": "Paper", "canonical_uri": "file:///paper.pdf",
                "page_start": 3, "page_end": 4, "heading_path": ["Results"],
                "text": "Relevant evidence", "document_version_id": "omit-me",
            },
            "reranker_score": 2.5, "component_scores": {"dense": 0.9},
        }],
        "trace": {"total_ms": 12.0, "dense_count": 40},
    })
    assert compact["results"] == [{
        "citation_id": "chunk-1", "title": "Paper",
        "source": "file:///paper.pdf", "page_start": 3, "page_end": 4,
        "heading": ["Results"], "evidence": "Relevant evidence", "score": 2.5,
    }]
    assert compact["retrieval_ms"] == 12.0
    assert "document_version_id" not in str(compact)


def test_tool_translation_preserves_source_id():
    client = FixtureClient()
    server = create_mcp(client)
    asyncio.run(server.call_tool("research_open_source", {
        "chunk_id": "8a70f5b5-bfe6-4ff7-a42d-b02db98befa8"
    }))
    assert client.calls[0][1] == "/v1/sources/8a70f5b5-bfe6-4ff7-a42d-b02db98befa8"


def test_api_client_surfaces_ollama_down_without_hiding_status(monkeypatch):
    class Response:
        status_code = 503
        text = "unavailable"
        request = None
        def raise_for_status(self):
            import httpx
            raise httpx.HTTPStatusError("down", request=self.request, response=self)
        def json(self):
            return {"detail": {"code": "ollama_unavailable"}}

    monkeypatch.setattr("httpx.request", lambda *args, **kwargs: Response())
    with pytest.raises(ResearchAPIError, match="ollama_unavailable"):
        ResearchAPIClient().post("/v1/answer", {"query": "x"})
