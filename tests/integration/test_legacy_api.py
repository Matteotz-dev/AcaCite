"""Compatibility checks for the original Cognee API surface.

These tests never ingest, recall, forget, or otherwise mutate Cognee state.
"""

from fastapi.testclient import TestClient

import server


client = TestClient(server.app)


def test_health_contract_is_preserved() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["memory_root"] == str(server.shared_memory.SHARED_ROOT)
    assert body["extraction_model"] == "qwen3.6:27b"


def test_retrieve_contract_without_accessing_durable_cognee(monkeypatch) -> None:
    calls = []

    async def fake_search(query, *, query_type, datasets, top_k):
        calls.append(
            {
                "query": query,
                "query_type": query_type,
                "datasets": datasets,
                "top_k": top_k,
            }
        )
        return [{"text": "mocked source chunk", "score": 0.9}]

    monkeypatch.setattr(server.cognee, "search", fake_search)

    response = client.post(
        "/retrieve",
        json={"query": "gradient model", "datasets": ["papers"], "limit": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "gradient model",
        "results": [{"text": "mocked source chunk", "score": 0.9}],
    }
    assert calls == [
        {
            "query": "gradient model",
            "query_type": server.SearchType.CHUNKS,
            "datasets": ["papers"],
            "top_k": 3,
        }
    ]
