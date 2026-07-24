"""Small HTTP adapter keeping MCP separate from RAG implementation details."""

from __future__ import annotations

from typing import Any

import httpx


class ResearchAPIError(RuntimeError):
    pass


class ResearchAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = httpx.request(
                method, f"{self.base_url}{path}", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise ResearchAPIError("research API returned a non-object response")
            return value
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            try:
                detail = str(exc.response.json().get("detail", detail))
            except (ValueError, AttributeError):
                pass
            raise ResearchAPIError(
                f"research API returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ResearchAPIError(f"research API unavailable: {exc}") from exc
