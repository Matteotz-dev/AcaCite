"""Small non-streaming Ollama HTTP adapter with explicit request-scoped models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaResult:
    text: str
    model: str


class OllamaAdapter:
    def __init__(self, base_url: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, *, model: str, system: str, prompt: str,
                 temperature: float) -> OllamaResult:
        payload = json.dumps({
            "model": model, "system": system, "prompt": prompt, "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": {"temperature": temperature},
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OllamaUnavailable(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable(f"Ollama is unavailable: {exc}") from exc
        text = body.get("response")
        if not isinstance(text, str):
            raise OllamaUnavailable("Ollama returned no textual response")
        return OllamaResult(text=text.strip(), model=str(body.get("model") or model))
