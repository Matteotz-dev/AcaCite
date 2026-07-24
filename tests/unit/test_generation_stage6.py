import json
from urllib.error import URLError
from uuid import uuid4

import pytest

from app.generation.ollama import OllamaAdapter, OllamaUnavailable
from app.generation.service import format_evidence, has_uncited_claim_warning
from app.retrieval.fusion import Candidate


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_ollama_adapter_sends_explicit_model_and_nonstreaming(monkeypatch):
    observed = {}

    def fake_open(request, timeout):
        observed.update(json.loads(request.data))
        observed["timeout"] = timeout
        return Response({"response": "grounded", "model": "devstral-small-2:24b"})

    monkeypatch.setattr("app.generation.ollama.urlopen", fake_open)
    result = OllamaAdapter("http://localhost:11434/", 9).generate(
        model="devstral-small-2:24b", system="system", prompt="question", temperature=0.2
    )
    assert result.text == "grounded"
    assert observed["model"] == "devstral-small-2:24b"
    assert observed["stream"] is False
    assert observed["options"]["temperature"] == 0.2
    assert observed["timeout"] == 9


def test_ollama_adapter_has_clear_unavailable_error(monkeypatch):
    monkeypatch.setattr("app.generation.ollama.urlopen",
                        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")))
    with pytest.raises(OllamaUnavailable, match="unavailable"):
        OllamaAdapter("http://localhost:11434", 1).generate(
            model="qwen", system="s", prompt="p", temperature=0
        )


def test_context_ids_are_server_owned_and_warning_detects_uncited_prose():
    chunk_id = uuid4()
    evidence = format_evidence([Candidate(chunk_id, {"text": "supported text", "title": "T"})])
    assert evidence.startswith(f"[SRC:{chunk_id}]")
    assert "supported text" in evidence
    assert has_uncited_claim_warning(
        "This is a substantive factual sentence with no citation attached to it.", set()
    )
