from uuid import UUID

from app.retrieval.context import pack_context
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.qdrant_store import VectorHit
from app.retrieval.service import looks_lexical
from app.retrieval.reranker import CrossEncoderReranker


def _hit(value: int, score: float, document: str = "doc"):
    chunk_id = UUID(int=value)
    return VectorHit(chunk_id, score, {
        "chunk_id": str(chunk_id), "document_id": document, "text": "evidence",
    })


def test_rrf_preserves_component_ranks_and_deduplicates():
    fused = reciprocal_rank_fusion({
        "dense": [_hit(1, .9), _hit(2, .8)],
        "sparse": [_hit(2, 12), _hit(1, 10)],
    })
    assert len(fused) == 2
    assert fused[0].component_ranks == {"dense": 1, "sparse": 2}
    assert fused[1].component_ranks == {"dense": 2, "sparse": 1}


def test_context_budget_and_source_dominance():
    candidates = reciprocal_rank_fusion({"dense": [_hit(i, 1 / i) for i in range(1, 7)]})
    packed = pack_context(candidates, max_chunks=6, token_budget=100)
    assert len(packed) == 3


def test_lexical_query_heuristics():
    assert looks_lexical("fluxCorrector_X9")
    assert looks_lexical("src/solver.C")
    assert not looks_lexical("how does the method work")


def test_reranker_reduces_batch_then_falls_back_to_cpu(monkeypatch):
    candidates = reciprocal_rank_fusion({"dense": [_hit(1, 1), _hit(2, .5)]})
    devices = []

    class Model:
        def __init__(self, device): self.device = device
        def predict(self, pairs, batch_size):
            if self.device == "cuda": raise RuntimeError("CUDA out of memory")
            return [0.2, 0.9]

    reranker = CrossEncoderReranker("fixture", device="cuda", batch_size=2)
    def load(device):
        devices.append(device)
        reranker._model = Model(device)
    monkeypatch.setattr(reranker, "_load", load)
    ranked = reranker.rerank("query", candidates)
    assert devices == ["cuda", "cpu"]
    assert ranked[0].chunk_id == UUID(int=2)
