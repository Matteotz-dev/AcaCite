import json
from collections import Counter
from pathlib import Path

import pytest

from app.evaluation.io import load_jsonl
from app.evaluation.metrics import evaluate_records
from app.evaluation.schema import GoldCase, RetrievalRecord


ROOT = Path(__file__).resolve().parents[2]


def test_versioned_gold_set_has_fixed_60_case_category_contract():
    cases = load_jsonl(ROOT / "evaluation/gold/v1.jsonl", GoldCase, identity="id")
    assert len(cases) == 60
    assert Counter(case.category for case in cases) == {
        "exact_code": 15, "paper_fact": 15, "cross_source": 10,
        "graph_multihop": 10, "insufficient_evidence": 10,
    }
    assert sum(case.must_abstain for case in cases) == 10


def test_all_five_offline_ablations_are_reproducible_and_gate_passes():
    cases = load_jsonl(ROOT / "evaluation/gold/v1.jsonl", GoldCase, identity="id")
    records = [RetrievalRecord.model_validate_json(line) for line in
               (ROOT / "evaluation/fixtures/v1_ablation_records.jsonl").read_text().splitlines()]
    report = evaluate_records(cases, records)
    assert set(report["strategies"]) == {
        "dense", "sparse", "hybrid", "hybrid_rerank", "hybrid_graph",
    }
    selected = report["strategies"]["hybrid_rerank"]
    assert selected["recall_at_5"] > report["strategies"]["dense"]["recall_at_5"]
    assert selected["recall_at_5"] > report["strategies"]["sparse"]["recall_at_5"]
    assert selected["citation_precision"] >= .95
    assert selected["citation_validity"] == 1
    assert selected["correct_version_rate"] >= .95
    assert selected["abstention_accuracy"] >= .9


def test_metric_math_reports_rank_version_citation_and_claim_failures():
    case = GoldCase(
        id="code-001", query="where is symbol", category="exact_code",
        relevant_document_ids=["d"], relevant_chunk_ids=["right"],
        relevant_version_ids=["current"], graded_relevance={"right": 3},
        required_facts=["forty two"],
    )
    run = RetrievalRecord(
        case_id=case.id, strategy="hybrid_rerank",
        hits=[{"chunk_id": "wrong", "document_id": "d", "document_version_id": "old"},
              {"chunk_id": "right", "document_id": "d", "document_version_id": "old"}],
        answer="forty two", cited_chunk_ids=["right"], invalid_citation_ids=["invented"],
        abstained=False, supported_claims=1, unsupported_claims=1,
    )
    metrics = evaluate_records([case], [run])["strategies"]["hybrid_rerank"]
    assert metrics["mrr"] == .5
    assert metrics["correct_version_rate"] == 0
    assert metrics["citation_precision"] == .5
    assert metrics["citation_validity"] == .5
    assert metrics["required_fact_coverage"] == 1
    assert metrics["unsupported_claim_rate"] == .5


def test_gold_schema_rejects_answerable_case_without_evidence(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({
        "schema_version": "1.0", "id": "bad-001", "query": "missing evidence",
        "category": "paper_fact", "must_abstain": False,
    }))
    with pytest.raises(ValueError, match="answerable cases require"):
        load_jsonl(path, GoldCase, identity="id")
