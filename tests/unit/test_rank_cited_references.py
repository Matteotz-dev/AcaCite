from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[2] / "scripts" / "rank_cited_references.py"
SPEC = spec_from_file_location("rank_cited_references", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reference_key_prefers_doi_and_normalizes_text():
    assert MODULE.reference_key("Paper doi:10.1000/ABC.1.") == "doi:10.1000/abc.1"
    assert MODULE.reference_key("A. Author, A useful SGS closure, 2024.") == (
        "text:author useful sgs closure 2024"
    )


def test_rank_rewards_shared_and_topic_relevant_references():
    candidate = MODULE.Candidate(
        "Clark et al., nonlinear gradient subgrid stress model, 1979.",
        ["paper-one", "paper-two"],
    )
    MODULE.rank(candidate, MODULE.DEFAULT_TOPIC)
    assert candidate.recommendation == "include"
    assert candidate.score >= 45
    assert "cited by both" in candidate.reason


def test_reference_classifier_rejects_prose_from_post_reference_appendices():
    assert MODULE.looks_like_reference(
        "Meneveau, Charles & Katz, Joseph 2000 Scale-invariance in turbulence."
    )
    assert not MODULE.looks_like_reference(
        "The first term corresponds to the Clark model (Jakhar et al. 2024)."
    )
