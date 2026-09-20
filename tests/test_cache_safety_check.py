"""Tests for jevrag.primitives.cache_safety_check — all offline.

Structural acceptance: the module may import jevrag.decision + stdlib only.
"""

import ast
from pathlib import Path

import pytest

from jevrag.decision import Decision, StubDecision
from jevrag.primitives.cache_safety_check import (
    POLICY_CACHE_SAFETY_CHECK,
    RISK_QUESTION_NAME,
    SAFE_QUESTION_NAME,
    CacheHitState,
    decide_hit,
    decide_hit_single,
    make_safety_record,
    risk_question,
    safe_question,
    verdict_for,
)


def _state(**over) -> CacheHitState:
    kw = dict(match_similarity=0.93, serve_threshold=0.85, entry_age_hours=2.0,
              ttl_hours=24.0, regeneration_cost_tokens=3000,
              prior_successful_serves=12, hit_id="h1")
    kw.update(over)
    return CacheHitState(**kw)


class CountingStub(StubDecision):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[tuple] = []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        return super().ask(state, questions)


def test_questions_are_two_booleans_one_call():
    assert safe_question().kind == "boolean"
    assert safe_question().name == SAFE_QUESTION_NAME
    assert risk_question().kind == "boolean"
    assert risk_question().name == RISK_QUESTION_NAME
    backend = CountingStub(confidences={"safe_to_serve": 0.8,
                                        "mismatch_risk": 0.2})
    res = decide_hit(_state(), backend)
    assert len(backend.calls) == 1  # one call, both questions in it
    assert len(backend.calls[0][1]) == 2
    assert res.confidence == pytest.approx({"safe_to_serve": 0.8,
                                            "mismatch_risk": 0.2})


def test_state_is_text_free():
    rendered = _state().to_jev_state()
    assert set(rendered) == {"match_similarity",
                             "similarity_margin_above_threshold",
                             "age_fraction_of_ttl", "ttl_expired",
                             "regeneration_cost_tokens",
                             "prior_successful_serves", "hit_id"}
    assert rendered["similarity_margin_above_threshold"] == pytest.approx(0.08)
    assert rendered["age_fraction_of_ttl"] == pytest.approx(round(2.0 / 24.0, 4))
    assert rendered["ttl_expired"] is False
    assert _state(entry_age_hours=30.0).to_jev_state()["ttl_expired"] is True
    non_id = {k: v for k, v in rendered.items() if k != "hit_id"}
    assert all(isinstance(v, (int, float, bool)) for v in non_id.values())


def test_state_rejects_nothing_but_clamps_garbage():
    rendered = _state(entry_age_hours=-5.0).to_jev_state()
    assert rendered["age_fraction_of_ttl"] == pytest.approx(0.0)


def test_verdict_needs_both_endorsement_and_no_risk():
    assert verdict_for(0.9, 0.1) == "serve"
    assert verdict_for(0.9, 0.7) == "regenerate"  # risk flagged
    assert verdict_for(0.3, 0.1) == "regenerate"  # no endorsement
    assert verdict_for(0.3, 0.8) == "regenerate"  # both bad
    assert verdict_for(0.5, 0.5 - 1e-9) == "serve"  # boundary uses >= / <
    assert verdict_for(0.9, 0.1, risk_ceiling=0.05) == "regenerate"


def test_record_guards_the_seam():
    rec = make_safety_record(hit_id="h1", safe_confidence=0.8,
                             risk_confidence=0.2, verdict="serve",
                             latency_ms=50.0, input_tokens=120)
    assert rec["policy"] == POLICY_CACHE_SAFETY_CHECK
    with pytest.raises(ValueError):
        make_safety_record(hit_id="h", safe_confidence=0.8,
                           risk_confidence=0.2, verdict="maybe",
                           latency_ms=0.0, input_tokens=0)
    with pytest.raises(ValueError):
        make_safety_record(hit_id="h", safe_confidence=1.4,
                           risk_confidence=0.2, verdict="serve",
                           latency_ms=0.0, input_tokens=0)


def test_single_ablation_is_one_question_one_call():
    backend = CountingStub(confidences={"safe_to_serve": 0.6})
    res = decide_hit_single(_state(), backend)
    assert len(backend.calls) == 1
    assert len(backend.calls[0][1]) == 1
    assert res.confidence == pytest.approx(0.6)


def test_protocol_conformance():
    assert isinstance(StubDecision(), Decision)


def test_acceptance_no_protected_imports():
    path = Path(__file__).resolve().parent.parent / "jevrag" / "primitives" \
        / "cache_safety_check.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
    banned = [m for m in imports
              if m.startswith(("jevrag.eval", "jevrag.benchmarks",
                               "jevrag.baselines", "jevrag.adapters"))
              or m in ("jevrag.primitives.cache_trust",
                       "jevrag.primitives.sufficiency",
                       "jevrag.primitives.chunk_boundary",
                       "jevrag.primitives.context_selection")]
    assert not banned, f"module reaches outside decision.py: {banned}"
    assert any(m == "jevrag.decision" for m in imports)
