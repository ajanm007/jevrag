"""Tests for jevrag.primitives.cache_trust — all offline.

Structural acceptance test included (same pattern as answer-abstain):
the primitive may import jevrag.decision + stdlib only.
"""

import ast
from pathlib import Path

import pytest

from jevrag.decision import DecisionResult, StubDecision
from jevrag.primitives.cache_trust import (
    POLICY_CACHE_TRUST,
    SERVE_QUESTION_NAME,
    STALENESS_QUESTION_NAME,
    CacheTrustState,
    action_for,
    cache_trust_questions,
    check_cache,
    check_cache_trust,
    extract_decision,
    make_cache_record,
    should_serve,
)


def make_state(**overrides):
    base = dict(similarity_score=0.85, similarity_threshold=0.70,
                cache_age_seconds=120.0, cache_ttl_seconds=3600,
                entry_scoped_to_conversation=True,
                query_length_chars=60, answer_length_chars=12)
    base.update(overrides)
    return CacheTrustState.make(**base)


def test_two_questions_one_call():
    calls = []

    class CountingStub(StubDecision):
        def ask(self, state, questions):
            calls.append((state, questions))
            return super().ask(state, questions)

    backend = CountingStub(confidences={SERVE_QUESTION_NAME: 0.8,
                                        STALENESS_QUESTION_NAME: 0.2})
    res = check_cache_trust(make_state(), backend)
    assert len(calls) == 1, "must be a single ask() with both questions"
    assert [q.name for q in calls[0][1]] == [SERVE_QUESTION_NAME,
                                             STALENESS_QUESTION_NAME]
    assert all(q.kind == "boolean" for q in calls[0][1])
    assert isinstance(res.confidence, dict)
    assert res.confidence[SERVE_QUESTION_NAME] == pytest.approx(0.8)
    assert res.confidence[STALENESS_QUESTION_NAME] == pytest.approx(0.2)


def test_state_is_non_text_structural():
    sent = make_state().to_jev_state()
    assert set(sent) == {"similarityScore", "similarityThreshold",
                         "scoreMargin", "cacheAgeSeconds", "cacheTtlSeconds",
                         "entryScopedToConversation", "queryLengthChars",
                         "answerLengthChars"}
    for key, value in sent.items():
        assert isinstance(value, (int, float, bool)) and not isinstance(value, str), \
            f"{key} must be numeric/bool, got {value!r}"
    assert "question" not in sent and "answer" not in sent, \
        "cached content must never travel to the evaluation call"


def test_make_derives_margin():
    st = CacheTrustState.make(similarity_score=0.85, similarity_threshold=0.70,
                              cache_age_seconds=10.0, cache_ttl_seconds=3600,
                              entry_scoped_to_conversation=False,
                              query_length_chars=10, answer_length_chars=5)
    assert st.score_margin == pytest.approx(0.15)


def test_verdict_truth_table():
    assert should_serve({"serve_from_cache": 0.9, "staleness_risk": 0.1})
    assert not should_serve({"serve_from_cache": 0.4, "staleness_risk": 0.1})
    assert not should_serve({"serve_from_cache": 0.9, "staleness_risk": 0.6})
    assert should_serve({"serve_from_cache": 0.5, "staleness_risk": 0.49})
    assert not should_serve({"serve_from_cache": 0.5, "staleness_risk": 0.5})
    assert action_for(None) == "serve", "invalid verdicts fail open"
    assert action_for({"serve_from_cache": 0.9, "staleness_risk": 0.1}) == "serve"
    assert action_for({"serve_from_cache": 0.9, "staleness_risk": 0.9}) == "regenerate"


def test_extract_decision_rejects_bad_shapes():
    good = DecisionResult(result={"a": 1}, confidence={SERVE_QUESTION_NAME: 0.7,
                                                      STALENESS_QUESTION_NAME: 0.3})
    assert extract_decision(good) == {"serve_from_cache": 0.7,
                                      "staleness_risk": 0.3}
    scalar = DecisionResult(result=0.5, confidence=0.5)
    assert extract_decision(scalar) is None
    missing = DecisionResult(result={}, confidence={SERVE_QUESTION_NAME: 0.7})
    assert extract_decision(missing) is None
    wrong_type = DecisionResult(
        result={}, confidence={SERVE_QUESTION_NAME: True,
                               STALENESS_QUESTION_NAME: 0.3})
    assert extract_decision(wrong_type) is None
    nan = DecisionResult(result={}, confidence={SERVE_QUESTION_NAME: float("nan"),
                                                STALENESS_QUESTION_NAME: 0.3})
    assert extract_decision(nan) is None


def test_check_cache_end_to_end_stub():
    rec, trace = check_cache(
        query_id="q", query="Q?", cached_answer="g", state=make_state(),
        decision=StubDecision(confidences={SERVE_QUESTION_NAME: 0.8,
                                           STALENESS_QUESTION_NAME: 0.2}))
    assert rec["action"] == "serve" and not rec["fallback_used"]
    assert rec["confidence"] == pytest.approx(0.8)
    assert rec["staleness_risk"] == pytest.approx(0.2)
    assert trace["input_tokens"] == 0

    rec2, _ = check_cache(
        query_id="q", query="Q?", cached_answer="g", state=make_state(),
        decision=StubDecision(confidences={SERVE_QUESTION_NAME: 0.8,
                                           STALENESS_QUESTION_NAME: 0.9}))
    assert rec2["action"] == "regenerate"


def test_check_cache_fails_open_on_exception():
    class RaisingBackend(StubDecision):
        def ask(self, state, questions):
            raise TimeoutError("jev down")

    rec, trace = check_cache(query_id="q", query="Q?", cached_answer="g",
                             state=make_state(), decision=RaisingBackend())
    assert rec["action"] == "serve" and rec["fallback_used"]
    assert rec["confidence"] is None
    assert trace["fallback"] == "exception"


def test_check_cache_fails_open_on_invalid_response():
    class ScalarBackend(StubDecision):
        def ask(self, state, questions):
            return DecisionResult(result=0.5, confidence=0.5,
                                  metadata={"backend": "weird"})

    rec, trace = check_cache(query_id="q", query="Q?", cached_answer="g",
                             state=make_state(), decision=ScalarBackend())
    assert rec["action"] == "serve" and rec["fallback_used"]
    assert trace["fallback"] == "invalid-response"


def test_make_cache_record_honest_shape():
    rec = make_cache_record(query_id="q", query="Q?", cached_answer="g",
                            confidence=0.8, staleness_risk=0.2, action="serve",
                            latency_ms=10.0, input_tokens=50)
    assert "rounds_used" not in rec
    assert rec["policy"] == POLICY_CACHE_TRUST
    with pytest.raises(ValueError):
        make_cache_record(query_id="q", query="Q?", cached_answer="g",
                          confidence=0.8, staleness_risk=0.2, action="maybe",
                          latency_ms=0.0, input_tokens=0)


def test_questions_carry_original_wording():
    by_name = {q.name: q for q in cache_trust_questions()}
    assert "instead of generating a fresh answer" in \
        by_name[SERVE_QUESTION_NAME].instructions
    assert "outdated or mismatched" in \
        by_name[STALENESS_QUESTION_NAME].instructions


def test_acceptance_no_protected_imports():
    path = Path(__file__).resolve().parent.parent / "jevrag" / "primitives" \
        / "cache_trust.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
    banned = [m for m in imports
              if m.startswith(("jevrag.eval", "jevrag.benchmarks",
                               "jevrag.baselines"))
              or m in ("jevrag.primitives.sufficiency",
                       "jevrag.primitives.chunk_boundary",
                       "jevrag.primitives.answer_abstain",
                       "jevrag.primitives.context_selection")]
    assert not banned, f"primitive reaches outside decision.py: {banned}"
    assert any(m == "jevrag.decision" for m in imports)
