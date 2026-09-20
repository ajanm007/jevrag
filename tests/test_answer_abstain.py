"""Tests for jevrag.primitives.answer_abstain — all offline.

Structural acceptance test included (same pattern as chunk-boundary):
the primitive may import jevrag.decision + stdlib only.
"""

import ast
from pathlib import Path

import pytest

from jevrag.decision import Decision, StubDecision
from jevrag.primitives.answer_abstain import (
    POLICY_ANSWER_ABSTAIN,
    GroundingState,
    action_for,
    check_answer,
    check_grounding,
    grounding_question,
    make_abstain_record,
)


def test_question_is_single_boolean():
    q = grounding_question()
    assert q.kind == "boolean"
    assert q.name == "grounded"
    assert "evidence" in q.instructions.lower()


def test_check_grounding_one_shot_passthrough():
    calls = []

    class CountingStub(StubDecision):
        def ask(self, state, questions):
            calls.append((state, questions))
            return super().ask(state, questions)

    backend = CountingStub(confidences={"grounded": 0.81})
    state = GroundingState(question="Q?", evidence=[{"title": "t", "text": "x"}],
                           prediction="x")
    res = check_grounding(state, backend)
    assert len(calls) == 1
    assert res.confidence == pytest.approx(0.81)
    sent = calls[0][0]
    assert sent["answer"] == "x" and len(sent["evidence"]) == 1


def test_action_for_threshold():
    assert action_for(0.7) == "pass"
    assert action_for(0.3) == "abstain"
    assert action_for(0.5, threshold=0.9) == "abstain"


def test_make_abstain_record_honest_shape():
    rec = make_abstain_record(question_id="q", question="Q?", gold="g",
                              prediction="p", confidence=0.4, action="abstain",
                              latency_ms=10.0, input_tokens=50)
    assert "rounds_used" not in rec
    assert rec["policy"] == POLICY_ANSWER_ABSTAIN
    with pytest.raises(ValueError):
        make_abstain_record(question_id="q", question="Q?", gold="g",
                            prediction="p", confidence=0.4, action="maybe",
                            latency_ms=0.0, input_tokens=0)


def test_check_answer_end_to_end_stub():
    rec, trace = check_answer(
        question_id="q", question="Q?", gold="g",
        evidence=[{"title": "t", "text": "x"}], prediction="x",
        decision=StubDecision(confidences={"grounded": 0.2}))
    assert rec["action"] == "abstain"
    assert rec["confidence"] == pytest.approx(0.2)
    assert trace["input_tokens"] == 0


def test_state_truncates_long_evidence():
    state = GroundingState(question="Q?",
                           evidence=[{"title": "t", "text": "x" * 5000}],
                           prediction="p")
    rendered = state.to_jev_state(max_chars=100)
    assert len(rendered["evidence"][0]["text"]) <= 101


def test_acceptance_no_protected_imports():
    path = Path(__file__).resolve().parent.parent / "jevrag" / "primitives" \
        / "answer_abstain.py"
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
                       "jevrag.primitives.chunk_boundary")]
    assert not banned, f"primitive reaches outside decision.py: {banned}"
    assert any(m == "jevrag.decision" for m in imports)
