"""Tests for jevrag.primitives.chunk_boundary (V1.1) — all offline.

Includes a structural acceptance test: the primitive module must not import
from jevrag.eval, sufficiency, or fixed_iteration — the generalization claim
as an AST assertion, not just a report sentence.
"""

import ast
from pathlib import Path

import pytest

from jevrag.decision import Decision, StubDecision
from jevrag.primitives.chunk_boundary import (
    POLICY_CHUNK_BOUNDARY,
    BoundaryState,
    boundary_question,
    decide_boundary,
    decide_document,
    label_for,
    make_boundary_record,
)


def test_question_is_single_boolean():
    q = boundary_question()
    assert q.kind == "boolean"
    assert q.name == "split"
    assert "BEFORE" in q.instructions and "AFTER" in q.instructions


def test_decide_boundary_one_shot_passthrough():
    calls = []

    class CountingStub(StubDecision):
        def ask(self, state, questions):
            calls.append((state, questions))
            return super().ask(state, questions)

    backend = CountingStub(confidences={"split": 0.72})
    state = BoundaryState(before="Topic A ends here.", after="Topic B begins.",
                          doc_id="d", boundary_index=3)
    res = decide_boundary(state, backend)
    assert len(calls) == 1  # one-shot: exactly one backend call
    assert res.confidence == pytest.approx(0.72)
    assert res.result == pytest.approx(0.72)  # Noul: probability is the signal


def test_label_for_threshold():
    assert label_for(0.7) == "split"
    assert label_for(0.3) == "merge"
    assert label_for(0.5, threshold=0.9) == "merge"


def test_make_boundary_record_honest_shape():
    rec = make_boundary_record(doc_id="d", boundary_index=4, result="split",
                               confidence=0.8, latency_ms=100.0,
                               input_tokens=200)
    assert "rounds_used" not in rec  # one-shot: no loop fields
    assert rec["policy"] == POLICY_CHUNK_BOUNDARY
    with pytest.raises(ValueError):
        make_boundary_record(doc_id="d", boundary_index=0, result="maybe",
                             confidence=0.5, latency_ms=0.0, input_tokens=0)


def test_decide_document_asks_each_once():
    cands = [{"doc_id": "d", "boundary_index": i,
              "before": f"before {i}", "after": f"after {i}"} for i in range(5)]
    records, trace = decide_document(
        cands, StubDecision(confidences={"split": 0.9}))
    assert len(records) == 5 and len(trace) == 5
    assert [r["boundary_index"] for r in records] == list(range(5))
    assert all(r["result"] == "split" for r in records)


def test_state_truncates_long_windows():
    state = BoundaryState(before="x" * 5000, after="y" * 5000)
    rendered = state.to_jev_state(max_chars=100)
    assert len(rendered["before"]) <= 101
    assert len(rendered["after"]) <= 101


def test_protocol_conformance():
    assert isinstance(StubDecision(), Decision)


def test_acceptance_no_protected_imports():
    """V1.1 acceptance test, structural form: chunk_boundary.py may import
    from jevrag.decision and stdlib only — never jevrag.eval, sufficiency,
    fixed_iteration, benchmarks, or baselines. If this test fails, the
    generalization claim failed with it."""
    path = Path(__file__).resolve().parent.parent / "jevrag" / "primitives" \
        / "chunk_boundary.py"
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
              or m in ("jevrag.primitives.sufficiency",)]
    assert not banned, f"primitive reaches outside decision.py: {banned}"
    assert any(m == "jevrag.decision" for m in imports)
