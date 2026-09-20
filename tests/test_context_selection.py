"""Tests for jevrag.primitives.context_selection — all offline.

Includes a structural acceptance test: the primitive module may import from
jevrag.decision and stdlib only — never jevrag.eval, the other primitives,
benchmarks, or baselines. That is the third-primitive generalization claim as
an AST assertion, not just a report sentence.
"""

import ast
from pathlib import Path

import pytest

from jevrag.decision import Decision, StubDecision
from jevrag.primitives.context_selection import (
    COST_SCOPE_CALL_SHARED,
    COST_SCOPE_PER_PASSAGE,
    POLICY_CONTEXT_SELECTION,
    PassageState,
    batched_question,
    batched_state,
    decide_passage,
    decide_query,
    decide_query_batched,
    make_selection_record,
    select_for,
    selection_question,
)


def _candidates(n: int = 3, query_id: str = "q1") -> list[dict]:
    return [
        {"query": "Who directed it?", "query_id": query_id,
         "passage_id": f"p{i}", "passage_title": f"Title {i}",
         "passage": f"Passage body {i} about the film."}
        for i in range(n)
    ]


class CountingStub(StubDecision):
    """Records every (state, questions) pair the primitive hands the backend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[tuple] = []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        return super().ask(state, questions)


def test_question_is_single_boolean():
    q = selection_question()
    assert q.kind == "boolean"
    assert q.name == "relevant"
    assert "QUERY" in q.instructions and "PASSAGE" in q.instructions
    assert set(q.criteria) == {"true", "false"}


def test_decide_passage_one_shot_passthrough():
    backend = CountingStub(confidences={"relevant": 0.72})
    state = PassageState(query="Who directed it?", passage="A film.",
                         query_id="q1", passage_id="p1")
    res = decide_passage(state, backend)
    assert len(backend.calls) == 1  # one-shot: exactly one backend call
    assert res.confidence == pytest.approx(0.72)
    assert res.result == pytest.approx(0.72)  # Noul: probability is the signal


def test_select_for_threshold():
    assert select_for(0.7) is True
    assert select_for(0.3) is False
    assert select_for(0.5, threshold=0.9) is False


def test_make_selection_record_honest_shape():
    rec = make_selection_record(query_id="q1", passage_id="p1",
                                confidence=0.8, selected=True, latency_ms=90.0,
                                input_tokens=120)
    assert "rounds_used" not in rec  # one-shot: no loop fields
    assert rec["policy"] == POLICY_CONTEXT_SELECTION
    assert rec["cost_scope"] == COST_SCOPE_PER_PASSAGE
    assert rec["selected"] is True


def test_record_rejects_non_bool_selected():
    with pytest.raises(ValueError):
        make_selection_record(query_id="q1", passage_id="p1", confidence=0.8,
                              selected="yes", latency_ms=0.0, input_tokens=0)


def test_record_rejects_non_probability_confidence():
    """The harness guard (calibration.py) catches out-of-range scores; the
    seam guard must fail earlier, with its own message, rather than clipping."""
    with pytest.raises(ValueError):
        make_selection_record(query_id="q1", passage_id="p1", confidence=12.5,
                              selected=True, latency_ms=0.0, input_tokens=0)


def test_decide_query_asks_each_candidate_once_independently():
    backend = CountingStub(confidences={"relevant": 0.9})
    records, trace = decide_query(_candidates(4), backend)
    assert len(backend.calls) == 4 and len(records) == 4 and len(trace) == 4
    assert [r["passage_id"] for r in records] == ["p0", "p1", "p2", "p3"]
    assert all(r["selected"] is True for r in records)
    # Independence: each state carries exactly one passage — its own.
    for cand, (state, _) in zip(_candidates(4), backend.calls):
        assert state["passage"] == cand["passage"]
        assert state["passage_id"] == cand["passage_id"]
        assert "passages" not in state


def test_decide_query_requires_candidate_fields():
    bad = [{"query": "q", "query_id": "q1", "passage": "p"}]  # no passage_id
    with pytest.raises(KeyError):
        decide_query(bad, StubDecision())


def test_state_truncates_long_fields():
    state = PassageState(query="q" * 5000, passage="p" * 5000)
    rendered = state.to_jev_state(max_chars=100)
    assert len(rendered["query"]) == 101  # 100 chars + the ellipsis marker
    assert len(rendered["passage"]) == 101
    assert rendered["passage"].endswith("…")


def test_protocol_conformance():
    assert isinstance(StubDecision(), Decision)


def test_confidence_and_correct_feed_unchanged_harness():
    """Item 4 in offline form: (confidence, correct) from this primitive is
    accepted by the untouched calibration harness — no adaptation layer."""
    from jevrag.eval.calibration import calibration_summary

    backend = CountingStub(confidences={"relevant": 0.6})
    records, _ = decide_query(_candidates(6), backend)
    correct = [1, 1, 0, 0, 1, 0]
    summary = calibration_summary([r["confidence"] for r in records], correct)
    assert summary["n"] == 6
    assert 0.0 <= summary["ece"] <= 1.0


def test_batched_question_names_carry_passage_id():
    q = batched_question("p3")
    assert q.kind == "boolean"
    assert q.name == "relevant:p3"
    assert "'p3'" in q.instructions


def test_batched_state_lists_all_passages_under_one_query():
    state = batched_state("Who directed it?", _candidates(2), query_id="q1")
    assert state["query_id"] == "q1"
    assert [p["id"] for p in state["passages"]] == ["p0", "p1"]
    assert state["passages"][1]["title"] == "Title 1"


def test_batched_is_one_call_for_the_whole_set():
    """The set-level shape fits the existing multi-question ask() with no
    protocol extension: one state, one question per passage."""
    backend = CountingStub(confidences={"relevant:p0": 0.9, "relevant:p1": 0.2,
                                        "relevant:p2": 0.55})
    records, trace = decide_query_batched(_candidates(3), backend)
    assert len(backend.calls) == 1
    state, questions = backend.calls[0]
    assert [q.name for q in questions] == ["relevant:p0", "relevant:p1",
                                           "relevant:p2"]
    assert state["passages"][0]["id"] == "p0"
    assert [r["confidence"] for r in records] == pytest.approx([0.9, 0.2, 0.55])
    assert [r["selected"] for r in records] == [True, False, True]
    assert trace["n"] == 3 and trace["query_id"] == "q1"


def test_batched_cost_is_marked_call_shared():
    """One call's latency/tokens, shared by every record it produced — tagged
    so nobody sums them into a false per-passage total."""
    backend = CountingStub(confidences={"relevant:p0": 0.9, "relevant:p1": 0.2})
    records, trace = decide_query_batched(_candidates(2), backend)
    assert {r["cost_scope"] for r in records} == {COST_SCOPE_CALL_SHARED}
    assert len({r["latency_ms"] for r in records}) == 1  # one shared call cost
    assert trace["call_latency_ms"] == records[0]["latency_ms"]
    assert trace["call_input_tokens"] == records[0]["input_tokens"]


def test_batched_rejects_mixed_query_ids():
    mixed = _candidates(1, "q1") + _candidates(1, "q2")
    with pytest.raises(ValueError):
        decide_query_batched(mixed, StubDecision())


def test_batched_empty_returns_empty():
    records, trace = decide_query_batched([], StubDecision())
    assert records == [] and trace["n"] == 0


def test_batched_missing_confidence_raises():
    """A backend that silently drops a question must fail loudly, not produce
    a fabricated or NaN confidence for that passage."""

    class DroppingStub(StubDecision):
        def ask(self, state, questions):
            res = super().ask(state, questions)
            res.confidence.pop("relevant:p1")
            return res

    with pytest.raises(ValueError):
        decide_query_batched(_candidates(2), DroppingStub())


def test_acceptance_no_protected_imports():
    """Packet 11 Item 3, structural form: context_selection.py may import from
    jevrag.decision and stdlib only — never jevrag.eval, the other
    primitives, benchmarks, or baselines. If this test fails, the third
    generalization claim failed with it."""
    path = Path(__file__).resolve().parent.parent / "jevrag" / "primitives" \
        / "context_selection.py"
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
              or m in ("jevrag.primitives.sufficiency",
                       "jevrag.primitives.chunk_boundary")]
    assert not banned, f"primitive reaches outside decision.py: {banned}"
    assert any(m == "jevrag.decision" for m in imports)

