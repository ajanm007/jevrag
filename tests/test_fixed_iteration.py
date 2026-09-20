"""Tests for jevrag.baselines.fixed_iteration — the thing sufficiency has to beat."""

import pytest

from jevrag.baselines.fixed_iteration import (
    run_fixed_iteration,
    run_fixed_iteration_dataset,
)
from jevrag.decision import StubDecision

DOCS = [{"title": f"D{i}", "text": f"Text {i}."} for i in range(20)]


def retrieve_fn(question, n_docs):
    return DOCS[:n_docs]


def answer_fn(question, evidence):
    return "an answer"


Q = {"id": "q1", "question": "Q?", "answer": "an answer", "type": "bridge"}


def test_always_runs_exactly_n_rounds():
    rec = run_fixed_iteration(
        Q, n_rounds=3, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
        decision=StubDecision(),
    )
    assert rec["rounds_used"] == 3
    assert rec["policy"] == "fixed_iteration_3"
    assert rec["prediction"] == "an answer"


def test_no_gate_signal_is_explicit_null():
    rec = run_fixed_iteration(
        Q, n_rounds=2, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
    )
    assert rec["confidence"] is None


def test_uses_same_stack_as_gated_run():
    # Same retrieve_fn/answer_fn wired in -> the baseline answer is exactly what
    # the gated path would produce after its final round.
    rec = run_fixed_iteration(
        Q, n_rounds=1, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
    )
    assert rec["prediction"] == answer_fn(Q["question"], retrieve_fn(Q["question"], 5))


def test_dataset_wrapper_preserves_order_and_fields():
    qs = [Q, dict(Q, id="q2", type="comparison")]
    recs = run_fixed_iteration_dataset(qs, n_rounds=2, retrieve_fn=retrieve_fn)
    assert [r["question_id"] for r in recs] == ["q1", "q2"]
    assert recs[1]["type"] == "comparison"


def test_n_rounds_validated():
    with pytest.raises(ValueError):
        run_fixed_iteration(Q, n_rounds=0, retrieve_fn=retrieve_fn)
