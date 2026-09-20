"""Tests for jevrag.primitives.sufficiency — no network, no API key needed."""

import pytest

from jevrag.decision import DecisionResult, StubDecision
from jevrag.primitives.sufficiency import (
    HANDOFF_FIELDS,
    POLICY_SUFFICIENCY,
    SufficiencyState,
    extractive_answer,
    make_handoff_record,
    run_question,
    run_sufficiency,
    run_sufficiency_with_trace,
    sufficiency_question,
    validate_handoff_record,
    write_records_jsonl,
)

DOCS = [
    {"title": f"Doc{i}", "text": f"This is document {i}. It has two sentences."}
    for i in range(15)
]


def retrieve_fn(question, n_docs):
    assert isinstance(question, str)
    return DOCS[:n_docs]


class ScriptedDecision:
    """Decision backend returning a scripted confidence per call."""

    backend_name = "scripted"

    def __init__(self, confidences):
        self.confidences = list(confidences)
        self.calls = []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        conf = self.confidences[min(len(self.calls) - 1, len(self.confidences) - 1)]
        return DecisionResult(
            result=conf,
            confidence=conf,
            metadata={
                "backend": self.backend_name,
                "latency_ms": 100.0,
                "input_tokens": 200,
                "cost_usd": 0.0,
            },
        )


def answer_fn(question, evidence):
    return "yes"


def test_sufficiency_question_is_boolean():
    q = sufficiency_question()
    assert q.kind == "boolean"
    assert q.name == "sufficient"
    assert q.instructions


def test_stops_early_above_threshold():
    backend = ScriptedDecision([0.3, 0.85, 0.9])
    record, trace = run_sufficiency_with_trace(
        question_id="q1", question="Q?", gold="yes", qtype="comparison",
        decision=backend, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
        threshold=0.7, max_rounds=3, per_round_k=5,
    )
    assert record["rounds_used"] == 2
    assert record["confidence"] == pytest.approx(0.85)
    assert record["prediction"] == "yes"
    assert record["latency_ms"] >= 0.0  # wall-clock for the whole question
    assert record["input_tokens"] == 400
    assert len(trace) == 2
    assert trace[0]["n_docs"] == 5
    assert trace[1]["n_docs"] == 10
    # Per-round Jev latency/tokens live in the trace, not the record.
    assert sum(t["latency_ms"] for t in trace) == pytest.approx(200.0)
    assert sum(t["input_tokens"] for t in trace) == 400
    # Retrieval is cumulative top-n.
    assert [c[0]["evidence"] for c in backend.calls][1].__len__() == 10


def test_cap_answers_on_last_round_anyway():
    backend = ScriptedDecision([0.1, 0.2, 0.3])
    record, trace = run_sufficiency_with_trace(
        question_id="q1", question="Q?", gold="yes", qtype="bridge",
        decision=backend, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
        threshold=0.7, max_rounds=3, per_round_k=5,
    )
    assert record["rounds_used"] == 3
    assert record["confidence"] == pytest.approx(0.3)
    assert record["prediction"] == "yes"  # still answers
    assert len(trace) == 3


def test_first_round_stop_calls_retrieve_once():
    seen_n = []

    def counting_retrieve(question, n_docs):
        seen_n.append(n_docs)
        return DOCS[:n_docs]

    record = run_sufficiency(
        question_id="q1", question="Q?", gold="yes", qtype="bridge",
        decision=ScriptedDecision([0.95]), retrieve_fn=counting_retrieve,
        answer_fn=answer_fn, threshold=0.7, max_rounds=3, per_round_k=5,
    )
    assert record["rounds_used"] == 1
    assert seen_n == [5]


def test_handoff_record_shape():
    record = run_sufficiency(
        question_id="id-1", question="Q?", gold="g", qtype="comparison",
        decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
    )
    # Agreed core keys present, plus the policy tag.
    assert all(k in record for k in HANDOFF_FIELDS)
    assert record["policy"] == POLICY_SUFFICIENCY
    assert record == {
        "question_id": "id-1",
        "question": "Q?",
        "gold": "g",
        "prediction": "yes",
        "confidence": pytest.approx(0.9),
        "rounds_used": 1,
        "latency_ms": pytest.approx(record["latency_ms"]),  # wall-clock
        "input_tokens": 200,
        "type": "comparison",
        "policy": POLICY_SUFFICIENCY,
    }
    assert record["latency_ms"] >= 0.0
    assert validate_handoff_record(record) is record


def test_validate_rejects_drift_but_allows_extras_and_none_confidence():
    good, _ = run_sufficiency_with_trace(
        question_id="q", question="Q?", gold="g", qtype="bridge",
        decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
    )
    missing = {k: v for k, v in good.items() if k != "confidence"}
    with pytest.raises(ValueError, match="Missing"):
        validate_handoff_record(missing)
    # Unknown extras are allowed: the measurement path ignores them.
    assert validate_handoff_record(dict(good, debug="x"))["debug"] == "x"
    # Gate-free baseline records carry null confidence.
    assert validate_handoff_record(dict(good, confidence=None))["confidence"] is None
    bad_conf = dict(good, confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        validate_handoff_record(bad_conf)


def test_make_handoff_record_coerces_types():
    record = make_handoff_record(
        question_id="q", question="Q?", gold="g", prediction="a",
        confidence=1, rounds_used=2, latency_ms=3, input_tokens=4, qtype="bridge",
    )
    assert isinstance(record["confidence"], float)
    assert isinstance(record["rounds_used"], int)
    assert record["policy"] == POLICY_SUFFICIENCY


def test_threshold_none_disables_gate():
    record, trace = run_sufficiency_with_trace(
        question_id="q", question="Q?", gold="g", qtype="bridge",
        decision=ScriptedDecision([0.99, 0.99, 0.99]),
        retrieve_fn=retrieve_fn, answer_fn=answer_fn,
        threshold=None, max_rounds=3, per_round_k=5,
    )
    assert record["rounds_used"] == 3
    assert len(trace) == 3


def test_run_question_dataset_mapping_and_gate_off():
    q = {"id": "abc", "question": "Q?", "answer": "g", "type": "bridge"}
    gated = run_question(
        q, decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
    )
    assert gated["question_id"] == "abc"
    assert gated["gold"] == "g"  # dataset `answer` -> `gold`
    assert gated["policy"] == POLICY_SUFFICIENCY

    baseline = run_question(
        q, decision=ScriptedDecision([0.99]), retrieve_fn=retrieve_fn,
        answer_fn=answer_fn, max_rounds=3, gate=False,
    )
    assert baseline["rounds_used"] == 3  # never stops early
    assert baseline["policy"] == "fixed_iteration_3"
    assert baseline["confidence"] is None  # no gate, no gate signal


def test_tuple_answer_fn_reports_generation_tokens():
    record = run_sufficiency(
        question_id="q", question="Q?", gold="g", qtype="bridge",
        decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
        answer_fn=lambda q, e: ("yes", {"input_tokens": 50}),
    )
    assert record["prediction"] == "yes"
    assert record["input_tokens"] == 250  # 200 Jev + 50 generation


def test_write_records_jsonl_roundtrip(tmp_path):
    import json

    records = [
        run_sufficiency(
            question_id=f"q{i}", question="Q?", gold="g", qtype="bridge",
            decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
            answer_fn=answer_fn,
        )
        for i in range(2)
    ]
    path = write_records_jsonl(records, tmp_path / "records" / "test.jsonl")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert all(validate_handoff_record(json.loads(line)) for line in lines)

    with pytest.raises(ValueError, match="Missing"):
        write_records_jsonl([{"question_id": "q"}], tmp_path / "bad.jsonl")


def test_extractive_answer_baseline():
    assert extractive_answer("Q?", []) == ""
    assert extractive_answer("Q?", [{"title": "t", "text": ""}]) == ""
    out = extractive_answer("Q?", [{"title": "t", "text": "First bit. Second bit."}])
    assert out == "First bit"


def test_state_truncates_long_docs():
    state = SufficiencyState(
        question="Q?",
        evidence=[{"title": "t", "text": "x" * 5000}],
        round=2,
    )
    rendered = state.to_jev_state(max_chars=100)
    assert len(rendered["evidence"][0]["text"]) <= 101
    assert rendered["round"] == 2


def test_invalid_policy_params_rejected():
    with pytest.raises(ValueError):
        run_sufficiency(
            question_id="q", question="Q?", gold="g", qtype="b",
            decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
            answer_fn=answer_fn, max_rounds=0,
        )
    with pytest.raises(ValueError):
        run_sufficiency(
            question_id="q", question="Q?", gold="g", qtype="b",
            decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
            answer_fn=answer_fn, threshold=2.0,
        )


def test_stub_backend_end_to_end_no_network():
    record = run_sufficiency(
        question_id="q", question="Q?", gold="g", qtype="bridge",
        decision=StubDecision(confidences={"sufficient": 0.9}),
        retrieve_fn=retrieve_fn, answer_fn=answer_fn,
    )
    assert record["rounds_used"] == 1
    assert record["confidence"] == pytest.approx(0.9)


def test_non_string_prediction_coerced():
    record = run_sufficiency(
        question_id="q", question="Q?", gold="g", qtype="bridge",
        decision=ScriptedDecision([0.9]), retrieve_fn=retrieve_fn,
        answer_fn=lambda q, e: 42,
    )
    assert record["prediction"] == "42"


def test_decision_none_runs_gate_free_baseline_shape():
    # fixed_iteration calls with decision=None: retrieval still runs,
    # no backend is asked, confidence is null.
    record, trace = run_sufficiency_with_trace(
        question_id="q", question="Q?", gold="g", qtype="bridge",
        decision=None, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
        threshold=None, max_rounds=3, per_round_k=5,
        policy="fixed_iteration_3",
    )
    assert record["rounds_used"] == 3
    assert record["confidence"] is None
    assert record["policy"] == "fixed_iteration_3"
    assert record["prediction"] != ""
    assert len(trace) == 3
    assert all(t["confidence"] is None for t in trace)
    assert validate_handoff_record(record) is record


def test_decision_none_with_gate_raises():
    with pytest.raises(ValueError, match="needs a decision backend"):
        run_sufficiency(
            question_id="q", question="Q?", gold="g", qtype="bridge",
            decision=None, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
            threshold=0.7,
        )


def test_run_question_gate_off_needs_no_backend():
    q = {"id": "abc", "question": "Q?", "answer": "g", "type": "bridge"}
    baseline = run_question(
        q, decision=None, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
        max_rounds=2, gate=False,
    )
    assert baseline["rounds_used"] == 2
    assert baseline["confidence"] is None
    assert baseline["policy"] == "fixed_iteration_2"
