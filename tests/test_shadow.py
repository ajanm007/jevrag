"""Tests for jevrag.backends.shadow — all offline, no network, no key.

The one that actually matters: a call through ShadowDecision produces the
identical DecisionResult a direct call to the wrapped backend would have
("shadow" really means "observe," not "modify"). Plus: a real ledger row
per call with the documented schema, the state-size policy, the
write-failure policy (production never breaks), the report script's real
behavior on a synthetic ledger, and the structural acceptance test
(wrapper imports decision.py + stdlib only).
"""

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from jevrag.backends.logprob_decision import LogprobDecision
from jevrag.backends.shadow import (
    MAX_STATE_STR,
    SHADOW_NOTE,
    ShadowDecision,
    summarize_state,
)
from jevrag.decision import Decision, StubDecision, TypedQuestion


def bool_q(name="grounded"):
    return TypedQuestion(
        name=name,
        kind="boolean",
        instructions="Is this answer supported by the evidence?",
    )


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- the one that matters: observe, never modify ---

def test_shadow_returns_identical_result_to_direct_call(tmp_path):
    inner = StubDecision(confidences={"grounded": 0.81})
    shadow = ShadowDecision(inner, tmp_path / "ledger.jsonl")
    state = {"question": "Q?", "evidence": "some text"}
    direct = inner.ask(state, [bool_q()])
    via_shadow = shadow.ask(state, [bool_q()])
    assert via_shadow.result == direct.result
    assert via_shadow.confidence == direct.confidence
    # Metadata identical modulo the clock both calls measure independently.
    assert {k: v for k, v in via_shadow.metadata.items()
            if k != "latency_ms"} == {
                k: v for k, v in direct.metadata.items() if k != "latency_ms"}
    assert via_shadow.metadata["backend"] == "stub"  # inner tag, not the wrapper's


def test_shadow_over_logprob_is_real_signal_zero_cost(tmp_path):
    shadow = ShadowDecision(LogprobDecision(),
                            tmp_path / "ledger.jsonl")
    res = shadow.ask({"mean_logprob": -0.2}, [bool_q()])
    assert res.confidence == pytest.approx(math.exp(-0.2))
    assert res.metadata["cost_usd"] == 0.0


def test_multi_question_result_passes_through_untouched(tmp_path):
    inner = StubDecision(confidences={"a": 0.9, "b": 0.2})
    shadow = ShadowDecision(inner, tmp_path / "ledger.jsonl")
    direct = inner.ask({}, [bool_q("a"), bool_q("b")])
    via = shadow.ask({}, [bool_q("a"), bool_q("b")])
    assert via.result == direct.result
    assert via.confidence == direct.confidence


def test_inner_exception_propagates_no_row_logged(tmp_path):
    class Failing:
        backend_name = "failing"

        def ask(self, state, questions):
            raise RuntimeError("inner blew up")

    ledger = tmp_path / "ledger.jsonl"
    shadow = ShadowDecision(Failing(), ledger)
    with pytest.raises(RuntimeError, match="inner blew up"):
        shadow.ask({}, [bool_q()])
    assert not ledger.exists()  # a failed decision is not an observation


def test_constructor_rejects_non_backend():
    with pytest.raises(TypeError, match="ask\\(state, questions\\)"):
        ShadowDecision(object(), "ledger.jsonl")


def test_constructor_creates_parents_and_fails_fast_on_bad_path(tmp_path):
    nested = tmp_path / "a" / "b" / "ledger.jsonl"
    ShadowDecision(StubDecision(), nested)
    assert nested.parent.is_dir()
    # A path whose parent cannot be created (blocked by a file) is a
    # programming error: loud at construction, not a silent drop later.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    with pytest.raises(OSError):
        ShadowDecision(StubDecision(), blocker / "ledger.jsonl")


# --- the ledger: a real row per call, documented schema ---

def test_ledger_row_schema_and_seq(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    shadow = ShadowDecision(StubDecision(confidences={"grounded": 0.7}),
                            ledger)
    shadow.ask({"question_id": "q1"}, [bool_q()])
    shadow.ask({"question_id": "q2"}, [bool_q()])
    rows = read_rows(ledger)
    assert len(rows) == 2
    assert [r["seq"] for r in rows] == [1, 2]
    r = rows[0]
    assert r["wrapper"] == "shadow"
    assert r["backend"] == "stub"
    assert "ts" in r and r["note"] == SHADOW_NOTE
    assert r["questions"] == [{
        "name": "grounded", "kind": "boolean",
        "instructions": "Is this answer supported by the evidence?",
        "criteria": None,
    }]
    assert r["result"] == pytest.approx(0.7)
    assert r["confidence"] == pytest.approx(0.7)
    assert r["state_summary"] == {"question_id": "q1"}
    assert shadow.calls == 2 and shadow.logged == 2
    assert shadow.errors == 0 and shadow.dropped == 0


def test_ledger_appends_across_instances(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ShadowDecision(StubDecision(), ledger).ask({}, [bool_q()])
    ShadowDecision(StubDecision(), ledger).ask({}, [bool_q()])
    assert len(read_rows(ledger)) == 2  # append, never truncate


def test_ledger_cost_fields_come_from_inner_metadata(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ShadowDecision(StubDecision(), ledger).ask({}, [bool_q()])
    r = read_rows(ledger)[0]
    assert r["input_tokens"] == 0
    assert r["cost_usd"] == 0.0
    assert r["latency_ms"] >= 0.0


# --- state policy: shape kept, bulk dropped ---

def test_state_summary_truncates_evidence_blobs():
    blob = "x" * 5000
    summary = summarize_state({
        "question": "short",
        "evidence": [{"title": "t", "text": blob}],
        "answer": "y",
    })
    assert summary["question"] == "short"
    ev = summary["evidence"]
    assert ev["n"] == 1
    text = ev["preview"][0]["text"]
    assert text["full_len"] == 5000
    assert len(text["truncated_str"]) == MAX_STATE_STR
    assert blob not in json.dumps(summary)  # bulk never reaches the file


def test_state_summary_scalars_verbatim_sequences_capped():
    assert summarize_state(0.5) == 0.5
    assert summarize_state("abc") == "abc"
    assert summarize_state(None) is None
    assert summarize_state({"logprobs": [-0.1] * 10}) == {
        "logprobs": {"n": 10, "preview": [-0.1, -0.1, -0.1]}}


def test_state_summary_non_mapping_states():
    assert summarize_state(["a", "b"])["n"] == 2
    s = summarize_state(object())
    assert "type" in s and "repr" in s


# --- failure policy: observation never breaks production ---

def test_write_failure_returns_result_and_counts_drop(tmp_path):
    ledger_dir = tmp_path / "is_a_directory"
    ledger_dir.mkdir()
    # open(path-as-directory, "a") raises OSError on every write.
    shadow = ShadowDecision(StubDecision(confidences={"grounded": 0.4}),
                            ledger_dir)
    res = shadow.ask({}, [bool_q()])
    assert res.confidence == pytest.approx(0.4)  # production unaffected
    assert shadow.calls == 1
    assert shadow.logged == 0
    assert shadow.errors == 1 and shadow.dropped == 1


# --- housekeeping ---

def test_close_forwards_when_present_and_noops_when_absent(tmp_path):
    closed = []

    class Closable(StubDecision):
        def close(self):
            closed.append(True)

    ShadowDecision(Closable(), tmp_path / "l.jsonl").close()
    assert closed == [True]
    ShadowDecision(StubDecision(), tmp_path / "l.jsonl").close()  # no-op


def test_protocol_conformance(tmp_path):
    assert isinstance(ShadowDecision(StubDecision(), tmp_path / "l.jsonl"),
                      Decision)


def test_answers_primitive_real_question(tmp_path):
    from jevrag.primitives.answer_abstain import (
        GroundingState,
        grounding_question,
    )

    state = GroundingState(
        question="Q?", evidence=[{"title": "t", "text": "x"}], prediction="x"
    ).to_jev_state()
    state["mean_logprob"] = -0.05
    shadow = ShadowDecision(LogprobDecision(), tmp_path / "l.jsonl")
    res = shadow.ask(state, [grounding_question()])
    assert res.confidence == pytest.approx(math.exp(-0.05))
    assert len(read_rows(tmp_path / "l.jsonl")) == 1


def test_acceptance_wrapper_imports_only_decision_and_stdlib():
    path = (Path(__file__).resolve().parent.parent / "jevrag" / "backends"
            / "shadow.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
    banned = [m for m in imports
              if m.startswith("jevrag.")
              and not m.startswith("jevrag.decision")]
    assert not banned, f"wrapper reaches past decision.py: {banned}"
    assert any(m == "jevrag.decision" for m in imports)


# --- the report script, exercised for real on a synthetic ledger ---

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_ledger(tmp_path, confidences):
    ledger = tmp_path / "ledger.jsonl"
    shadow = ShadowDecision(
        StubDecision(confidences={"grounded": 0.5}), ledger)
    for i, c in enumerate(confidences):
        shadow.inner.confidences["grounded"] = c
        shadow.ask({"qid": f"q{i}"}, [bool_q()])
    return ledger


def _run_report(*argv):
    return subprocess.run(
        [sys.executable, "scripts/report_shadow_ledger.py", *argv],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)


def test_report_script_reads_ledger(tmp_path):
    ledger = _make_ledger(tmp_path, [0.9, 0.1, 0.8])
    proc = _run_report("--ledger", str(ledger))
    assert proc.returncode == 0, proc.stderr
    assert "calls observed: 3" in proc.stdout
    assert "stub" in proc.stdout
    assert "grounded" in proc.stdout


def test_report_script_join_and_mismatch_refusal(tmp_path):
    ledger = _make_ledger(tmp_path, [0.9, 0.1, 0.8, 0.4])
    records = tmp_path / "records.jsonl"
    rows = [
        {"confidence": 0.9, "correct": 1, "action": "pass"},
        {"confidence": 0.1, "correct": 0, "action": "abstain"},
        {"confidence": 0.8, "correct": 1, "action": "pass"},
        {"confidence": 0.4, "correct": 0, "action": "abstain"},
    ]
    records.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    proc = _run_report("--ledger", str(ledger), "--records", str(records))
    assert proc.returncode == 0, proc.stderr
    assert "agreement rate: 1.0000 (4/4)" in proc.stdout
    assert "challenger AURC" in proc.stdout

    short = tmp_path / "short.jsonl"
    short.write_text(json.dumps(rows[0]) + "\n")
    proc = _run_report("--ledger", str(ledger), "--records", str(short))
    assert proc.returncode == 2
    assert "positional join refused" in proc.stderr


def test_report_script_refuses_corrupt_ledger(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"seq": 1, "ts": "x"}\nnot json\n')
    proc = _run_report("--ledger", str(bad))
    assert proc.returncode == 2
    assert "error:" in proc.stderr
