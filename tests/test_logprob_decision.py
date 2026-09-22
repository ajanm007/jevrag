"""Tests for jevrag.backends.logprob_decision — all offline, no network.

Covers the confidence-mapping logic specifically (a case mapping to ~1.0,
one to ~0.0, one boundary case), the typed-question handling (boolean/noul
pass through, choice/score refuse loudly), the state-contract validation,
and the structural acceptance test (backend imports decision.py + stdlib
only — the same seam every primitive is held to).
"""

import ast
import math
from pathlib import Path

import pytest

from jevrag.backends.logprob_decision import (
    LogprobDecision,
    confidence_from_mean_logprob,
)
from jevrag.decision import Decision, TypedQuestion


def bool_q(name="grounded"):
    return TypedQuestion(
        name=name,
        kind="boolean",
        instructions="Is this answer supported by the evidence?",
    )


# --- the confidence mapping itself ---

def test_mapping_perfect_logprob_maps_to_one():
    assert confidence_from_mean_logprob(0.0) == pytest.approx(1.0)


def test_mapping_very_negative_maps_to_near_zero():
    assert confidence_from_mean_logprob(-10.0) == pytest.approx(4.539993e-05)


def test_mapping_boundary_case():
    # ln(0.5): the midpoint of the probability scale.
    assert confidence_from_mean_logprob(math.log(0.5)) == pytest.approx(0.5)


def test_mapping_is_geometric_mean_token_probability():
    # exp(mean(log p_i)) == (prod p_i)^(1/n): the documented justification,
    # checked against the definitional form, not just itself.
    logprobs = [-0.1, -0.3, -0.05]
    mean_lp = sum(logprobs) / len(logprobs)
    assert confidence_from_mean_logprob(mean_lp) == pytest.approx(
        math.exp(sum(logprobs) / len(logprobs))
    )
    assert 0.0 <= confidence_from_mean_logprob(mean_lp) <= 1.0


# --- typed-question handling ---

def test_boolean_result_is_confidence():
    backend = LogprobDecision()
    res = backend.ask({"mean_logprob": -0.5}, [bool_q()])
    expected = math.exp(-0.5)
    assert res.result == pytest.approx(expected)
    assert res.confidence == pytest.approx(expected)


def test_noul_alias_agrees_with_boolean():
    backend = LogprobDecision()
    q = TypedQuestion(name="grounded", kind="noul", instructions="x")
    res = backend.ask({"mean_logprob": -0.2}, [q])
    assert res.confidence == pytest.approx(math.exp(-0.2))


def test_per_token_array_path_averages():
    backend = LogprobDecision()
    res = backend.ask({"logprobs": [-0.1, -0.3]}, [bool_q()])
    assert res.confidence == pytest.approx(math.exp(-0.2))
    assert res.metadata["raw_response"]["grounded"]["n_logprob_tokens"] == 2


def test_raw_array_takes_precedence_over_stale_mean():
    backend = LogprobDecision()
    res = backend.ask(
        {"logprobs": [-0.4, -0.6], "mean_logprob": 0.0}, [bool_q()]
    )
    assert res.confidence == pytest.approx(math.exp(-0.5))


def test_choice_raises_not_implemented():
    backend = LogprobDecision()
    q = TypedQuestion(name="dept", kind="choice", instructions="Route it.",
                      criteria={"a": "x", "b": "y"})
    with pytest.raises(NotImplementedError):
        backend.ask({"mean_logprob": -0.1}, [q])


def test_score_raises_not_implemented():
    backend = LogprobDecision()
    q = TypedQuestion(name="risk", kind="score", instructions="How risky?",
                      criteria=["low", "high"])
    with pytest.raises(NotImplementedError):
        backend.ask({"mean_logprob": -0.1}, [q])


def test_unknown_kind_raises_value_error():
    backend = LogprobDecision()
    q = TypedQuestion(name="q", kind="bogus", instructions="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        backend.ask({"mean_logprob": -0.1}, [q])


def test_empty_questions_raises():
    with pytest.raises(ValueError):
        LogprobDecision().ask({"mean_logprob": -0.1}, [])


def test_multi_question_returns_dicts():
    backend = LogprobDecision()
    res = backend.ask({"mean_logprob": -0.25}, [bool_q("a"), bool_q("b")])
    assert res.result == {"a": pytest.approx(math.exp(-0.25)),
                          "b": pytest.approx(math.exp(-0.25))}
    assert res.metadata["confidences"]["a"] == pytest.approx(math.exp(-0.25))


# --- state-contract validation: loud refusal, never silent numbers ---

def test_missing_logprob_keys_raise_helpfully():
    with pytest.raises(ValueError, match="no logprob data"):
        LogprobDecision().ask({"question": "q", "answer": "a"}, [bool_q()])


def test_non_mapping_state_raises_type_error():
    with pytest.raises(TypeError, match="must be a mapping"):
        LogprobDecision().ask("just some text", [bool_q()])


def test_empty_logprob_array_raises():
    with pytest.raises(ValueError, match="empty"):
        LogprobDecision().ask({"logprobs": []}, [bool_q()])


def test_positive_mean_logprob_raises():
    with pytest.raises(ValueError, match="> 0"):
        LogprobDecision().ask({"mean_logprob": 0.5}, [bool_q()])


def test_positive_per_token_logprob_raises():
    with pytest.raises(ValueError, match="> 0"):
        LogprobDecision().ask({"logprobs": [-0.1, 0.3]}, [bool_q()])


def test_nan_mean_logprob_raises():
    with pytest.raises(ValueError, match="NaN"):
        LogprobDecision().ask({"mean_logprob": float("nan")}, [bool_q()])


def test_negative_inf_maps_to_zero():
    res = LogprobDecision().ask({"mean_logprob": float("-inf")}, [bool_q()])
    assert res.confidence == pytest.approx(0.0)


# --- zero-cost metadata shape ---

def test_metadata_is_zero_cost():
    res = LogprobDecision().ask(
        {"mean_logprob": -0.1, "n_logprob_tokens": 3,
         "generator": "openrouter:openai/gpt-4o-mini"}, [bool_q()])
    md = res.metadata
    assert md["backend"] == "logprob"
    assert md["model"] == "openrouter:openai/gpt-4o-mini"
    assert md["input_tokens"] == 0
    assert md["output_tokens"] == 0
    assert md["cost_usd"] == 0.0
    assert md["latency_ms"] >= 0.0
    assert md["raw_response"]["grounded"]["mean_logprob"] == pytest.approx(-0.1)


# --- the actual plug-in proof at the question level ---

def test_answers_primitive_real_question_without_touching_primitive():
    """The backend answers answer-abstain's own grounding question, given a
    state carrying the primitive's text fields PLUS logprob data — the
    primitive itself is imported read-only, never modified."""
    from jevrag.primitives.answer_abstain import (
        GroundingState,
        grounding_question,
    )

    state = GroundingState(
        question="Q?", evidence=[{"title": "t", "text": "x"}], prediction="x"
    ).to_jev_state()
    state["mean_logprob"] = -0.05
    state["n_logprob_tokens"] = 2
    res = LogprobDecision().ask(state, [grounding_question()])
    assert res.confidence == pytest.approx(math.exp(-0.05))


def test_protocol_conformance():
    assert isinstance(LogprobDecision(), Decision)


def test_acceptance_backend_imports_only_decision_and_stdlib():
    path = (Path(__file__).resolve().parent.parent / "jevrag" / "backends"
            / "logprob_decision.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
    allowed_prefix = ("jevrag.decision",)
    banned = [m for m in imports
              if m.startswith("jevrag.")
              and not m.startswith(allowed_prefix)]
    assert not banned, f"backend reaches past decision.py: {banned}"
    assert any(m == "jevrag.decision" for m in imports)
