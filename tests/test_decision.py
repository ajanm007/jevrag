"""Tests for jevrag.decision — no network, no API key needed."""

import os

import pytest

from jevrag.decision import (
    JEV_PRICE_PER_M_INPUT_TOKENS,
    Decision,
    DecisionResult,
    JevDecision,
    MissingApiKeyError,
    StubDecision,
    TypedQuestion,
    estimate_cost_usd,
    resolve_api_key,
)


def bool_q(name="sufficient"):
    return TypedQuestion(
        name=name,
        kind="boolean",
        instructions="Is the evidence sufficient?",
    )


def test_estimate_cost_usd():
    assert estimate_cost_usd(1_000_000) == pytest.approx(JEV_PRICE_PER_M_INPUT_TOKENS)
    assert estimate_cost_usd(0) == 0.0
    # Output is free: cost depends only on input tokens.
    assert estimate_cost_usd(500_000) == pytest.approx(0.021)


def test_resolve_api_key_explicit_wins(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "env-key")
    assert resolve_api_key("explicit-key") == "explicit-key"


def test_resolve_api_key_primary_then_fallback(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        resolve_api_key()
    monkeypatch.setenv("TYPESAFE_API_KEY", "fallback-key")
    assert resolve_api_key() == "fallback-key"
    monkeypatch.setenv("JEV_API_KEY", "primary-key")
    assert resolve_api_key() == "primary-key"


def test_stub_boolean_passthrough():
    stub = StubDecision(confidences={"sufficient": 0.82})
    res = stub.ask({"question": "q"}, [bool_q()])
    assert isinstance(res, DecisionResult)
    assert res.result == pytest.approx(0.82)
    assert res.confidence == pytest.approx(0.82)
    assert res.metadata["backend"] == "stub"
    assert res.metadata["latency_ms"] >= 0.0
    assert res.metadata["input_tokens"] == 0
    assert res.metadata["cost_usd"] == 0.0


def test_stub_default_confidence():
    res = StubDecision().ask({}, [bool_q()])
    assert res.confidence == pytest.approx(0.5)


def test_stub_multi_question_returns_dicts():
    stub = StubDecision(confidences={"a": 0.9, "b": 0.2})
    res = stub.ask({}, [bool_q("a"), bool_q("b")])
    assert res.result == {"a": pytest.approx(0.9), "b": pytest.approx(0.2)}
    assert res.confidence == {"a": pytest.approx(0.9), "b": pytest.approx(0.2)}
    assert res.metadata["confidences"] == {"a": 0.9, "b": 0.2}


def test_stub_empty_questions_raises():
    with pytest.raises(ValueError):
        StubDecision().ask({}, [])


def test_protocol_conformance():
    assert isinstance(StubDecision(), Decision)
    assert isinstance(JevDecision(api_key="dummy"), Decision)


def test_jev_question_mapping():
    jev = JevDecision(api_key="dummy")
    sdk_qs = jev._to_sdk_questions(
        [
            bool_q(),
            TypedQuestion(
                name="dept", kind="choice", instructions="Route it.",
                criteria={"billing": "money stuff", "other": None},
            ),
            TypedQuestion(
                name="risk", kind="score", instructions="How risky?",
                criteria=["low", "high"],
            ),
        ]
    )
    assert sdk_qs["sufficient"].type == "noul"
    assert sdk_qs["dept"].type == "choice"
    assert sdk_qs["risk"].type == "score"


def test_jev_question_mapping_rejects_bad_specs():
    jev = JevDecision(api_key="dummy")
    with pytest.raises(ValueError):
        jev._to_sdk_questions(
            [TypedQuestion(name="c", kind="choice", instructions="x")]
        )
    with pytest.raises(ValueError):
        jev._to_sdk_questions(
            [TypedQuestion(name="q", kind="bogus", instructions="x")]  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        jev.ask({}, [])


def test_jev_extract_noul_confidence_is_probability():
    from typesafe_sdk import NoulAnswer

    result, confidence, raw = JevDecision._extract(NoulAnswer(noul=0.73))
    assert result == pytest.approx(0.73)
    # Noul answers carry no separate confidence: probability is the signal.
    assert confidence == pytest.approx(0.73)
    assert raw == {"type": "noul", "noul": pytest.approx(0.73)}


def test_jev_extract_choice_and_score():
    from typesafe_sdk import ChoiceAnswer, ScoreAnswer

    result, confidence, raw = JevDecision._extract(
        ChoiceAnswer(choice="a", confidence=0.9, probabilities={"a": 0.9, "b": 0.1})
    )
    assert result == "a"
    assert confidence == pytest.approx(0.9)
    assert raw["probabilities"] == {"a": 0.9, "b": 0.1}

    result, confidence, raw = JevDecision._extract(
        ScoreAnswer(
            score=1.5, confidence=0.6,
            legend={0: "low", 1: "high"}, probabilities={0: 0.5, 1: 0.5},
        )
    )
    assert result == pytest.approx(1.5)
    assert confidence == pytest.approx(0.6)


class _FakeUsage:
    def __init__(self, in_tok=190, out_tok=0):
        self.input_tokens = in_tok
        self.output_tokens = out_tok


class _FakeResponse:
    def __init__(self, answers, in_tok=190):
        self.answers = answers
        self.usage = _FakeUsage(in_tok)
        self.model = "jev-1.13.0"


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return self._response


def test_jev_ask_records_latency_and_tokens():
    from typesafe_sdk import NoulAnswer

    jev = JevDecision(api_key="dummy")
    jev._client = _FakeClient(_FakeResponse({"sufficient": NoulAnswer(noul=0.66)}))
    res = jev.ask({"question": "q"}, [bool_q()])
    assert res.result == pytest.approx(0.66)
    assert res.confidence == pytest.approx(0.66)
    md = res.metadata
    assert md["backend"] == "jev"
    assert md["model"] == "jev-1.13.0"
    assert md["latency_ms"] >= 0.0
    assert md["input_tokens"] == 190
    assert md["cost_usd"] == pytest.approx(190 / 1_000_000 * JEV_PRICE_PER_M_INPUT_TOKENS)


def test_jev_ask_multi_question_dicts():
    from typesafe_sdk import NoulAnswer

    jev = JevDecision(api_key="dummy")
    jev._client = _FakeClient(
        _FakeResponse({"a": NoulAnswer(noul=0.9), "b": NoulAnswer(noul=0.1)})
    )
    res = jev.ask({}, [bool_q("a"), bool_q("b")])
    assert res.result == {"a": pytest.approx(0.9), "b": pytest.approx(0.1)}
    assert res.confidence == {"a": pytest.approx(0.9), "b": pytest.approx(0.1)}


def test_jev_ask_without_key_raises(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    jev = JevDecision()
    with pytest.raises(MissingApiKeyError):
        jev.ask({}, [bool_q()])
