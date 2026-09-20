"""JevRAG decision abstraction + Jev backend (V1, decision path).

The shape of the whole thing::

    state -> Decision -> confidence -> action

A ``Decision`` backend takes an opaque ``state`` and a list of typed questions
and returns a ``DecisionResult`` carrying the typed answer, a confidence in
[0, 1], and metadata (backend name, latency, token/cost accounting, model id,
raw response). Calling code owns the action policy (thresholds, retries) —
the model never decides what happens next.

Backends
--------
- :class:`JevDecision` — the V1 backend. Calls Jev through TypeSafe's own API
  (``typesafe-sdk``, model ``jev-latest``). NOT the Vercel AI Gateway route
  (``POST /v1/evaluate`` -> 404, chat-completions with the Jev model -> 400).
- :class:`StubDecision` — a deterministic, zero-cost test double implementing
  the same protocol. Exists so the sufficiency loop, the handoff record, and
  the measurement path can be exercised end to end before the Jev key
  arrives. Its outputs must never be reported as Jev numbers.

Confidence convention (flagged, see note below)
-----------------------------------------------
For Boolean (``noul``) questions — which is what evidence-sufficiency uses —
the SDK returns only a probability (``NoulAnswer.noul``); there is no separate
``confidence`` field. The probability IS the gate signal, so
``DecisionResult.confidence`` is set to it. For ``choice``/``score``
questions, ``confidence`` is the answer's own ``.confidence`` field and the
full distribution is kept in metadata.

NOTE: some documentation describes confidence as arriving at
``result.providerMetadata.typesafe.confidence``. That path exists only on the
AI SDK 7 ``experimental_evaluate()`` route (Node). On the direct TypeSafe API
/ Python SDK used here, Choice/Score confidence is ``answer.confidence`` and
Noul has no confidence field at all. This module implements the direct-SDK
semantics; if a future backend uses the AI SDK route it must map
``providerMetadata.typesafe.confidence`` onto ``DecisionResult.confidence``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

# Jev pricing: $0.042 per million input tokens, output free.
# (TypeSafe launch pricing, 2026-09-15.)
JEV_PRICE_PER_M_INPUT_TOKENS = 0.042

# Alias that tracks the stable release. Pin a versioned id (e.g. jev-1.13.0)
# once an operating threshold has been tuned against it.
JEV_MODEL_DEFAULT = "jev-latest"

#: Env var the key is read from. ``TYPESAFE_API_KEY`` (the SDK's native var)
#: is honored as a fallback since that is what the ecosystem tooling sets.
JEV_API_KEY_ENV = "JEV_API_KEY"
JEV_API_KEY_FALLBACK_ENV = "TYPESAFE_API_KEY"

QuestionKind = Literal["boolean", "noul", "choice", "score"]


@dataclass(frozen=True)
class TypedQuestion:
    """One typed question to a decision backend.

    ``kind`` uses PRD vocabulary (``boolean``); ``noul`` is accepted as an
    alias because that is what the TypeSafe SDK calls it. ``criteria`` is
    backend-specific: for ``choice`` a mapping of option name to description
    (description may be None); for ``score`` an ordered list of 2-10 level
    descriptions; for ``boolean``/``noul`` an optional ``{"true": ...,
    "false": ...}`` description pair.
    """

    name: str
    kind: QuestionKind
    instructions: str
    criteria: Any = None


@dataclass
class DecisionResult:
    """The answer to one ``ask()`` call.

    Single-question calls (the V1 contract) carry a scalar ``result`` and a
    scalar ``confidence``. Multi-question calls carry ``result`` as a dict of
    per-question answers, ``confidence`` as a dict of per-question confidences,
    and the same per-question breakdown under ``metadata["confidences"]``.
    """

    result: Any
    confidence: float | dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Decision(Protocol):
    """Swappable decision backend. The acceptance test for this abstraction
    (PRD section 2): adding a new backend or a new primitive must not require
    touching the harness, the controller, or any other primitive."""

    def ask(
        self, state: Any, questions: list[TypedQuestion]
    ) -> DecisionResult: ...


class MissingApiKeyError(RuntimeError):
    """Raised when a live backend has no API key to call with."""


def resolve_api_key(
    explicit: str | None = None,
    *,
    primary_env: str = JEV_API_KEY_ENV,
    fallback_env: str = JEV_API_KEY_FALLBACK_ENV,
) -> str:
    """Return an API key, or raise :class:`MissingApiKeyError`.

    Precedence: explicit argument, ``JEV_API_KEY``, ``TYPESAFE_API_KEY``.
    The fallback exists because the SDK and ecosystem docs set
    ``TYPESAFE_API_KEY``; supporting both avoids a silent misconfiguration
    where the key is present but unread. Never log or print the key itself.
    """
    if explicit:
        return explicit
    key = os.environ.get(primary_env) or os.environ.get(fallback_env)
    if not key:
        raise MissingApiKeyError(
            f"No Jev API key found. Ask Anmol for one and set {primary_env} "
            f"(or {fallback_env}). The key is never hardcoded or committed."
        )
    return key


def estimate_cost_usd(input_tokens: int) -> float:
    """Jev cost for a call. Output tokens are free."""
    return (input_tokens / 1_000_000) * JEV_PRICE_PER_M_INPUT_TOKENS


def _normalize_kind(kind: QuestionKind) -> str:
    # PRD says "boolean"; the SDK calls the same primitive "noul".
    return "noul" if kind == "boolean" else kind


class JevDecision:
    """V1 decision backend: Jev via TypeSafe's own API (``typesafe-sdk``).

    Args:
        model: Jev model id. Defaults to the ``jev-latest`` alias; pin a
            versioned id once a threshold is tuned against it.
        api_key: Optional explicit key. If omitted, resolved via
            :func:`resolve_api_key` (``JEV_API_KEY`` then ``TYPESAFE_API_KEY``).
        timeout: Per-call HTTP timeout in seconds.
    """

    backend_name = "jev"

    def __init__(
        self,
        model: str = JEV_MODEL_DEFAULT,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.timeout = timeout
        self._client: Any = None

    def _client_or_raise(self) -> Any:
        if self._client is None:
            from typesafe_sdk import TypeSafeClient

            key = resolve_api_key(self._api_key)
            self._client = TypeSafeClient(
                api_key=key, model=self.model, timeout=self.timeout
            )
        return self._client

    def _to_sdk_questions(
        self, questions: list[TypedQuestion]
    ) -> dict[str, Any]:
        from typesafe_sdk import Choice, Noul, Score

        sdk_questions: dict[str, Any] = {}
        for q in questions:
            kind = _normalize_kind(q.kind)
            if kind == "noul":
                kwargs: dict[str, Any] = {"instructions": q.instructions}
                if q.criteria is not None:
                    kwargs["criteria"] = q.criteria
                sdk_questions[q.name] = Noul(**kwargs)
            elif kind == "choice":
                if not q.criteria:
                    raise ValueError(
                        f"Choice question {q.name!r} needs criteria "
                        "{option: description}."
                    )
                sdk_questions[q.name] = Choice(
                    instructions=q.instructions, criteria=dict(q.criteria)
                )
            elif kind == "score":
                if not q.criteria:
                    raise ValueError(
                        f"Score question {q.name!r} needs an ordered list of "
                        "2-10 level descriptions."
                    )
                sdk_questions[q.name] = Score(
                    instructions=q.instructions, criteria=list(q.criteria)
                )
            else:
                raise ValueError(f"Unknown question kind {q.kind!r}.")
        return sdk_questions

    @staticmethod
    def _extract(answer: Any) -> tuple[Any, float, dict[str, Any]]:
        """Return (result, confidence, raw) for one SDK answer object."""
        answer_type = getattr(answer, "type", None)
        if answer_type == "noul":
            # No separate confidence on Noul answers by design: the
            # probability is already the measure. It is the gate signal.
            prob = float(answer.noul)
            raw = {"type": "noul", "noul": prob}
            return prob, prob, raw
        if answer_type == "choice":
            raw = {
                "type": "choice",
                "choice": answer.choice,
                "confidence": float(answer.confidence),
                "probabilities": dict(answer.probabilities),
            }
            return answer.choice, float(answer.confidence), raw
        if answer_type == "score":
            raw = {
                "type": "score",
                "score": float(answer.score),
                "confidence": float(answer.confidence),
                "legend": dict(answer.legend),
                "probabilities": {
                    str(k): float(v)
                    for k, v in dict(answer.probabilities).items()
                },
            }
            return float(answer.score), float(answer.confidence), raw
        raise ValueError(f"Unexpected Jev answer type {answer_type!r}.")

    def ask(
        self, state: Any, questions: list[TypedQuestion]
    ) -> DecisionResult:
        if not questions:
            raise ValueError("ask() needs at least one question.")
        client = self._client_or_raise()
        sdk_questions = self._to_sdk_questions(questions)

        started = time.perf_counter()
        response = client.system_one(state=state, questions=sdk_questions)
        latency_ms = (time.perf_counter() - started) * 1000.0

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        model_id = getattr(response, "model", self.model)

        results: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        raw_answers: dict[str, Any] = {}
        for q in questions:
            answer = response.answers[q.name]
            result, confidence, raw = self._extract(answer)
            results[q.name] = result
            confidences[q.name] = confidence
            raw_answers[q.name] = raw

        # The cost/latency table in the success bar is built from this
        # metadata. Record latency and input tokens on EVERY call.
        metadata: dict[str, Any] = {
            "backend": self.backend_name,
            "model": model_id,
            "model_requested": self.model,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": estimate_cost_usd(input_tokens),
            "raw_response": raw_answers,
        }
        if len(questions) == 1:
            q = questions[0]
            return DecisionResult(
                result=results[q.name],
                confidence=confidences[q.name],
                metadata=metadata,
            )
        metadata["confidences"] = confidences
        return DecisionResult(
            result=results, confidence=confidences, metadata=metadata
        )

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None


class StubDecision:
    """Deterministic zero-cost test double for the :class:`Decision` protocol.

    Returns a fixed confidence per question name (default 0.5) without any
    network call. Latency is the measured in-process time; token counts are 0.
    For ``boolean`` questions the result is ``confidence >= 0.5``.

    This exists to exercise the loop, the handoff record, and the measurement
    path before the Jev key arrives — and to prove a second backend drops in
    without touching anything else. Never report its outputs as Jev numbers;
    every record it touches is tagged ``backend: "stub"``.
    """

    backend_name = "stub"

    def __init__(self, confidences: Mapping[str, float] | None = None) -> None:
        self.confidences = dict(confidences or {})

    def ask(
        self, state: Any, questions: list[TypedQuestion]
    ) -> DecisionResult:
        if not questions:
            raise ValueError("ask() needs at least one question.")
        started = time.perf_counter()
        results: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        for q in questions:
            conf = float(self.confidences.get(q.name, 0.5))
            kind = _normalize_kind(q.kind)
            if kind == "noul":
                results[q.name] = conf
            elif kind == "choice":
                options = list((q.criteria or {}).keys())
                results[q.name] = options[0] if options else None
            elif kind == "score":
                levels = list(q.criteria or [])
                results[q.name] = float(len(levels) // 2)
            else:
                raise ValueError(f"Unknown question kind {q.kind!r}.")
            confidences[q.name] = conf
        latency_ms = (time.perf_counter() - started) * 1000.0
        metadata: dict[str, Any] = {
            "backend": self.backend_name,
            "model": "stub",
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "raw_response": {
                name: {"stub_confidence": conf}
                for name, conf in confidences.items()
            },
        }
        if len(questions) == 1:
            q = questions[0]
            return DecisionResult(
                result=results[q.name],
                confidence=confidences[q.name],
                metadata=metadata,
            )
        metadata["confidences"] = confidences
        return DecisionResult(
            result=results, confidence=confidences, metadata=metadata
        )
