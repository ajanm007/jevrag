"""LogprobDecision — a second real ``Decision`` backend (v0.2.0 item 6).

Confidence signal: the mean per-token log-probability of the generator's own
completion for the answer being scored. Zero marginal cost — logprobs arrive
free with generation, so this backend makes no network calls and needs no
API key. Every primitive now has an available free baseline to compare Jev
against.

Two real design decisions, made and documented here rather than defaulted:

State contract: ``state`` must be a Mapping carrying PRE-COMPUTED logprob
data — either ``"logprobs"`` (a per-token sequence of floats) or
``"mean_logprob"`` (a float), with an optional ``"n_logprob_tokens"`` count.
Any other state keys (question/evidence/answer text, generator tags, …) are
ignored. This backend never calls a generator itself, for three reasons:

1. The generate/eval seam is deliberate: ``jevrag eval`` never calls a model,
   and generation lives in ``scripts/``. A decision backend that secretly
   needed network access, keys, and model config would smuggle infrastructure
   through exactly the seam the project keeps clean.
2. The records this backend is meant to score already carry the signal
   (``outputs/answer_abstain_4omini_val.jsonl`` stores ``mean_logprob`` per
   row), so pre-computed state is runnable today with zero spend.
3. Re-generating inside the decision would double-spend and confound
   generator identity with decision identity — the exact confound the
   earlier logprob-vs-Jev comparison refused when it rejected mixing one
   generator's logprobs onto another's predictions.

Because the ``Decision`` protocol types ``state`` as opaque ``Any``, no
protocol change was needed: the logprob fields are additive keys the backend
reads while ignoring the text fields every primitive already passes. The
abstraction's signature turned out wide enough — that is itself a finding
about its generality, not a forced fit.

Confidence mapping: ``confidence = exp(mean_logprob)``. If ``m`` is the mean
of per-token log-probabilities ``log p_i``, then ``e^m`` is the geometric
mean of the per-token probabilities ``(Π p_i)^(1/n)`` — the model's average
per-token probability. That makes this a genuine probability in (0, 1] with
no tunable parameters, not an arbitrary squash. It is also strictly
monotonic in ``m``, so ranking metrics (AURC) are directly comparable to the
earlier throwaway-script comparison that scored raw mean logprobs. Rejected
alternatives: a sigmoid/logistic squash (its slope and intercept would be
arbitrary — or worse, fitted to the eval set being reported on), min-max
normalization against a reference range (arbitrary endpoints, corpus-relative
rather than per-row), and shipping raw ``m`` as the confidence (violates the
``[0, 1]`` convention ``eval/calibration.py`` enforces loudly).

Boolean/``noul`` questions: ``result`` is the confidence itself — the
probability IS the gate signal, the same convention as Jev ``noul`` answers
and ``StubDecision``. ``choice``/``score`` questions raise a clear
``NotImplementedError``: a single per-completion scalar cannot rank discrete
options without inventing numbers, so this backend refuses loudly instead of
producing silent nonsense.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from jevrag.decision import DecisionResult, TypedQuestion

#: Backend tag written into every record this backend touches.
BACKEND_NAME = "logprob"


def _is_boolean(kind: str) -> bool:
    # PRD says "boolean"; the TypeSafe SDK calls the same primitive "noul".
    # (Same alias rule as decision.py; re-stated here so this module imports
    # only public names from jevrag.decision.)
    return kind in ("boolean", "noul")


def _extract_mean_logprob(state: Any) -> tuple[float, int | None]:
    """Return ``(mean_logprob, n_tokens)`` from a state mapping.

    A per-token ``"logprobs"`` array takes precedence when present (the mean
    is computed here, so the caller cannot smuggle in an inconsistent
    average); otherwise a pre-averaged ``"mean_logprob"`` is used as-is.
    Anything else — non-mapping state, missing keys, empty arrays, NaN,
    positive "log"-probabilities (mathematically impossible) — raises loudly
    rather than producing a silent number.
    """
    if not isinstance(state, Mapping):
        raise TypeError(
            "LogprobDecision state must be a mapping carrying pre-computed "
            "logprob data ('logprobs' per-token array or 'mean_logprob' "
            f"float); got {type(state).__name__}. This backend never calls "
            "a generator — run generation first and pass its logprobs in."
        )
    raw = state.get("logprobs", None)
    if raw is not None:
        try:
            values = [float(v) for v in raw]
        except TypeError:
            raise ValueError(
                "LogprobDecision 'logprobs' must be a sequence of numbers, "
                f"got {raw!r}."
            )
        if not values:
            raise ValueError(
                "LogprobDecision got an empty 'logprobs' array — "
                "a completion with no scored tokens has no mean logprob."
            )
        for v in values:
            if math.isnan(v):
                raise ValueError(
                    "LogprobDecision 'logprobs' contains NaN — refusing to "
                    "average over unknown token probabilities."
                )
            if v > 0.0:
                raise ValueError(
                    f"LogprobDecision 'logprobs' contains {v!r} > 0: "
                    "log-probabilities are <= 0 by definition, so this "
                    "input is not a logprob array — fix it at the source."
                )
        return sum(values) / len(values), len(values)
    if state.get("mean_logprob", None) is not None:
        mean_lp = float(state["mean_logprob"])
        if math.isnan(mean_lp):
            raise ValueError(
                "LogprobDecision 'mean_logprob' is NaN — no signal to map."
            )
        if mean_lp == math.inf:
            raise ValueError(
                "LogprobDecision 'mean_logprob' is +inf: log-probabilities "
                "are <= 0 by definition, so this input is not a mean "
                "logprob — fix it at the source."
            )
        if mean_lp > 0.0:
            raise ValueError(
                f"LogprobDecision 'mean_logprob' is {mean_lp!r} > 0: "
                "log-probabilities are <= 0 by definition — fix it at "
                "the source."
            )
        # -inf is mathematically fine (the model assigned ~zero probability
        # to the completion) and maps to confidence 0.0 below.
        n_tokens = state.get("n_logprob_tokens", None)
        return mean_lp, (int(n_tokens) if n_tokens is not None else None)
    raise ValueError(
        "LogprobDecision state carries no logprob data: need either a "
        "'logprobs' per-token array or a 'mean_logprob' float "
        f"(state keys: {sorted(str(k) for k in state)}). Generate first, "
        "then pass the generation's logprobs in — this backend makes no "
        "generation call of its own."
    )


def confidence_from_mean_logprob(mean_logprob: float) -> float:
    """Map a mean token log-probability to a confidence in [0, 1].

    ``exp(m)`` is the geometric-mean per-token probability (see the module
    docstring for the justification). ``m = 0`` maps to exactly 1.0;
    ``m -> -inf`` maps to 0.0.
    """
    return min(1.0, math.exp(mean_logprob))


class LogprobDecision:
    """Second real ``Decision`` backend: generator-logprob confidence.

    Pure function of pre-computed logprobs — no network, no key, zero cost.
    Answers ``boolean``/``noul`` questions; raises ``NotImplementedError``
    for ``choice``/``score``. Implements the ``Decision`` protocol
    unmodified (``ask(state, questions) -> DecisionResult``).
    """

    backend_name = BACKEND_NAME

    def ask(
        self, state: Any, questions: list[TypedQuestion]
    ) -> DecisionResult:
        if not questions:
            raise ValueError("ask() needs at least one question.")
        for q in questions:
            if not _is_boolean(q.kind) and q.kind not in ("choice", "score"):
                raise ValueError(f"Unknown question kind {q.kind!r}.")
        unsupported = [q for q in questions if not _is_boolean(q.kind)]
        if unsupported:
            names = ", ".join(f"{q.kind}:{q.name}" for q in unsupported)
            raise NotImplementedError(
                f"LogprobDecision cannot answer {names}: a per-completion "
                "logprob scalar cannot rank discrete options without "
                "inventing numbers. Use the Jev backend for choice/score."
            )
        started = time.perf_counter()
        mean_lp, n_tokens = _extract_mean_logprob(state)
        confidence = confidence_from_mean_logprob(mean_lp)
        latency_ms = (time.perf_counter() - started) * 1000.0
        # The signal belongs to whichever generator produced the completion;
        # echo its tag when the caller provides one so provenance travels
        # with the record (the project's generator-parity ethos).
        generator = "unknown"
        if isinstance(state, Mapping) and state.get("generator") is not None:
            generator = str(state.get("generator"))
        results: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        raw: dict[str, Any] = {}
        for q in questions:
            # Boolean convention (same as Jev noul + Stub): the probability
            # IS the gate signal, so result and confidence coincide.
            results[q.name] = confidence
            confidences[q.name] = confidence
            raw[q.name] = {
                "mean_logprob": mean_lp,
                "n_logprob_tokens": n_tokens,
                "confidence": confidence,
            }
        metadata: dict[str, Any] = {
            "backend": self.backend_name,
            "model": generator,
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "raw_response": raw,
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
