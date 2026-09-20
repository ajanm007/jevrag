"""Evidence-sufficiency primitive (V1 — the one primitive).

The loop::

    retrieve -> ask Jev "is this evidence enough to answer?"
      -> confidence below threshold: retrieve another round
      -> confidence at/above threshold (or rounds exhausted): stop and answer

This module owns the decision path for sufficiency: building the Jev state,
asking the Boolean question, applying the stopping policy, and emitting one
per-question handoff record for the measurement path.

It does NOT own retrieval or answer generation: both are injected callables
so the primitive stays decoupled from any specific retriever or generator.
The HotpotQA wiring (RAG-Gate ``bm25_retriever`` / ``dense_retriever`` /
``data_loader``, prebuilt indices, real generator) lives on the measurement
side (``jevrag/benchmarks/hotpotqa.py``). Do not import retrievers here.

Handoff record (the seam with the measurement path — contract written
down in INTERFACE.md)
------------------------------------------------------------------------------
Agreed core keys::

    {
      "question_id": str,   # HotpotQA id (measurement joins on this for splits)
      "question": str,
      "gold": str,          # dataset field is `answer`; copied to `gold`
      "prediction": str,    # final generated answer ("" if the policy abstains)
      "confidence": float,  # Jev Boolean probability AT THE STOPPING POINT
      "rounds_used": int,   # retrieval rounds actually run, >= 1
      "latency_ms": float,  # wall-clock for the whole question
      "input_tokens": int,  # all input tokens for the question (Jev + generator)
      "type": str,          # bridge | comparison
      "policy": str,        # "sufficiency_jev" (gated) | "fixed_iteration_N"
    }

``confidence`` is the gate signal: it goes straight into ``selective.py``
and the ECE/Brier math. ``null`` is allowed only for gate-free baseline
records (``fixed_iteration``), which the baseline module builds — the
sufficiency loop itself always emits a float.

``correct`` is derived downstream from ``prediction`` vs ``gold`` via
RAG-Gate ``evaluator.py`` (EM/F1). :func:`validate_handoff_record` asserts the
required keys so either side fails fast if the seam drifts; unknown extra
fields are allowed (the measurement path ignores what it doesn't know).
Per-round detail is available separately via :func:`run_sufficiency_with_trace`
and is NOT eval input.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jevrag.decision import Decision, DecisionResult, TypedQuestion

#: Required keys of the per-question handoff record, in order.
HANDOFF_FIELDS = (
    "question_id",
    "question",
    "gold",
    "prediction",
    "confidence",
    "rounds_used",
    "latency_ms",
    "input_tokens",
    "type",
)

#: Policy tag for records emitted by the gated sufficiency loop.
POLICY_SUFFICIENCY = "sufficiency_jev"

#: Name of the Boolean question asked each round.
SUFFICIENCY_QUESTION_NAME = "sufficient"

#: Default stopping threshold. An operating point, not a truth: tune on VAL,
#: apply on TEST (threshold transferability is the harness's job).
#: ``None`` means "never stop early" — run all ``max_rounds``. That is the
#: gate-off knob the fixed-iteration baseline uses through this same loop.
DEFAULT_THRESHOLD: float | None = 0.7

#: Default round cap. HotpotQA questions need at most 2 hops; 3 rounds gives
#: one spare retrieval pass without unbounded cost.
DEFAULT_MAX_ROUNDS = 3

#: Docs added to the evidence set per round (cumulative top-n).
DEFAULT_PER_ROUND_K = 5

#: Per-doc character cap for evidence placed in the Jev state. Jev state
#: budget is 32k tokens; this keeps multi-round evidence well inside it.
DEFAULT_MAX_EVIDENCE_CHARS = 2000


@dataclass
class SufficiencyState:
    """State handed to the decision backend each round."""

    question: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    round: int = 1

    def to_jev_state(self, max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS) -> dict:
        """Render as a plain JSON-ish dict for the Jev call."""
        docs = []
        for doc in self.evidence:
            text = str(doc.get("text", ""))
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            docs.append(
                {"title": doc.get("title", ""), "text": text}
            )
        return {
            "question": self.question,
            "round": self.round,
            "evidence": docs,
        }


def sufficiency_question() -> TypedQuestion:
    """The V1 Boolean question: is this evidence enough to answer?"""
    return TypedQuestion(
        name=SUFFICIENCY_QUESTION_NAME,
        kind="boolean",
        instructions=(
            "Given the question and the evidence retrieved so far, "
            "is the evidence sufficient to answer the question correctly? "
            "Answer true only if the evidence states or directly implies "
            "everything needed for the answer."
        ),
        criteria={
            "true": "The evidence is enough to answer correctly.",
            "false": "More evidence is needed before answering.",
        },
    )


def extractive_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    """Weak default answer generator: first sentence of the top doc.

    This is a clearly-labeled baseline so the primitive runs standalone. The
    measurement side should inject a real generator via ``answer_fn``; any
    accuracy claim rests on that generator, not on this fallback. Gated and
    baseline runs must share whatever generator is used.
    """
    _ = question
    if not evidence:
        return ""
    text = str(evidence[0].get("text", "")).strip()
    if not text:
        return ""
    for sep in (". ", ".\n", "? ", "! "):
        if sep in text:
            return text.split(sep)[0].strip()[:300]
    return text[:300]


def make_handoff_record(
    *,
    question_id: str,
    question: str,
    gold: str,
    prediction: str,
    confidence: float | None,
    rounds_used: int,
    latency_ms: float,
    input_tokens: int,
    qtype: str,
    policy: str = POLICY_SUFFICIENCY,
) -> dict[str, Any]:
    """Build a handoff record: the agreed core keys plus ``policy``."""
    return {
        "question_id": question_id,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "confidence": None if confidence is None else float(confidence),
        "rounds_used": int(rounds_used),
        "latency_ms": float(latency_ms),
        "input_tokens": int(input_tokens),
        "type": qtype,
        "policy": policy,
    }


def validate_handoff_record(record: dict[str, Any]) -> dict[str, Any]:
    """Assert a record matches the agreed shape; return it unchanged.

    Requires the core keys, a prediction string, and confidence either
    ``None`` (gate-free baseline records only) or in [0, 1]. Unknown extra
    fields are allowed — the measurement path ignores what it doesn't know.
    Raises :class:`ValueError` otherwise, so either side fails fast on drift.
    """
    missing = [k for k in HANDOFF_FIELDS if k not in record]
    if missing:
        raise ValueError(
            "Handoff record keys drifted from the agreed shape. "
            f"Missing: {missing}."
        )
    conf = record["confidence"]
    if conf is not None and (
        not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0
    ):
        raise ValueError(
            f"Handoff confidence must be in [0, 1] or None, got {conf!r}."
        )
    if not isinstance(record["prediction"], str):
        raise ValueError("Handoff prediction must be a string.")
    return record


def _call_answer_fn(
    answer_fn: Callable[..., Any],
    question: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, int]:
    """Call the answer generator; return (prediction, generation input tokens).

    ``answer_fn`` may return a plain answer string (generation token usage
    unknown, counted as 0) or a ``(prediction, usage)`` pair where usage is a
    mapping with ``input_tokens``. The pair form is how a real generator
    reports its tokens so the cost table stays complete.
    """
    out = answer_fn(question, evidence)
    gen_tokens = 0
    if isinstance(out, tuple):
        prediction, usage = out
        if isinstance(usage, dict):
            gen_tokens = int(usage.get("input_tokens", 0) or 0)
    else:
        prediction = out
    if not isinstance(prediction, str):
        prediction = str(prediction)
    return prediction, gen_tokens


def run_sufficiency(
    *,
    question_id: str,
    question: str,
    gold: str,
    qtype: str,
    decision: Decision | None,
    retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    answer_fn: Callable[..., Any] | None = None,
    threshold: float | None = DEFAULT_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    per_round_k: int = DEFAULT_PER_ROUND_K,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    policy: str = POLICY_SUFFICIENCY,
) -> dict[str, Any]:
    """Run the sufficiency loop for one question; return its handoff record.

    Args:
        decision: Any :class:`Decision` backend (Jev, stub, future backends).
            May be ``None`` only when ``threshold`` is also ``None`` (gate
            off and no backend to ask): retrieval still runs, no Jev calls
            are made, and the record carries ``confidence=None``. This is the
            shape the fixed-iteration baseline uses.
        retrieve_fn: ``(question, n_docs) -> top-n doc dicts``. Called with
            cumulative counts (``per_round_k``, ``2 * per_round_k``, ...) so
            any top-k retriever (BM25, dense) plugs in unwrapped. Each doc
            needs at least ``title`` and ``text``.
        answer_fn: ``(question, evidence) -> answer string``, or
            ``-> (answer string, usage dict)`` so a real generator can report
            its input tokens. Defaults to :func:`extractive_answer` (weak
            baseline — inject the real generator for measured runs).
        threshold: Stop and answer when confidence >= threshold. ``None``
            disables the gate (run all ``max_rounds``) — the knob the
            fixed-iteration baseline uses through this same loop.
        max_rounds: Hard cap on retrieval rounds. Always answers after the
            final round regardless of confidence.

    ``latency_ms`` is wall-clock for the whole question (retrieval + Jev
    calls + generation); per-round Jev latency/tokens are in the trace from
    :func:`run_sufficiency_with_trace`.
    """
    record, _ = run_sufficiency_with_trace(
        question_id=question_id,
        question=question,
        gold=gold,
        qtype=qtype,
        decision=decision,
        retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
        threshold=threshold,
        max_rounds=max_rounds,
        per_round_k=per_round_k,
        max_evidence_chars=max_evidence_chars,
        policy=policy,
    )
    return record


def run_sufficiency_with_trace(
    *,
    question_id: str,
    question: str,
    gold: str,
    qtype: str,
    decision: Decision | None,
    retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    answer_fn: Callable[..., Any] | None = None,
    threshold: float | None = DEFAULT_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    per_round_k: int = DEFAULT_PER_ROUND_K,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
    policy: str = POLICY_SUFFICIENCY,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Same as :func:`run_sufficiency`, plus a per-round trace.

    The trace is diagnostic only (one entry per round: round, n_docs,
    confidence, latency_ms, input_tokens for that round's Jev call). It is
    NOT part of the handoff shape and must not be consumed as eval input.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1.")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1] or None.")
    if decision is None and threshold is not None:
        raise ValueError("A gate (threshold set) needs a decision backend.")
    answer = answer_fn or extractive_answer
    ask_question = sufficiency_question()

    evidence: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    jev_latency_ms = 0.0
    jev_input_tokens = 0
    confidence: float | None = 0.0
    rounds_used = 0

    wall_start = time.perf_counter()
    for round_no in range(1, max_rounds + 1):
        rounds_used = round_no
        evidence = list(retrieve_fn(question, round_no * per_round_k))
        if decision is None:
            # Gate off, no backend: retrieve only; no gate signal exists.
            confidence = None
            trace.append(
                {
                    "round": round_no,
                    "n_docs": len(evidence),
                    "confidence": None,
                    "latency_ms": 0.0,
                    "input_tokens": 0,
                }
            )
            continue
        state = SufficiencyState(
            question=question, evidence=evidence, round=round_no
        )
        result: DecisionResult = decision.ask(
            state.to_jev_state(max_chars=max_evidence_chars),
            [ask_question],
        )
        confidence = float(result.confidence)
        round_latency = float(result.metadata.get("latency_ms", 0.0))
        round_tokens = int(result.metadata.get("input_tokens", 0))
        jev_latency_ms += round_latency
        jev_input_tokens += round_tokens
        trace.append(
            {
                "round": round_no,
                "n_docs": len(evidence),
                "confidence": confidence,
                "latency_ms": round_latency,
                "input_tokens": round_tokens,
            }
        )
        if threshold is not None and confidence >= threshold:
            break

    prediction, gen_input_tokens = _call_answer_fn(answer, question, evidence)
    wall_latency_ms = (time.perf_counter() - wall_start) * 1000.0
    record = make_handoff_record(
        question_id=question_id,
        question=question,
        gold=gold,
        prediction=prediction,
        confidence=confidence,
        rounds_used=rounds_used,
        latency_ms=wall_latency_ms,
        input_tokens=jev_input_tokens + gen_input_tokens,
        qtype=qtype,
        policy=policy,
    )
    return validate_handoff_record(record), trace


def run_question(
    question: dict[str, Any],
    *,
    decision: Decision | None,
    retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    answer_fn: Callable[..., Any] | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    threshold: float | None = DEFAULT_THRESHOLD,
    gate: bool = True,
    per_round_k: int = DEFAULT_PER_ROUND_K,
    policy: str | None = None,
) -> dict[str, Any]:
    """Run one dataset question dict through the loop; return its record.

    ``question`` is a HotpotQA-style dict with ``id``, ``question``,
    ``answer`` (copied to ``gold``), and ``type``. ``gate=False`` disables
    the stopping rule (run all ``max_rounds``) — the single knob the
    fixed-iteration baseline turns; equivalent to ``threshold=None``.
    ``policy`` defaults to ``"sufficiency_jev"`` when gated and
    ``f"fixed_iteration_{max_rounds}"`` when gate-free.
    """
    if not gate:
        threshold = None
        policy = policy or f"fixed_iteration_{max_rounds}"
    else:
        policy = policy or POLICY_SUFFICIENCY
    record = run_sufficiency(
        question_id=str(question["id"]),
        question=str(question["question"]),
        gold=str(question.get("answer", "")),
        qtype=str(question.get("type", "unknown")),
        decision=decision,
        retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
        threshold=threshold,
        max_rounds=max_rounds,
        per_round_k=per_round_k,
        policy=policy,
    )
    if not gate:
        # No gate, no gate signal: baseline records carry null confidence.
        # (Use run_sufficiency(threshold=None) directly if the last-round
        # confidence is wanted for analysis.)
        record["confidence"] = None
        validate_handoff_record(record)
    return record


def write_records_jsonl(
    records: list[dict[str, Any]], path: str | Path
) -> Path:
    """Write validated handoff records as JSONL (one record per line).

    Convention: ``records/<policy>_<dataset>.jsonl`` relative to repo root.
    Each record is validated before writing, so a bad seam never lands
    silently in a file the measurement path will consume.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for record in records:
            validate_handoff_record(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
