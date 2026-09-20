"""Answer-abstain primitive (reopened scope: post-generation grounding check).

Question: given a question, the evidence retrieved for it, and a generated
answer, does the answer stay grounded in that evidence — or drift, add
unsupported detail, or answer something the evidence doesn't support?

Same shape as every other primitive (action lives in code, never in the model):
- state: question + retrieved evidence passage text + generated answer.
- Decision: one Jev Boolean call, "is this answer supported by this
  evidence?" — the same TypedQuestion(kind="boolean") pattern as
  sufficiency/chunk-boundary, via the Decision protocol unmodified.
- action: confidence >= threshold → "pass" (answer goes through);
  below → "abstain" (no answer returned). No regeneration loop in this
  first cut — regeneration is a pipeline policy above the primitive, not
  part of the decision; stating the boundary instead of absorbing it.

Ground-truth note: no dataset of (generated answer, is-grounded) pairs
exists. First-cut proxy:
correctness (EM) as the harness label — "wrong answer" correlates with
"ungrounded answer" without being identical; the failure-mode split
(prediction content absent from vs present in evidence) separates the
cleaner ungrounded cases from sufficiency/generator failures. Both the
proxy and its limitation travel with every number reported from this
module. A rag-gate logprob comparison is real future work, explicitly not
this packet (different signal: answer trust vs evidence grounding).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jevrag.decision import Decision, DecisionResult, TypedQuestion

#: Policy tag for records emitted by the grounding check.
POLICY_ANSWER_ABSTAIN = "answer_abstain_jev"

#: Name of the Boolean question asked per answer.
GROUNDING_QUESTION_NAME = "grounded"

#: Default pass/abstain operating threshold. Presentation default only —
#: ranking metrics don't use it.
DEFAULT_ABSTAIN_THRESHOLD = 0.5

#: Default per-evidence-doc character cap in the Jev state.
DEFAULT_MAX_EVIDENCE_CHARS = 2000


@dataclass
class GroundingState:
    """State handed to the decision backend for one generated answer."""

    question: str
    evidence: list[dict[str, Any]]
    prediction: str
    question_id: str = ""

    def to_jev_state(self, max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS) -> dict:
        """Render as a plain JSON-ish dict for the Jev call."""
        docs = []
        for doc in self.evidence:
            text = str(doc.get("text", ""))
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            docs.append({"title": doc.get("title", ""), "text": text})
        return {
            "question": self.question,
            "evidence": docs,
            "answer": self.prediction,
        }


def grounding_question() -> TypedQuestion:
    """The Boolean question: is this answer supported by this evidence?"""
    return TypedQuestion(
        name=GROUNDING_QUESTION_NAME,
        kind="boolean",
        instructions=(
            "A question was answered using the shown retrieved evidence. "
            "Is the given answer actually supported by that evidence? "
            "Answer true only if every key fact in the answer is stated in "
            "or directly implied by the evidence passages."
        ),
        criteria={
            "true": "The answer is grounded in the evidence.",
            "false": ("The answer drifts, adds unsupported detail, or answers "
                      "something the evidence does not support."),
        },
    )


def check_grounding(
    state: GroundingState,
    decision: Decision,
    *,
    max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> DecisionResult:
    """One-shot grounding check. One Jev call, no loop.

    Uses the Decision protocol and any backend implementing it, unmodified.
    Returns the raw DecisionResult (result = P(grounded)); the pass/abstain
    action is the caller's, via ``action_for``.
    """
    return decision.ask(
        state.to_jev_state(max_chars=max_chars), [grounding_question()])


def action_for(confidence: float,
               threshold: float = DEFAULT_ABSTAIN_THRESHOLD) -> str:
    """The action half: confidence → code-side pass/abstain decision."""
    return "pass" if float(confidence) >= threshold else "abstain"


def make_abstain_record(
    *,
    question_id: str,
    question: str,
    gold: str,
    prediction: str,
    confidence: float,
    action: str,
    latency_ms: float,
    input_tokens: int,
    policy: str = POLICY_ANSWER_ABSTAIN,
) -> dict[str, Any]:
    """Handoff record for one checked answer — one-shot shape, no rounds."""
    if action not in ("pass", "abstain"):
        raise ValueError(f"action must be 'pass' or 'abstain', got {action!r}")
    return {
        "question_id": question_id,
        "question": question,
        "gold": gold,
        "prediction": prediction,
        "confidence": float(confidence),
        "action": action,
        "latency_ms": float(latency_ms),
        "input_tokens": int(input_tokens),
        "policy": policy,
    }


def check_answer(
    *,
    question_id: str,
    question: str,
    gold: str,
    evidence: list[dict[str, Any]],
    prediction: str,
    decision: Decision,
    threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    max_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check one generated answer; return (record, trace).

    The trace is diagnostic only (Jev latency/tokens for the single call).
    """
    wall_start = time.perf_counter()
    result: DecisionResult = check_grounding(
        GroundingState(question=question, evidence=evidence,
                       prediction=prediction, question_id=question_id),
        decision, max_chars=max_chars)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    conf = float(result.confidence)
    record = make_abstain_record(
        question_id=question_id, question=question, gold=gold,
        prediction=prediction, confidence=conf,
        action=action_for(conf, threshold),
        latency_ms=float(result.metadata.get("latency_ms", 0.0)),
        input_tokens=int(result.metadata.get("input_tokens", 0)))
    trace = {"latency_ms": record["latency_ms"],
             "input_tokens": record["input_tokens"],
             "wall_ms": wall_ms}
    return record, trace
