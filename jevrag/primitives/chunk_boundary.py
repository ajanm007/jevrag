"""Chunk-boundary primitive (V1.1: the generalization test).

V1 (sufficiency) proved the Decision abstraction works once, for one
primitive. Per docs/prd/v3.md §4 that is explicitly NOT a generality claim —
this module is the test. Chunk-boundary is structurally different from
sufficiency in every way that matters: one-shot and pre-retrieval ("split
here or not?", asked once per candidate, never revisited, no loop) versus
iterative multi-round stopping. If this plugs into the same Decision
protocol, the same JevDecision backend, and the same calibration harness
(confidence + correctness in, AURC/ECE out) WITHOUT touching any of them,
the generalization claim is earned for real. If it can't, report that
plainly — do not quietly reshape the abstraction to fit.

The mechanism is not novel and isn't the point (LitSeg proved
margin-sampled split/no-split works). What's tested is hosting, not chunking.

Shape (same as always — action lives in code, never in the model):
- state: text window either side of one candidate boundary.
- Decision: one Jev Boolean call per candidate ("should this be a split?").
  The Noul probability IS the split score (same convention as sufficiency).
- action: split iff confidence >= threshold, else merge; low confidence
  falls back to the structural boundary (paragraph/sentence), per the v1
  PRD's own framing. Thresholds live with the caller, not here.

Handoff record: analogous to sufficiency's but honestly one-shot — no
``rounds_used`` that doesn't apply. Correctness labels come from reference
boundaries eval-side (is-split 0/1); confidence + correct feed the unchanged
harness exactly as sufficiency's do.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jevrag.decision import Decision, DecisionResult, TypedQuestion

#: Policy tag for records emitted by the one-shot boundary decision.
POLICY_CHUNK_BOUNDARY = "chunk_boundary_jev"

#: Name of the Boolean question asked per candidate boundary.
BOUNDARY_QUESTION_NAME = "split"

#: Default split/merge operating threshold. A presentation default only —
#: ranking metrics (AURC) don't use it; tune on labeled data for real use.
DEFAULT_SPLIT_THRESHOLD = 0.5


@dataclass
class BoundaryState:
    """State handed to the decision backend for one candidate boundary."""

    before: str
    after: str
    doc_id: str = ""
    boundary_index: int = 0

    def to_jev_state(self, max_chars: int = 2000) -> dict:
        """Render as a plain JSON-ish dict for the Jev call."""
        before, after = self.before.strip(), self.after.strip()
        if len(before) > max_chars:
            before = "…" + before[-max_chars:]
        if len(after) > max_chars:
            after = after[:max_chars] + "…"
        return {
            "before": before,
            "after": after,
            "doc_id": self.doc_id,
            "boundary_index": self.boundary_index,
        }


def boundary_question(window_sentences: int = 2) -> TypedQuestion:
    """The V1.1 Boolean question: should a chunk split go here?"""
    return TypedQuestion(
        name=BOUNDARY_QUESTION_NAME,
        kind="boolean",
        instructions=(
            "Two passages of a document are shown: BEFORE (the last "
            f"{window_sentences} sentences before a candidate boundary) and "
            f"AFTER (the first {window_sentences} sentences after it). "
            "Should a chunk boundary — a split point — go between them? "
            "Answer true only if the AFTER passage starts a new topic, "
            "section, or line of thought that does not belong in the same "
            "chunk as BEFORE."
        ),
        criteria={
            "true": "A new topic/section starts here; split.",
            "false": "Same continuing topic; keep merged.",
        },
    )


def decide_boundary(
    state: BoundaryState,
    decision: Decision,
    *,
    threshold: float = DEFAULT_SPLIT_THRESHOLD,
    max_chars: int = 2000,
) -> DecisionResult:
    """One-shot boundary decision. One Jev call, no loop, no revisit.

    Uses the Decision protocol and any backend implementing it, unmodified —
    this function is the entire V1.1 generalization test on the decision
    side. Returns the raw DecisionResult (result = P(split), confidence =
    P(split)); the split/merge action is the caller's, via ``label_for``.
    """
    return decision.ask(
        state.to_jev_state(max_chars=max_chars), [boundary_question()])


def label_for(confidence: float, threshold: float = DEFAULT_SPLIT_THRESHOLD
              ) -> str:
    """The action half: confidence → code-side split/merge decision."""
    return "split" if float(confidence) >= threshold else "merge"


def make_boundary_record(
    *,
    doc_id: str,
    boundary_index: int,
    result: str,
    confidence: float,
    latency_ms: float,
    input_tokens: int,
    policy: str = POLICY_CHUNK_BOUNDARY,
) -> dict[str, Any]:
    """Handoff record for one decided boundary — one-shot shape, no rounds."""
    if result not in ("split", "merge"):
        raise ValueError(f"result must be 'split' or 'merge', got {result!r}")
    return {
        "doc_id": doc_id,
        "boundary_index": int(boundary_index),
        "result": result,
        "confidence": float(confidence),
        "latency_ms": float(latency_ms),
        "input_tokens": int(input_tokens),
        "policy": policy,
    }


def decide_document(
    candidates: list[dict[str, Any]],
    decision: Decision,
    *,
    threshold: float = DEFAULT_SPLIT_THRESHOLD,
    max_chars: int = 2000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decide every candidate boundary of one document, independently.

    ``candidates``: dicts with ``doc_id``, ``boundary_index``, ``before``,
    ``after``. Each is asked exactly once — no loop, no shared state between
    candidates. Returns (records, trace); the trace holds per-candidate Jev
    latency/tokens and is diagnostic only.
    """
    ask_question = boundary_question()
    records, trace = [], []
    wall_start = time.perf_counter()
    for cand in candidates:
        state = BoundaryState(
            before=cand["before"], after=cand["after"],
            doc_id=str(cand.get("doc_id", "")),
            boundary_index=int(cand.get("boundary_index", 0)))
        result: DecisionResult = decision.ask(
            state.to_jev_state(max_chars=max_chars), [ask_question])
        conf = float(result.confidence)
        records.append(make_boundary_record(
            doc_id=state.doc_id, boundary_index=state.boundary_index,
            result=label_for(conf, threshold), confidence=conf,
            latency_ms=float(result.metadata.get("latency_ms", 0.0)),
            input_tokens=int(result.metadata.get("input_tokens", 0))))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    for rec in records:
        trace.append({"boundary_index": rec["boundary_index"],
                      "confidence": rec["confidence"],
                      "latency_ms": rec["latency_ms"],
                      "input_tokens": rec["input_tokens"]})
    return records, [{"wall_ms": wall_ms, "n": len(records), **t} for t in trace]
