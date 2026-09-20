"""Context-selection primitive (V2 path — §10's deliberate rebuild).

Given a query and one candidate passage, decide whether the passage is
relevant enough to include in the context handed to the generator. One Jev
call per candidate, no loop, no revisit: a filter/select decision. That makes
it the third structurally distinct shape this abstraction has hosted:

- sufficiency: iterative, per-round stop/go over evidence accumulated so far.
- chunk_boundary: one-shot, per-candidate split/merge on a text window.
- context_selection (this module): one-shot, per-candidate include/exclude.

The acceptance test (PRD §2): if this plugs into the
existing ``Decision`` protocol, the existing ``JevDecision`` backend and the
existing calibration harness with no edits to any of them, the generalization
claim earns a third data point. If it can't, report that plainly rather than
reshaping the abstraction to fit.

Shape (same as always — action lives in code, never in the model):
- state: the query plus one candidate passage (title + text).
- Decision: one Jev Boolean call per candidate. The Noul probability IS the
  relevance score, the same convention sufficiency and chunk_boundary use.
- action: include iff confidence >= threshold. What the action means matters:
  excluding a passage is *not* a claim that it is wrong, only that it did not
  clear the budget the caller set. Thresholds live with the caller, not here.

Correctness labels are NOT this module's business. ``selected`` is the calling
code's action; the record carries only identifying fields (``query_id``,
``passage_id``, ``passage_title``) so the measurement path can join its own
relevance labels and feed ``(confidence, correct)`` to the unchanged harness.

Set-level variant
-----------------
:func:`decide_query_batched` asks the same decisions as one multi-question
``ask()`` call — a genuinely different question ("is this passage relevant
*given* the others"), which is the stretch-goal variant the packet named. It
needs **no** protocol extension: the existing multi-question path carries one
typed question per passage and per-question ``instructions`` distinguish them
by id. It is deliberately not the default path — see its docstring for why
(per-item cost attribution, and comparability against a per-document
relevance scorer).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jevrag.decision import Decision, DecisionResult, TypedQuestion

#: Policy tag for records emitted by the one-shot passage decision.
POLICY_CONTEXT_SELECTION = "context_selection_jev"

#: Name of the Boolean question asked per candidate passage.
SELECTION_QUESTION_NAME = "relevant"

#: Default include/exclude operating threshold. A presentation default only —
#: ranking metrics (AURC) don't use it; tune on labeled data for real use.
DEFAULT_INCLUDE_THRESHOLD = 0.5

#: Per-field character cap for the rendered Jev state.
DEFAULT_MAX_CHARS = 2000

#: Marks records whose ``latency_ms``/``input_tokens`` are the *whole batched
#: call's*, shared by every record from that call. Summing them double-counts.
COST_SCOPE_CALL_SHARED = "call_shared"

#: Marks records whose ``latency_ms``/``input_tokens`` are that passage's own.
COST_SCOPE_PER_PASSAGE = "per_passage"


@dataclass
class PassageState:
    """State handed to the decision backend for one candidate passage."""

    query: str
    passage: str
    query_id: str = ""
    passage_id: str = ""
    passage_title: str = ""

    def to_jev_state(self, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
        """Render as a plain JSON-ish dict for the Jev call.

        Long fields are truncated at the *head* (the opening sentences of a
        passage and of a question carry the topical content). The truncation
        is deliberate and visible — an ellipsis marker, never a silent cut.
        """
        query, passage = self.query.strip(), self.passage.strip()
        if len(query) > max_chars:
            query = query[:max_chars] + "…"
        if len(passage) > max_chars:
            passage = passage[:max_chars] + "…"
        return {
            "query": query,
            "passage": passage,
            "passage_title": self.passage_title,
            "query_id": self.query_id,
            "passage_id": self.passage_id,
        }


def selection_question() -> TypedQuestion:
    """The Boolean question: is this passage relevant enough to include?"""
    return TypedQuestion(
        name=SELECTION_QUESTION_NAME,
        kind="boolean",
        instructions=(
            "A QUERY and one candidate PASSAGE are shown. Is this passage "
            "relevant enough to include in the context used to answer the "
            "query? Answer true if the passage contains information that "
            "helps answer the query — partial evidence for one step counts. "
            "Answer false if it is off-topic or does not help. Judge "
            "relevance to the query, not whether the passage alone answers "
            "it, and do not reward superficial keyword overlap."
        ),
        criteria={
            "true": "Relevant enough to include for this query.",
            "false": "Not relevant enough; exclude from the context.",
        },
    )


def decide_passage(
    state: PassageState,
    decision: Decision,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> DecisionResult:
    """One-shot passage decision. One Jev call, no loop, no revisit.

    Uses the ``Decision`` protocol and any backend implementing it, unmodified.
    Returns the raw ``DecisionResult`` (result = confidence = P(relevant)); the
    include/exclude action is the caller's, via :func:`select_for`.
    """
    return decision.ask(
        state.to_jev_state(max_chars=max_chars), [selection_question()])


def select_for(
    confidence: float, threshold: float = DEFAULT_INCLUDE_THRESHOLD
) -> bool:
    """The action half: confidence → code-side include/exclude decision."""
    return float(confidence) >= threshold


def make_selection_record(
    *,
    query_id: str,
    passage_id: str,
    confidence: float,
    selected: bool,
    latency_ms: float,
    input_tokens: int,
    passage_title: str = "",
    policy: str = POLICY_CONTEXT_SELECTION,
    cost_scope: str = COST_SCOPE_PER_PASSAGE,
) -> dict[str, Any]:
    """Handoff record for one decided passage — one-shot shape, no rounds.

    Guards at the seam, not downstream: ``selected`` must be a real bool and
    ``confidence`` a probability in [0, 1], so a non-probability never lands
    in a record the calibration harness will read. (The harness would catch it
    too, loudly — this just fails earlier, with a clearer message.)
    """
    if not isinstance(selected, bool):
        raise ValueError(f"selected must be a bool, got {selected!r}")
    conf = float(confidence)
    if not 0.0 <= conf <= 1.0:
        raise ValueError(
            f"confidence must be a probability in [0, 1]; got {conf!r}. If "
            "the score isn't a calibrated probability, map it explicitly at "
            "the call site and say so — don't clip it here."
        )
    return {
        "query_id": str(query_id),
        "passage_id": str(passage_id),
        "passage_title": str(passage_title),
        "confidence": conf,
        "selected": bool(selected),
        "latency_ms": float(latency_ms),
        "input_tokens": int(input_tokens),
        "cost_scope": cost_scope,
        "policy": policy,
    }


#: Keys every candidate dict must carry for :func:`decide_query`.
CANDIDATE_FIELDS = ("query", "query_id", "passage", "passage_id")


def _candidate_state(cand: dict[str, Any]) -> PassageState:
    missing = [k for k in CANDIDATE_FIELDS if k not in cand]
    if missing:
        raise KeyError(
            f"candidate is missing required field(s) {missing}; "
            f"expected all of {list(CANDIDATE_FIELDS)}"
        )
    return PassageState(
        query=str(cand["query"]),
        passage=str(cand["passage"]),
        query_id=str(cand["query_id"]),
        passage_id=str(cand["passage_id"]),
        passage_title=str(cand.get("passage_title", "")),
    )


def decide_query(
    candidates: list[dict[str, Any]],
    decision: Decision,
    *,
    threshold: float = DEFAULT_INCLUDE_THRESHOLD,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decide every candidate passage of one query, independently.

    ``candidates``: dicts with ``query``, ``query_id``, ``passage``,
    ``passage_id`` (``passage_title`` optional). Each candidate is asked
    exactly once and sees only its own passage — no shared state, no
    cross-passage context, which is what makes it directly comparable to a
    per-document relevance scorer.

    Returns ``(records, trace)``. The trace is diagnostic only: per-candidate
    latency/tokens, with no set-level figure repeated on every row (the caller
    times the set itself).
    """
    question = selection_question()
    records: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for cand in candidates:
        state = _candidate_state(cand)
        result: DecisionResult = decision.ask(
            state.to_jev_state(max_chars=max_chars), [question])
        conf = float(result.confidence)
        latency = float(result.metadata.get("latency_ms", 0.0))
        tokens = int(result.metadata.get("input_tokens", 0))
        records.append(make_selection_record(
            query_id=state.query_id,
            passage_id=state.passage_id,
            passage_title=state.passage_title,
            confidence=conf,
            selected=select_for(conf, threshold),
            latency_ms=latency,
            input_tokens=tokens))
        trace.append({
            "query_id": state.query_id,
            "passage_id": state.passage_id,
            "confidence": conf,
            "latency_ms": latency,
            "input_tokens": tokens,
        })
    return records, trace


def batched_question(passage_id: str) -> TypedQuestion:
    """One typed question per passage, for the set-level call.

    The question *name* carries the passage id and the instructions scope the
    judgment to that id — so a single shared state (query + all passages) can
    still carry per-passage questions. No protocol change required.
    """
    return TypedQuestion(
        name=f"{SELECTION_QUESTION_NAME}:{passage_id}",
        kind="boolean",
        instructions=(
            "A QUERY and a numbered list of PASSAGES (each with an id) are "
            f"shown. Consider passage id {passage_id!r} only. Is that "
            "passage relevant enough to include in the context used to "
            "answer the query? Answer true if it contains information that "
            "helps answer the query — partial evidence for one step counts — "
            "and false if it is off-topic or does not help. Judge it against "
            "the query, not against the other passages, and do not reward "
            "superficial keyword overlap."
        ),
        criteria={
            "true": "Relevant enough to include for this query.",
            "false": "Not relevant enough; exclude from the context.",
        },
    )


def batched_state(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    query_id: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """The shared state for one set-level call: query + all passages."""
    passages = []
    for cand in candidates:
        text = str(cand["passage"]).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        passages.append({
            "id": str(cand["passage_id"]),
            "title": str(cand.get("passage_title", "")),
            "text": text,
        })
    q = query.strip()
    if len(q) > max_chars:
        q = q[:max_chars] + "…"
    return {"query": q, "query_id": query_id, "passages": passages}


def decide_query_batched(
    candidates: list[dict[str, Any]],
    decision: Decision,
    *,
    threshold: float = DEFAULT_INCLUDE_THRESHOLD,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Set-level variant: all of one query's passages, ONE ``ask()`` call.

    This is the stretch shape — judging each passage in the context
    of the others — and it fits the existing protocol with no extension:
    ``ask(state, questions)`` already accepts many typed questions, and each
    question's name/instructions can name its own passage.

    Why it is not the baseline:

    1. **Comparability.** A per-document relevance scorer's judgment is
       per-document, so a per-passage decision is the like-for-like comparison.
    2. **Cost attribution.** Multi-question ``ask()`` reports one
       ``latency_ms``/``input_tokens`` for the whole call, with no per-question
       breakdown, so per-passage cost would have to be estimated. Every record
       here is therefore tagged :data:`COST_SCOPE_CALL_SHARED` — summing those
       numbers counts each call once per passage. That is a genuine (small)
       friction in the existing abstraction, reported rather than papered over.

    ``candidates`` need ``query``/``query_id``/``passage``/``passage_id`` as in
    :func:`decide_query`. Returns ``(records, trace)`` where the trace is the
    call-level summary.
    """
    if not candidates:
        return [], {"n": 0, "wall_ms": 0.0, "call_latency_ms": 0.0,
                    "call_input_tokens": 0}
    states = [_candidate_state(c) for c in candidates]
    query_ids = sorted({s.query_id for s in states})
    if len(query_ids) != 1:
        raise ValueError(
            "decide_query_batched expects candidates from ONE query; got "
            f"query_ids {query_ids}"
        )
    state = batched_state(
        states[0].query, candidates, query_id=query_ids[0], max_chars=max_chars)
    questions = [batched_question(s.passage_id) for s in states]

    wall_start = time.perf_counter()
    result: DecisionResult = decision.ask(state, questions)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    confidences = result.confidence
    if not isinstance(confidences, dict):
        raise ValueError(
            "batched ask() must return per-question confidences (a dict); "
            f"got {type(confidences).__name__}"
        )
    call_latency = float(result.metadata.get("latency_ms", 0.0))
    call_tokens = int(result.metadata.get("input_tokens", 0))
    records: list[dict[str, Any]] = []
    for s in states:
        name = f"{SELECTION_QUESTION_NAME}:{s.passage_id}"
        if name not in confidences:
            raise ValueError(
                f"backend returned no confidence for question {name!r}")
        conf = float(confidences[name])
        records.append(make_selection_record(
            query_id=s.query_id,
            passage_id=s.passage_id,
            passage_title=s.passage_title,
            confidence=conf,
            selected=select_for(conf, threshold),
            latency_ms=call_latency,
            input_tokens=call_tokens,
            cost_scope=COST_SCOPE_CALL_SHARED))
    trace = {
        "query_id": query_ids[0],
        "n": len(records),
        "wall_ms": wall_ms,
        "call_latency_ms": call_latency,
        "call_input_tokens": call_tokens,
    }
    return records, trace
