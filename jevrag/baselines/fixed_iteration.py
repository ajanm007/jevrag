"""jevrag.baselines.fixed_iteration — always retrieve exactly N rounds, no gate.

This is what sufficiency has to beat: the same retrieval and
answer-generation machinery, with only the stopping policy changed — retrieve N
rounds unconditionally. Without it we have a number but not a result.

Implementation: one thin delegation to ``run_sufficiency(threshold=None)`` —
the primitive's own "never stop early" mode. The confound Claude flagged in
review (baseline secretly built on a different stack) is impossible by
construction here. An optional ``decision=`` backend may be passed to keep the
call signature symmetric with the gated path, but it is never consulted.

Gate-free records carry ``confidence: null`` and
``policy: "fixed_iteration_N"`` — there is no gate signal, and we say so
instead of inventing one.
"""

from __future__ import annotations

from typing import Any, Callable

from ..primitives.sufficiency import run_sufficiency


def run_fixed_iteration(
    question: dict[str, Any],
    *,
    n_rounds: int,
    retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    answer_fn: Callable[..., Any] | None = None,
    decision=None,
    **kwargs,
) -> dict:
    """Run one dataset question dict with the fixed-iteration policy.

    ``question`` is the HotpotQA-style dict from ``load_questions()`` (id,
    question, answer, type). Same ``retrieve_fn``/``answer_fn`` as the gated
    run — only the stopping rule differs.
    """
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    record = run_sufficiency(
        question_id=str(question["id"]),
        question=str(question["question"]),
        gold=str(question.get("answer", "")),
        qtype=str(question.get("type", "unknown")),
        decision=decision,  # never consulted: threshold=None
        retrieve_fn=retrieve_fn,
        answer_fn=answer_fn,
        threshold=None,
        max_rounds=n_rounds,
        policy=f"fixed_iteration_{n_rounds}",
        **kwargs,
    )
    record["confidence"] = None  # no gate signal exists for this policy
    return record


def run_fixed_iteration_dataset(
    questions: list[dict],
    *,
    n_rounds: int,
    retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    answer_fn: Callable[..., Any] | None = None,
    decision=None,
    **kwargs,
) -> list[dict]:
    """Fixed-iteration over a dataset slice, as returned by load_questions()."""
    return [
        run_fixed_iteration(
            q,
            n_rounds=n_rounds,
            retrieve_fn=retrieve_fn,
            answer_fn=answer_fn,
            decision=decision,
            **kwargs,
        )
        for q in questions
    ]

