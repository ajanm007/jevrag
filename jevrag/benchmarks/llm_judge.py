"""jevrag.benchmarks.llm_judge — LLM-as-judge scorer for long-form answers.

HotpotQA's EM scorer is string equality over short spans; DocBench golds are
full sentences, where EM silently reports ~0 regardless of correctness. This
module scores semantic equivalence instead: one judge call per question
(question + gold + prediction in, binary verdict out).

Design calls, stated plainly: **binary**, not graded — the harness downstream
consumes a binary `correct` label (same shape as EM), and graded scores would
need a threshold to become one anyway. **4o-mini via OpenRouter** — confirmed
with Anmol as the right cost tier for low-volume judging (hundreds of calls,
not bulk generation). The judge's raw output and reason are always recorded
("capture now, decide later"); usage tokens are recorded, cost is not
invented. F1 stays alongside the verdict — more signal, not less.

General on purpose: nothing DocBench-specific in here. Any dataset with
long-form golds can use it.
"""

from __future__ import annotations

import re
from typing import Any

#: 4o-mini through OpenRouter (OpenAI-compatible chat API).
JUDGE_MODEL = "openai/gpt-4o-mini"

JUDGE_PROMPT = """You grade whether a system's answer to a question is correct, given a reference answer.

Question: {question}

Reference answer: {gold}

System answer: {prediction}

Rules:
- CORRECT if the system answer states the same key fact(s) as the reference, even with different wording, extra context, or different length.
- INCORRECT if it contradicts the reference, omits the key fact, answers a different question, or is empty/refuses.
- An empty or blank system answer is ALWAYS INCORRECT — there is nothing to judge charitably.
- Judge the answer, not the style. Partial overlap that keeps the key fact counts as CORRECT.

Reply in exactly this shape, two lines:
VERDICT: CORRECT
REASON: <one sentence saying what matched or what was missing/wrong>"""

_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(CORRECT|INCORRECT)\s*$",
                         re.IGNORECASE | re.MULTILINE)


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """Parse the judge's reply → (verdict, reason).

    verdict is True/False, or None when the reply doesn't contain a parseable
    VERDICT line — never guessed. The caller decides what None means (for a
    one-doc chain: loud failure, not a silent label).
    """
    match = _VERDICT_RE.search(text or "")
    verdict = None if match is None else match.group(1).upper() == "CORRECT"
    reason = ""
    for line in (text or "").splitlines():
        if line.strip().upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
            break
    return verdict, reason


def make_judge_fn(client, model: str = JUDGE_MODEL):
    """One judge closure over an OpenAI-compatible client.

    Returns ``judge_fn(question, gold, prediction) -> dict`` with ``verdict``
    (True/False/None), ``reason``, ``raw`` (full reply), ``model``, and
    ``usage`` (prompt/completion tokens as reported, else 0s).
    """

    def judge_fn(question: str, gold: str, prediction: str) -> dict[str, Any]:
        # Structural empty-answer rule (not prompt wording): an empty or
        # whitespace-only prediction is deterministically INCORRECT — no model
        # call, no cost, no chance of the judge disagreeing with itself run
        # to run, and immune to future prompt edits. Found live: 4o-mini was
        # scoring blank predictions CORRECT despite the prompt's empty-rule.
        if not prediction or not prediction.strip():
            return {
                "verdict": False,
                "reason": "empty or whitespace-only prediction",
                "raw": None,
                "model": model,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        prompt = JUDGE_PROMPT.format(question=question, gold=gold,
                                     prediction=prediction)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        verdict, reason = parse_verdict(raw)
        usage = getattr(resp, "usage", None)
        return {
            "verdict": verdict,
            "reason": reason,
            "raw": raw,
            "model": model,
            "usage": {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            },
        }

    return judge_fn
