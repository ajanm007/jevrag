"""ShadowDecision — observe-only wrapper around any ``Decision`` backend.

v0.2.0 item 4 (shadow / observe mode). The reusable unit from
``rag-gate-py``'s shipped observe mode is the *pattern*, not the code: that
module is a streaming-token interceptor (async SSE frame reassembly, a
lookahead buffer for retracting verdicts mid-stream) — structurally unlike
JevRAG's discrete ``Decision.ask(state, questions) -> DecisionResult``
calls. So this is a fresh implementation of the pattern against JevRAG's
own call shape, not a port.

The real design fork, resolved here rather than defaulted: in
``rag-gate-py``, shadow mode means "evaluate live traffic that would
otherwise flow un-evaluated, without cutting anything." In JevRAG every
``ask()`` call *already* just evaluates — the action lives in caller code,
never in the model — so "evaluate without deciding" is vacuous. The only
thing shadow mode can add here is a **second opinion logged alongside the
first**: run a would-be-different policy (another backend, another
threshold) over the same states, record what it *would* have decided,
never touch what production actually does.

One mechanism serves both uses: wrap the governing backend and the ledger
is an audit trail of production; wrap a challenger driven with the same
states and the ledger is a what-if comparison (the demo in
``scripts/demo_shadow_mode.py`` is the latter — a ``LogprobDecision``
challenger replayed alongside recorded Jev production). Either way the
wrapper's contract is identical: the inner call happens for real, one
JSONL row is appended, the inner ``DecisionResult`` object is returned
**untouched** — "shadow" means "observe," never "modify."

State-logging policy (deliberate, not an afterthought): ``state`` routinely
carries large evidence blobs (full passage texts), so the ledger never
stores raw state. ``summarize_state()`` keeps scalar values verbatim
(strings truncated to ``MAX_STATE_STR`` chars with their full length
noted), reduces mappings per-key, heavily nests nothing, and reduces
sequences to a length plus a short preview. Shape is preserved; bulk is
not. Nothing in the ledger can reconstruct a full evidence set, by design.

Failure policy (also deliberate): observation must never break production,
in either direction. A misconfigured ledger path (uncreatable parent)
raises loudly at construction — that's a programming error, fail fast. A
*run-time* write failure (disk full, path vanished mid-run) never raises
out of ``ask()``: the inner result is still returned, the drop is counted
on ``errors``/``dropped`` (and ``logged`` stops advancing, so gaps in the
file's ``seq`` column are explained, not mysterious). Nothing is silent,
nothing is fatal.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jevrag.decision import Decision, DecisionResult, TypedQuestion

#: Wrapper tag: recorded on every ledger row and exposed as ``backend_name``.
WRAPPER_NAME = "shadow"

#: Longest string value kept verbatim in a ledger row's state summary.
MAX_STATE_STR = 200

#: Sequence items previewed in a state summary (length is always recorded).
STATE_PREVIEW_ITEMS = 3

#: Ledger note stamped on every row: the guarantee, in the file itself.
SHADOW_NOTE = (
    "shadow-observe: the wrapped backend's result was returned unchanged; "
    "this row records what would have been decided, with no production effect."
)


def _json_safe(value: Any) -> Any:
    """Coerce an arbitrary value into JSON-safe data, deterministically.

    The important types (None/bool/numbers/strings/mappings/sequences)
    survive verbatim; anything else becomes its ``str()`` rather than
    exploding ``json.dumps`` mid-ledger-write.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value  # NaN/Inf survive as such; json emits them, loader reads them back
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _summarize_value(value: Any) -> Any:
    """Summarize one state value: shape kept, bulk dropped (see module docstring)."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= MAX_STATE_STR:
            return value
        return {
            "truncated_str": value[:MAX_STATE_STR],
            "full_len": len(value),
        }
    if isinstance(value, Mapping):
        return {str(k): _summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "n": len(value),
            "preview": [_summarize_value(v)
                        for v in list(value)[:STATE_PREVIEW_ITEMS]],
        }
    return {"type": type(value).__name__, "repr": str(value)[:MAX_STATE_STR]}


def summarize_state(state: Any) -> Any:
    """Public state-summary used for ledger rows (tested directly)."""
    return _summarize_value(state)


def _question_row(q: TypedQuestion) -> dict[str, Any]:
    # Questions are code constants (name/kind/instructions/criteria), not
    # user data or evidence bulk — safe to keep verbatim.
    return {
        "name": q.name,
        "kind": q.kind,
        "instructions": q.instructions,
        "criteria": _json_safe(q.criteria),
    }


class ShadowDecision:
    """Observe-only ``Decision`` wrapper: real call, ledger row, unchanged return.

    Args:
        inner: any object with ``ask(state, questions) -> DecisionResult``
            (Jev, ``LogprobDecision``, ``StubDecision``, …). Duck-typed on
            purpose — the ``Decision`` protocol is structural, so the
            wrapper is too.
        ledger_path: JSONL file appended with one row per ``ask()`` call.
            Parent directories are created at construction (loudly — a bad
            path is a programming error, fail fast here, not mid-run).
    """

    backend_name = WRAPPER_NAME

    def __init__(self, inner: Decision, ledger_path: str | Path) -> None:
        if not callable(getattr(inner, "ask", None)):
            raise TypeError(
                "ShadowDecision wraps a Decision backend (something with "
                f"ask(state, questions)); got {type(inner).__name__}."
            )
        self._inner = inner
        self._ledger_path = Path(ledger_path)
        # Fail fast on a misconfigured path: anything else would turn every
        # later call into a counted drop.
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.logged = 0
        self.errors = 0
        self.dropped = 0

    @property
    def inner(self) -> Decision:
        return self._inner

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path

    def _inner_name(self) -> str:
        return str(getattr(self._inner, "backend_name",
                           type(self._inner).__name__))

    def ask(
        self, state: Any, questions: list[TypedQuestion]
    ) -> DecisionResult:
        result = self._inner.ask(state, questions)
        self.calls += 1
        md = result.metadata if isinstance(result.metadata, dict) else {}
        row = {
            "seq": self.calls,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "wrapper": self.backend_name,
            "backend": self._inner_name(),
            "questions": [_question_row(q) for q in questions],
            "result": _json_safe(result.result),
            "confidence": _json_safe(result.confidence),
            "latency_ms": md.get("latency_ms"),
            "input_tokens": md.get("input_tokens"),
            "output_tokens": md.get("output_tokens"),
            "cost_usd": md.get("cost_usd"),
            "state_summary": summarize_state(state),
            "note": SHADOW_NOTE,
        }
        try:
            with open(self._ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False,
                                   allow_nan=True) + "\n")
        except OSError:
            # Observation failed; production must not. Counted, never fatal
            # (see module docstring) — seq gaps in the file are explained by
            # errors/dropped, not mysterious.
            self.errors += 1
            self.dropped += 1
        else:
            self.logged += 1
        return result

    def close(self) -> None:
        """Forward ``close()`` when the inner backend has one; else no-op."""
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
