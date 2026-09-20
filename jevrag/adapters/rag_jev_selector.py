"""Adapter: `rag-jev` (PyPI, v0.2.0, MIT) wrapped for our calibration harness.

`rag-jev` is real prior art — shipped, evaluated on SciFact by its author.
Per PRD §5/§10 it is WRAPPED, not rebuilt: this module depends on the real
package (pinned) and maps its output onto per-passage records our harness
consumes as (confidence, correct) pairs. No `rag-jev` code is vendored,
forked, or reimplemented here.

Verified interface (read from installed 0.2.0 source, not guessed):
- `Jev(api_key=..., model="jev-latest")` — keyword-only key; one Noul
  ("relevant?") per document against state {query, passage}.
- `ContextSelector(provider, timeout_ms=5000, max_concurrency=8,
  on_error="passthrough")`, `await selector.select_request(SelectRequest(
  query, documents=[Document(id, text)], mode="filter", min_relevance=...))`.
- `result.decisions[i]`: relevance (0–1 float or None), selected (bool or
  None), reason (retained|below_threshold|beyond_top_n|beyond_token_budget|
  bypassed|pinned|group_retained), per-doc input/output tokens.
- `result.status`: applied|shadow|bypassed; `result.usage`: aggregate tokens.

Honesty rules at this boundary:
- The adapter is SYNC; `rag-jev` is async. `asyncio.run()` lives inside the
  entry point — `rag-jev` is called as designed, not rewritten.
- `status == "bypassed"` (their error/timeout path) surfaces as
  confidence None records — never a fabricated score. Same discipline as
  `parse_verdict → None` and the CLI's unreportable omission.
- `min_relevance` (required by their validator for filter mode) is an
  operating point, default 0.5 (neutral midpoint), documented here and
  overridable — not buried.
- Our `JevDecision` is NOT involved: `rag-jev` makes its own Jev calls
  through its own provider. Cost below is genuinely theirs.
"""

from __future__ import annotations

import asyncio
from typing import Any

#: Policy tag marking records produced through this adapter (never JevRAG-native).
POLICY_RAG_JEV = "rag_jev_adapter"

#: Operating-point default for filter mode. Their validator requires a
#: threshold; 0.5 is the neutral midpoint of a 0–1 relevance probability.
#: Any reported number is conditional on this choice — same status as our
#: own SUFFICIENCY_THRESHOLD.
DEFAULT_MIN_RELEVANCE = 0.5


def select_passages(
    question: str,
    passages: list[dict[str, Any]],
    *,
    api_key: str,
    model: str = "jev-latest",
    mode: str = "filter",
    min_relevance: float | None = DEFAULT_MIN_RELEVANCE,
    timeout_ms: float = 5000,
    max_concurrency: int = 8,
) -> dict[str, Any]:
    """Score one question's candidate passages via `rag-jev`, synchronously.

    Args:
        question: the query string.
        passages: list of ``{"id": str, "text": str}`` (ids unique).
        api_key: Jev key (explicit — never resolved from ambient env here,
            so runs can't silently pick up the wrong account's key).
        min_relevance: filter threshold (their validator requires it unless
            mode == "rerank").

    Returns ``{"records": [...], "usage": {...}, "status": str,
    "elapsed_ms": float, "models": [...]}``. Each record carries
    ``doc_id``, ``confidence`` (= their relevance, None on bypass),
    ``selected``, ``reason``, ``input_tokens``/``output_tokens``,
    ``model``, and ``policy``. A bypassed run yields all-None confidences —
    measurable absence, not invented signal.
    """
    from rag_jev import ContextSelector, Document, Jev, SelectRequest

    request = SelectRequest(
        query=question,
        documents=[Document(id=p["id"], text=p["text"]) for p in passages],
        mode=mode,  # type: ignore[arg-type]
        min_relevance=min_relevance,
    )
    provider = Jev(api_key=api_key, model=model)
    selector = ContextSelector(provider, timeout_ms=timeout_ms,
                               max_concurrency=max_concurrency)

    async def _run():
        try:
            return await selector.select_request(request)
        finally:
            await provider.aclose()

    result = asyncio.run(_run())
    records = [
        {
            "doc_id": d.id,
            "confidence": d.relevance,
            "selected": d.selected,
            "reason": d.reason,
            "input_tokens": d.input_tokens or 0,
            "output_tokens": d.output_tokens or 0,
            "model": d.model,
            "policy": POLICY_RAG_JEV,
        }
        for d in result.decisions
    ]
    return {
        "records": records,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "completed_documents": result.usage.completed_documents,
        },
        "status": result.status,
        "error_code": result.error_code,
        "elapsed_ms": result.elapsed_ms,
        "models": result.models,
    }
