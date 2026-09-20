"""Cache-trust primitive (extraction, not invention).

Question: given a semantic-cache hit for the current query, is the cached
answer safe to serve, or should the system regenerate a fresh answer?

Extracted from the real prior art — ``agentic-chat`` PR #52
(github.com/shubho0908/agentic-chat, ``lib/jev/cacheGate.ts``), ported to
the ``Decision`` protocol unmodified. Port notes (what was kept verbatim,
what was decided):

- State: the real 8-field structural struct — similarity score, the cache's
  own cutoff, score margin, entry age, TTL, conversation scoping, and the
  query/answer lengths. The cached question and answer content NEVER travel
  to the evaluation call (deliberate in the original: privacy/cost). This
  makes cache-trust the first primitive whose ``state`` is non-text
  structural metadata rather than a text blob.
- Two questions per ``ask()`` call (also a first for this project):
  ``serve_from_cache`` ("safe to serve instead of regenerating?") and
  ``staleness_risk`` ("risk of being outdated/mismatched?"), with the
  original instructions/criteria text. Verdict rule verbatim:
  serve_conf >= 0.5 AND stale_risk < 0.5 — one confident signal either way
  is enough to veto the serve.
- Fail-open: the original serves from cache on provider errors AND on
  invalid/mistyped answers (a Jev outage must never convert every hit into
  a miss and overload the generator). Mirrored here, as an explicit call:
  availability over staleness-avoidance during an outage, with the
  asymmetry stated — a fail-open serve can hand back a stale answer, which
  is the cheaper failure only while the alternative is no answers at all.
  Fallback serves are tagged ``fallback_used=True`` and carry
  ``confidence=None`` (NaN) so the harness excludes rather than scores
  them — no fabricated confidence.

License disclosure: ``agentic-chat`` ships NO license (no LICENSE file,
no package.json license field, GitHub license API returns null — checked
2026-09-20, same situation as DocBench earlier this session). Public repo,
research/evaluation use, disclosed plainly. Only the decision logic
(questions, state shape, verdict rule) is ported — no code copied.

Ground-truth note: no (cache hit, was-it-safe) dataset exists. First-cut
evaluation constructs the cache-hit scenario (paraphrase vs
different-question queries, real embedding similarities, assigned
ages/scoping) but keeps the label real: EM of the cached answer against
the current query's gold. See scripts/eval_cache_trust.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from jevrag.decision import Decision, DecisionResult, TypedQuestion

#: Policy tag for records emitted by the cache-trust check.
POLICY_CACHE_TRUST = "cache_trust_jev"

#: Question names — kept identical to the original's answer keys.
SERVE_QUESTION_NAME = "serve_from_cache"
STALENESS_QUESTION_NAME = "staleness_risk"

#: Verdict thresholds, verbatim from shouldServeCachedAnswer.
DEFAULT_SERVE_THRESHOLD = 0.5
DEFAULT_STALENESS_THRESHOLD = 0.5

#: Scenario default for "the cache's own configured cutoff" (the original
#: reads this from cache config; our eval fixes it and discloses it).
DEFAULT_SIMILARITY_THRESHOLD = 0.70

#: Scenario default TTL, seconds (original reads this from cache config).
DEFAULT_TTL_SECONDS = 3600


@dataclass
class CacheTrustState:
    """Structural cache-hit state. Numbers and one boolean — no text.

    The cached question and answer content never leave for the evaluation
    call, by design (ported from the original).
    """

    similarity_score: float
    similarity_threshold: float
    score_margin: float
    cache_age_seconds: float
    cache_ttl_seconds: float
    entry_scoped_to_conversation: bool
    query_length_chars: int
    answer_length_chars: int

    @classmethod
    def make(
        cls,
        *,
        similarity_score: float,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        cache_age_seconds: float,
        cache_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        entry_scoped_to_conversation: bool,
        query_length_chars: int,
        answer_length_chars: int,
    ) -> "CacheTrustState":
        """Build a state, deriving margin as score minus threshold."""
        return cls(
            similarity_score=float(similarity_score),
            similarity_threshold=float(similarity_threshold),
            score_margin=float(similarity_score) - float(similarity_threshold),
            cache_age_seconds=float(cache_age_seconds),
            cache_ttl_seconds=float(cache_ttl_seconds),
            entry_scoped_to_conversation=bool(entry_scoped_to_conversation),
            query_length_chars=int(query_length_chars),
            answer_length_chars=int(answer_length_chars),
        )

    def to_jev_state(self) -> dict[str, Any]:
        """Render as a plain JSON-ish dict of numbers/bool for the Jev call."""
        return {
            "similarityScore": self.similarity_score,
            "similarityThreshold": self.similarity_threshold,
            "scoreMargin": self.score_margin,
            "cacheAgeSeconds": self.cache_age_seconds,
            "cacheTtlSeconds": self.cache_ttl_seconds,
            "entryScopedToConversation": self.entry_scoped_to_conversation,
            "queryLengthChars": self.query_length_chars,
            "answerLengthChars": self.answer_length_chars,
        }


def cache_trust_questions() -> list[TypedQuestion]:
    """The two questions, instructions/criteria ported from cacheGate.ts."""
    return [
        TypedQuestion(
            name=SERVE_QUESTION_NAME,
            kind="boolean",
            instructions=(
                "Is this semantic cache hit safe to serve instead of "
                "generating a fresh answer?"
            ),
            criteria={
                "true": ("The similarity score clears the threshold with "
                         "margin and the entry is fresh enough that the "
                         "stored answer still applies."),
                "false": ("The similarity is marginal or the entry is old "
                          "enough that the stored answer may no longer fit "
                          "the query."),
            },
        ),
        TypedQuestion(
            name=STALENESS_QUESTION_NAME,
            kind="boolean",
            instructions=(
                "Does serving this cached answer risk being outdated or "
                "mismatched for the current query?"
            ),
            criteria={
                "true": ("Age, thin score margin, or loose scoping make a "
                         "stale or off-target answer plausible."),
                "false": ("The entry is recent, strongly matched, and "
                          "scoped tightly enough to trust."),
            },
        ),
    ]


def check_cache_trust(
    state: CacheTrustState,
    decision: Decision,
) -> DecisionResult:
    """One-shot cache-trust check. One ask() call, TWO questions.

    Uses the Decision protocol and any backend implementing it, unmodified.
    Multi-question calls carry per-question answers/confidences as dicts
    (see DecisionResult); the serve/regenerate action is the caller's, via
    ``action_for`` after ``extract_decision``.
    """
    return decision.ask(state.to_jev_state(), cache_trust_questions())


def extract_decision(result: DecisionResult) -> dict[str, float] | None:
    """Port of mapJevCacheGateResult: dict confidences -> decision, else None.

    Returns None when either answer is missing or mistyped; callers treat
    None as invalid and keep the production behavior (serve, fail-open).
    """
    conf = result.confidence
    if not isinstance(conf, dict):
        return None
    serve = conf.get(SERVE_QUESTION_NAME)
    stale = conf.get(STALENESS_QUESTION_NAME)
    if isinstance(serve, bool) or not isinstance(serve, (int, float)):
        return None
    if isinstance(stale, bool) or not isinstance(stale, (int, float)):
        return None
    serve_f, stale_f = float(serve), float(stale)
    if math.isnan(serve_f) or math.isnan(stale_f):
        return None
    return {"serve_from_cache": serve_f, "staleness_risk": stale_f}


def should_serve(
    decision: dict[str, float],
    *,
    serve_threshold: float = DEFAULT_SERVE_THRESHOLD,
    staleness_threshold: float = DEFAULT_STALENESS_THRESHOLD,
) -> bool:
    """Port of shouldServeCachedAnswer: confident-safe AND low-staleness."""
    return (decision["serve_from_cache"] >= serve_threshold
            and decision["staleness_risk"] < staleness_threshold)


def action_for(
    decision: dict[str, float] | None,
    *,
    serve_threshold: float = DEFAULT_SERVE_THRESHOLD,
    staleness_threshold: float = DEFAULT_STALENESS_THRESHOLD,
) -> str:
    """The action half: verdict dict (or None on invalid) -> serve/regenerate.

    None (invalid/mistyped answers) fails open to "serve", mirroring the
    original's invalid-response path.
    """
    if decision is None:
        return "serve"
    return ("serve" if should_serve(decision, serve_threshold=serve_threshold,
                                    staleness_threshold=staleness_threshold)
            else "regenerate")


def make_cache_record(
    *,
    query_id: str,
    query: str,
    cached_answer: str,
    confidence: float | None,
    staleness_risk: float | None,
    action: str,
    latency_ms: float,
    input_tokens: int,
    fallback_used: bool = False,
    policy: str = POLICY_CACHE_TRUST,
) -> dict[str, Any]:
    """Handoff record for one cache-hit check — one-shot shape, no rounds."""
    if action not in ("serve", "regenerate"):
        raise ValueError(f"action must be 'serve' or 'regenerate', got {action!r}")
    return {
        "query_id": query_id,
        "query": query,
        "cached_answer": cached_answer,
        "confidence": (None if confidence is None
                       or (isinstance(confidence, float)
                           and math.isnan(confidence)) else float(confidence)),
        "staleness_risk": staleness_risk,
        "action": action,
        "latency_ms": float(latency_ms),
        "input_tokens": int(input_tokens),
        "fallback_used": bool(fallback_used),
        "policy": policy,
    }


def check_cache(
    *,
    query_id: str,
    query: str,
    cached_answer: str,
    state: CacheTrustState,
    decision: Decision,
    serve_threshold: float = DEFAULT_SERVE_THRESHOLD,
    staleness_threshold: float = DEFAULT_STALENESS_THRESHOLD,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check one cache hit; return (record, trace).

    Fail-open (mirroring the original): provider errors and invalid/mistyped
    answers serve from cache with ``fallback_used=True`` and no confidence,
    rather than converting the hit into a miss. The trace is diagnostic
    only (Jev latency/tokens for the single call).
    """
    wall_start = time.perf_counter()
    try:
        result: DecisionResult = check_cache_trust(state, decision)
    except Exception:
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        record = make_cache_record(
            query_id=query_id, query=query, cached_answer=cached_answer,
            confidence=None, staleness_risk=None, action="serve",
            latency_ms=wall_ms, input_tokens=0, fallback_used=True)
        return record, {"latency_ms": wall_ms, "input_tokens": 0,
                        "wall_ms": wall_ms, "fallback": "exception"}
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    verdict = extract_decision(result)
    record = make_cache_record(
        query_id=query_id, query=query, cached_answer=cached_answer,
        confidence=(verdict["serve_from_cache"] if verdict is not None
                    else None),
        staleness_risk=(verdict["staleness_risk"] if verdict is not None
                        else None),
        action=action_for(verdict, serve_threshold=serve_threshold,
                          staleness_threshold=staleness_threshold),
        latency_ms=float(result.metadata.get("latency_ms", 0.0)),
        input_tokens=int(result.metadata.get("input_tokens", 0)),
        fallback_used=(verdict is None))
    trace = {"latency_ms": record["latency_ms"],
             "input_tokens": record["input_tokens"],
             "wall_ms": wall_ms,
             "fallback": "invalid-response" if verdict is None else None}
    return record, trace
