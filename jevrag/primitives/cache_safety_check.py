"""Independent cache-safety check — a design alternative, not a port.

Cold-problem design: given a semantic-cache hit, is it safe to serve the
cached answer, or should the system regenerate? Built without reading the
existing ``cache_trust`` implementation, so the design choices below are
genuinely independent of it.

The one hard constraint shapes everything: the cached question and answer
**text must never be sent to the judge call**. So the state is purely
structural — numbers and categories that describe the match, never its
content:

- ``match_similarity``: the embedding cosine similarity that produced the
  hit, in [0, 1]. The retrieval signal, and on its own a weak one.
- ``similarity_margin``: match_similarity minus the system's serve
  threshold — how far above the bar this hit sits. A hit at 0.94 when the
  bar is 0.90 is a different animal than a hit at 0.905.
- ``entry_age_hours`` / ``ttl_hours``: staleness as a fraction, not a raw
  clock reading (a 23-hour-old hit under a 24h TTL is nearly expired).
- ``regeneration_cost_tokens``: what regeneration would cost. A cheap regen
  means the rational bar for serving should be higher — this lets the judge
  weigh risk against stakes with full context.
- ``prior_successful_serves``: how many times this entry already served
  without complaint. Zero means untested; a large count is real evidence.

Question shape: two Booleans in ONE ``ask()`` call, because the call is the
unit that shares the state most efficiently:

- ``safe_to_serve``: is serving this cached answer acceptable given the
  signals? (positive framing)
- ``mismatch_risk``: is there a concrete reason to suspect the cached answer
  does not apply here — a weak/near-threshold match, a stale or untested
  entry? (risk framing)

The point of asking both framings is that they are not logical negations of
each other — a judge can be unsure in both directions at once (both ~0.5:
"nothing says safe, nothing says risky, match mediocre") and that pattern
is itself informative. The verdict rule uses both:

    serve iff P(safe) >= threshold AND P(risk) < risk_ceiling

A candidate passes only if the judge affirmatively endorses serving *and*
does not flag risk. Thresholds live with the caller, never in the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jevrag.decision import Decision, DecisionResult, TypedQuestion

#: Policy tag for records emitted by this decision.
POLICY_CACHE_SAFETY_CHECK = "cache_safety_check"

#: Names of the two Boolean questions asked per cache hit.
SAFE_QUESTION_NAME = "safe_to_serve"
RISK_QUESTION_NAME = "mismatch_risk"

#: Default verdict thresholds. Presentation defaults only — tune on labeled
#: data for real use. ``threshold`` gates the positive endorsement;
#: ``risk_ceiling`` gates the negative flag.
DEFAULT_THRESHOLD = 0.5
DEFAULT_RISK_CEILING = 0.5


@dataclass
class CacheHitState:
    """Structural signals about one semantic-cache hit. No content, ever.

    ``match_similarity`` is the cosine similarity [0, 1] that produced the
    hit; ``serve_threshold`` is the system's own bar for serving; the margin
    between them is exposed explicitly because distance-above-bar is a
    different fact than the raw score. ``entry_age_hours``/``ttl_hours`` pin
    staleness as a fraction; ``regeneration_cost_tokens`` states the stakes
    of being wrong; ``prior_successful_serves`` is the entry's track record.
    """

    match_similarity: float
    serve_threshold: float
    entry_age_hours: float
    ttl_hours: float
    regeneration_cost_tokens: int
    prior_successful_serves: int
    hit_id: str = ""

    def to_jev_state(self) -> dict[str, Any]:
        """Render as a plain dict of numbers. Text-free by construction —
        there is no content field to leak, so no redaction step can fail."""
        sim = float(self.match_similarity)
        bar = float(self.serve_threshold)
        age = max(0.0, float(self.entry_age_hours))
        ttl = max(float(self.ttl_hours), 1e-9)
        return {
            "match_similarity": round(sim, 4),
            "similarity_margin_above_threshold": round(sim - bar, 4),
            "age_fraction_of_ttl": round(min(1.0, age / ttl), 4),
            "ttl_expired": age >= float(self.ttl_hours),
            "regeneration_cost_tokens": int(self.regeneration_cost_tokens),
            "prior_successful_serves": int(self.prior_successful_serves),
            "hit_id": str(self.hit_id),
        }


def safe_question() -> TypedQuestion:
    """Positive framing: is serving acceptable on these signals?"""
    return TypedQuestion(
        name=SAFE_QUESTION_NAME,
        kind="boolean",
        instructions=(
            "A semantic-cache hit is described by structural signals only "
            "(no question or answer text is available to you, by design): "
            "match_similarity (embedding cosine, 0-1), "
            "similarity_margin_above_threshold (how far the hit sits above "
            "the system's own serve bar — positive means it cleared it), "
            "age_fraction_of_ttl (0 = fresh, 1 = at expiry; ttl_expired "
            "flags past-expiry), regeneration_cost_tokens (what a fresh "
            "regeneration would cost), and prior_successful_serves (how many "
            "times this entry served without complaint). Is serving the "
            "cached answer acceptable given these signals? Answer true when "
            "the match is comfortably above bar on a fresh, tested entry, "
            "false when the match is marginal, the entry is stale or "
            "untested, or the combination otherwise looks unsafe."
        ),
        criteria={
            "true": "Safe to serve the cached answer.",
            "false": "Not clearly safe; regenerate instead.",
        },
    )


def risk_question() -> TypedQuestion:
    """Risk framing: is there concrete reason to suspect a mismatch?"""
    return TypedQuestion(
        name=RISK_QUESTION_NAME,
        kind="boolean",
        instructions=(
            "Same structural cache-hit signals, read adversarially: "
            "match_similarity, similarity_margin_above_threshold (near zero "
            "or negative means the hit barely cleared or missed the system's "
            "own bar), age_fraction_of_ttl and ttl_expired (stale entries "
            "drift from the truth), regeneration_cost_tokens (cheap "
            "regeneration makes any doubt decisive), and "
            "prior_successful_serves (zero means this entry has never proven "
            "itself). Is there a concrete reason to suspect the cached "
            "answer does not apply here? Answer true when the signals show a "
            "weak match, a stale or untested entry, or a combination that "
            "smells wrong — false when nothing in the signals gives pause."
        ),
        criteria={
            "true": "Mismatch risk flagged; regenerate.",
            "false": "No concrete risk visible in the signals.",
        },
    )


def decide_hit(
    state: CacheHitState,
    decision: Decision,
) -> DecisionResult:
    """One cache hit, ONE ask() call, two Boolean questions.

    Returns the multi-question result: ``confidence`` is a dict
    ``{"safe_to_serve": ..., "mismatch_risk": ...}``. The verdict is the
    caller's, via :func:`verdict_for`.
    """
    return decision.ask(state.to_jev_state(), [safe_question(), risk_question()])


def decide_hit_single(
    state: CacheHitState,
    decision: Decision,
) -> DecisionResult:
    """Ablation: the positive question only, one call. Used in the eval to
    test whether the second (risk-framed) question earns its keep."""
    return decision.ask(state.to_jev_state(), [safe_question()])


def verdict_for(
    safe_confidence: float,
    risk_confidence: float,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    risk_ceiling: float = DEFAULT_RISK_CEILING,
) -> str:
    """The action half: two confidences → code-side serve/regenerate verdict.

    Serve requires BOTH an affirmative endorsement AND no flagged risk. Any
    other pattern — low endorsement, flagged risk, or both — regenerates.
    """
    if (float(safe_confidence) >= threshold
            and float(risk_confidence) < risk_ceiling):
        return "serve"
    return "regenerate"


def make_safety_record(
    *,
    hit_id: str,
    safe_confidence: float,
    risk_confidence: float,
    verdict: str,
    latency_ms: float,
    input_tokens: int,
    policy: str = POLICY_CACHE_SAFETY_CHECK,
) -> dict[str, Any]:
    """Handoff record for one judged hit — verdict plus both confidences.

    Guards at the seam: both confidences must be probabilities in [0, 1] and
    the verdict one of the two allowed values.
    """
    if verdict not in ("serve", "regenerate"):
        raise ValueError(f"verdict must be 'serve' or 'regenerate', "
                         f"got {verdict!r}")
    for name, conf in (("safe", safe_confidence), ("risk", risk_confidence)):
        c = float(conf)
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"{name} confidence must be a probability in "
                             f"[0, 1]; got {c!r}")
    return {
        "hit_id": str(hit_id),
        "safe_confidence": float(safe_confidence),
        "risk_confidence": float(risk_confidence),
        "verdict": verdict,
        "latency_ms": float(latency_ms),
        "input_tokens": int(input_tokens),
        "policy": policy,
    }

