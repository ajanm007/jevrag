"""jevrag.backends — swappable Decision backends beyond Jev.

``logprob_decision.LogprobDecision`` is the second real backend: a
zero-marginal-cost generator-logprob confidence signal implementing the
``Decision`` protocol unmodified. See its module docstring for the state
contract and the confidence-mapping justification.

``shadow.ShadowDecision`` is the observe-only wrapper (v0.2.0 item 4): it
runs any backend for real, appends one JSONL ledger row per call, and
returns the inner result untouched — an audit trail when wrapped around
the governor, a what-if comparison when wrapped around a challenger.
"""

from jevrag.backends.logprob_decision import LogprobDecision
from jevrag.backends.shadow import ShadowDecision, summarize_state

__all__ = ["LogprobDecision", "ShadowDecision", "summarize_state"]
