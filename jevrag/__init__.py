"""JevRAG package — a decision substrate for RAG pipelines.

Five decisions are built on the shared ``Decision`` abstraction and
calibration harness: evidence-sufficiency, chunk-boundary,
context-selection, answer-abstain, and cache-trust. Jev is the first
backend wired up; the abstraction itself is the point, not any one
decision or backend. See README.md for what's built and evaluated.
"""

__version__ = "0.1.0"

from jevrag.decision import (
    JEV_MODEL_DEFAULT,
    JEV_PRICE_PER_M_INPUT_TOKENS,
    Decision,
    DecisionResult,
    JevDecision,
    StubDecision,
    TypedQuestion,
)

__all__ = [
    "JEV_MODEL_DEFAULT",
    "JEV_PRICE_PER_M_INPUT_TOKENS",
    "Decision",
    "DecisionResult",
    "JevDecision",
    "StubDecision",
    "TypedQuestion",
    "__version__",
]
