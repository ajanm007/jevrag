"""Bridge to code and data this project reuses rather than reimplements.

The risk-coverage/AURC math, the EM/F1 evaluator, and the CRC
threshold-selection core are vendored directly into ``jevrag/_vendor/``
(with attribution) — see ``selective()``, ``evaluator()``, and ``crc()``
below — so the package installs and runs standalone with no external
checkout required for that part.

The HotpotQA dataset itself (the frozen 1000-question val/test split,
prebuilt BM25/FAISS indices) is a separate, larger dependency that is
NOT vendored here — it currently lives in a private research checkout.
``rag_gate_root()`` still resolves that data location, overridable via
the RAG_GATE_PATH environment variable. Only code that touches HotpotQA
data needs it; the Decision protocol, every primitive, and the
calibration harness itself do not.

A second, unrelated external root — DocBench/SciFact eval data and
generated Kaggle records — is resolved the same way via
``jevrag_kaggle_root()``, overridable via JEVRAG_KAGGLE_PATH. Only the
`scripts/eval_*_doc*.py` / `scripts/eval_scifact_*.py` / `demo_docbench.py`
runners need it.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

from ._vendor import crc as _crc_module
from ._vendor import evaluator as _evaluator_module
from ._vendor import selective as _selective_module

_DEFAULT_ROOT = Path(r"D:\Research\RAG-Gate")
_DEFAULT_KAGGLE_ROOT = Path(r"D:\JevRAG-kaggle")


def rag_gate_root() -> Path:
    """Root of the RAG-Gate data checkout (env-overridable via RAG_GATE_PATH).

    Only needed for HotpotQA data access (see jevrag/benchmarks/hotpotqa.py).
    """
    return Path(os.environ.get("RAG_GATE_PATH", str(_DEFAULT_ROOT)))


def jevrag_kaggle_root() -> Path:
    """Root of the DocBench/SciFact eval data + generated Kaggle records
    (env-overridable via JEVRAG_KAGGLE_PATH). Only needed by the
    real-document/SciFact eval scripts, not by the installable package.
    """
    return Path(os.environ.get("JEVRAG_KAGGLE_PATH", str(_DEFAULT_KAGGLE_ROOT)))


def selective() -> ModuleType:
    """Risk-coverage / AURC implementation (Geifman & El-Yaniv, Kamath) — vendored."""
    return _selective_module


def evaluator() -> ModuleType:
    """EM/F1 evaluator with SQuAD-style normalization — vendored."""
    return _evaluator_module


def crc() -> ModuleType:
    """CRC threshold selection (Angelopoulos et al. 2022) — vendored."""
    return _crc_module
