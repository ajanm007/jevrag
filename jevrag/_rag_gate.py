"""Single point of contact with the RAG-Gate research codebase.

Do NOT reimplement risk-coverage/AURC or EM/F1 — import them from
``D:\\Research\\RAG-Gate\\src\\selective.py`` and ``src\\evaluator.py``. The Rust
scorer the PRD mentions is an axum HTTP proxy with no Python bindings; selective.py
is the same math, in-process, already unit-tested (documented PRD correction).

The repo location is overridable via the RAG_GATE_PATH environment variable so this
keeps working on machines where the checkout lives elsewhere.

This is the ONLY module in jevrag that knows where RAG-Gate lives.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

_DEFAULT_ROOT = Path(r"D:\Research\RAG-Gate")


def rag_gate_root() -> Path:
    """Root of the RAG-Gate checkout (env-overridable via RAG_GATE_PATH)."""
    return Path(os.environ.get("RAG_GATE_PATH", str(_DEFAULT_ROOT)))


def _load(module_name: str, relpath: str) -> ModuleType:
    root = rag_gate_root()
    path = root / relpath
    if not path.exists():
        raise ImportError(
            f"RAG-Gate module not found at {path}. "
            "Set the RAG_GATE_PATH environment variable to the RAG-Gate checkout. "
            "These modules are reused, not vendored."
        )
    # src/ modules do `from config.settings import ...` style imports, so the
    # repo root must be importable before exec_module runs.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selective() -> ModuleType:
    """RAG-Gate's risk-coverage / AURC implementation (Geifman & El-Yaniv, Kamath)."""
    return _load("rag_gate_selective", "src/selective.py")


def evaluator() -> ModuleType:
    """RAG-Gate's EM/F1 evaluator with SQuAD-style normalization."""
    return _load("rag_gate_evaluator", "src/evaluator.py")
