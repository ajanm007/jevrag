"""jevrag.benchmarks.hotpotqa — HotpotQA loading and correctness labels.

Data is prepped and split-frozen in the RAG-Gate checkout
(``data/hotpotqa_1000.json``, 1000 questions, stratified val/test, seed-frozen by
``data_loader.py``). No download, no index build on this path.

Correctness labels come from RAG-Gate ``evaluator.py``: EM with SQuAD-style
normalization, plus F1. The binary EM label is what selective.py consumes; F1 is
kept as a secondary measure (HotpotQA convention).
"""

from __future__ import annotations

from pathlib import Path

from .._rag_gate import evaluator as _load_evaluator, rag_gate_root

_eval = _load_evaluator()

# Re-exported unchanged — do not reimplement.
normalize_answer = _eval.normalize_answer
exact_match = _eval.exact_match
f1_score = _eval.f1_score
evaluate_results = _eval.evaluate_results


def dataset_path(name: str = "hotpotqa") -> Path:
    """Path to the frozen dataset JSON in the RAG-Gate checkout."""
    if name != "hotpotqa":
        raise ValueError(
            f"unknown dataset {name!r}; V1 supports only 'hotpotqa' "
            "(a second dataset is a real open item, not in this scope)"
        )
    return rag_gate_root() / "data" / "hotpotqa_1000.json"


def load_questions(split: str | None = None) -> list[dict]:
    """Load HotpotQA questions with their frozen val/test split labels.

    Returns the raw question dicts (id, question, answer, type, level, split, ...).
    Raises if the split label is absent or unexpected — the frozen split is the
    whole point (threshold on val, report on test), so a missing label is a bug,
    not something to work around.
    """
    import json

    path = dataset_path("hotpotqa")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The prepared HotpotQA data ships with the RAG-Gate "
            "checkout; set RAG_GATE_PATH if it lives elsewhere."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions: list[dict] = data["questions"]
    for q in questions:
        if q.get("split") not in ("val", "test"):
            raise ValueError(
                f"question {q.get('id')!r} has split={q.get('split')!r}; "
                "expected the frozen 'val'/'test' labels from data_loader.py"
            )
    if split is not None:
        if split not in ("val", "test"):
            raise ValueError("split must be 'val', 'test', or None")
        questions = [q for q in questions if q["split"] == split]
    return questions


def split_lookup() -> dict[str, str]:
    """question_id -> 'val' | 'test', for joining against decision-path records."""
    return {q["id"]: q["split"] for q in load_questions()}


def label_record(record: dict) -> dict:
    """Attach EM/F1 correctness to one decision-path record (INTERFACE.md shape).

    ``correct`` is the binary EM label selective.py consumes. ``f1`` rides along
    as the secondary HotpotQA measure. Mutates nothing; returns a new dict.
    """
    pred = record.get("prediction", "")
    gold = record.get("gold", "")
    if not isinstance(pred, str):
        pred = str(pred)
    if not isinstance(gold, str):
        gold = str(gold)
    out = dict(record)
    out["correct"] = exact_match(pred, gold)
    out["f1"] = f1_score(pred, gold)
    return out


def label_records(records: list[dict]) -> list[dict]:
    """label_record over a list, preserving order."""
    return [label_record(r) for r in records]
