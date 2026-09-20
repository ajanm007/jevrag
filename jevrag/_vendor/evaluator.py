"""
Evaluator — Exact Match / F1
==============================
Exact Match (EM) and F1 score computation with SQuAD-style
text normalization.

Vendored from a separate, private research project (RAG-Gate) with the
author's own permission, so this package installs and runs standalone.
Unchanged from the original except for this header and dropping the
standalone CLI entry point (not needed here) — see
jevrag/benchmarks/hotpotqa.py for how it's used.
"""

from collections import Counter
import re
import string


def normalize_answer(s: str) -> str:
    """
    SQuAD-style answer normalization.
    Lowercase, remove articles, punctuation, and extra whitespace.
    """
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(c for c in s if c not in string.punctuation)
    # Collapse whitespace
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> int:
    """Return 1 if normalized prediction matches gold answer exactly."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """
    Token-level F1 score between prediction and gold answer.
    Returns 0.0 if no token overlap, 1.0 if perfect match.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_results(results: list[dict]) -> dict:
    """
    Compute aggregate metrics over a list of results.

    Args:
        results: List of dicts, each must have:
            - 'prediction': model's answer string
            - 'gold': ground truth answer string
            - 'type': question type ('bridge' or 'comparison')

    Returns:
        Dict with EM and F1 scores (overall + per question type)
    """
    em_scores = []
    f1_scores = []

    # Per-type tracking
    type_em: dict[str, list[int]] = {}
    type_f1: dict[str, list[float]] = {}

    for r in results:
        pred: str = r.get("prediction", "")
        gold: str = r.get("gold", "") or r.get("answer", "")
        if not isinstance(pred, str):
            pred = str(pred)
        if not isinstance(gold, str):
            gold = str(gold)

        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)

        em_scores.append(em)
        f1_scores.append(f1)

        # Track by question type
        q_type = r.get("type", "unknown")
        if q_type not in type_em:
            type_em[q_type] = []
            type_f1[q_type] = []
        type_em[q_type].append(em)
        type_f1[q_type].append(f1)

    # Compute averages
    by_type: dict[str, dict[str, float | int]] = {}
    for q_type in sorted(type_em.keys()):
        by_type[q_type] = {
            "em": sum(type_em[q_type]) / max(len(type_em[q_type]), 1),
            "f1": sum(type_f1[q_type]) / max(len(type_f1[q_type]), 1),
            "count": len(type_em[q_type]),
        }

    metrics = {
        "overall": {
            "em": sum(em_scores) / max(len(em_scores), 1),
            "f1": sum(f1_scores) / max(len(f1_scores), 1),
            "count": len(em_scores),
        },
        "by_type": by_type,
    }

    return metrics
