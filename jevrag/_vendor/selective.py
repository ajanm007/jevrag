"""
Selective Prediction / Risk-Coverage
=====================================
Risk-coverage formalism for abstention gates, following Geifman & El-Yaniv
(NeurIPS 2017) and Kamath et al. (ACL 2020).

Vendored from a separate, private research project (RAG-Gate) with the
author's own permission, so this package installs and runs standalone.
Unchanged from the original except for this header — see
jevrag/eval/calibration.py for how it's used here.

Setup. Each question i has a binary correctness label c_i in {0,1} (EM, or
F1-thresholded) and a gate signal score s_i. For a threshold T, the ANSWERED set
is A(T) = { i : s_i >= T }. Higher score = more confident = more likely answered.

Abstention accounting rule (explicit, per Kamath): abstained questions are
EXCLUDED from the selective-accuracy numerator and denominator. They lower
coverage but never count as correct or wrong. Selective accuracy/risk are
computed over ANSWERED questions only.

    coverage(T)  = |A(T)| / N
    sel_acc(T)   = mean(c_i for i in A(T))          (undefined if A(T) empty)
    sel_risk(T)  = 1 - sel_acc(T)
    AURC         = integral of risk d(coverage)  (lower is better)  -- primary metric

This module is pure-Python + numpy, no I/O, so it is unit-tested on synthetic
data.
"""

from __future__ import annotations

import numpy as np

# Convention: at threshold T, answer iff score >= T. Sweeping T from low->high
# moves coverage from 1.0 down to 0.0.


def risk_coverage_curve(
    scores: np.ndarray,
    correct: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Compute the risk-coverage curve by sweeping thresholds over `scores`.
    Runs natively in O(N log N) using sorting.
    Handles NaN correctly (NaN scores are never answered; NaN labels lower
    coverage if answered, but don't count towards selective accuracy).

    Args:
        scores:     gate signal per question, shape (N,). Higher = more confident.
        correct:    binary correctness c_i in {0,1}, shape (N,).
        thresholds: thresholds to evaluate. If None, uses each unique score
                    (the canonical step-curve: every distinct coverage level).

    Returns dict of equal-length arrays, sorted by ascending coverage:
        threshold, coverage, selective_accuracy, selective_risk, n_answered
    Threshold rows with an empty valid answered set are dropped.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("empty scores")
    if len(correct) != n:
        raise ValueError("scores and correct must have equal length")

    valid_scores_mask = ~np.isnan(scores)
    clean_scores = scores[valid_scores_mask]
    clean_correct = correct[valid_scores_mask]

    order = np.argsort(clean_scores)
    asc_scores = clean_scores[order]
    asc_correct = clean_correct[order]

    valid_correct = ~np.isnan(asc_correct)
    asc_correct_zeroed = np.where(valid_correct, asc_correct, 0.0)
    asc_valid_count = np.where(valid_correct, 1.0, 0.0)

    cum_correct = np.concatenate([[0.0], np.cumsum(asc_correct_zeroed)])
    cum_valid = np.concatenate([[0.0], np.cumsum(asc_valid_count)])

    if thresholds is None:
        if len(clean_scores) == 0:
            t_vals = np.array([-np.inf])
        else:
            t_vals = np.concatenate([[-np.inf], np.unique(clean_scores)])
    else:
        t_vals = np.asarray(thresholds, dtype=float)

    idx_less = np.searchsorted(asc_scores, t_vals, side='left')

    n_ans = len(asc_scores) - idx_less
    valid_ans = cum_valid[-1] - cum_valid[idx_less]
    acc_sum = cum_correct[-1] - cum_correct[idx_less]

    # Drop thresholds that don't have any valid labeled answers
    valid_mask = valid_ans > 0
    t_vals = t_vals[valid_mask]
    n_ans = n_ans[valid_mask]
    valid_ans = valid_ans[valid_mask]
    acc_sum = acc_sum[valid_mask]

    cov = n_ans / n  # Global coverage denominator is ALWAYS N
    acc = acc_sum / valid_ans
    risk = 1.0 - acc

    sort_idx = np.argsort(cov)
    return {
        "threshold": t_vals[sort_idx],
        "coverage": cov[sort_idx],
        "selective_accuracy": acc[sort_idx],
        "selective_risk": risk[sort_idx],
        "n_answered": n_ans[sort_idx],
    }


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """
    Area Under the Risk-Coverage curve (lower = better). Integrates selective
    risk over coverage in (0, 1] using the trapezoid rule on the full step curve
    (every unique score). This is the standard AURC; it rewards a signal that
    keeps risk low while coverage is high.
    """
    curve = risk_coverage_curve(scores, correct, thresholds=None)
    cov = curve["coverage"]
    risk = curve["selective_risk"]
    if len(cov) < 2:
        # Degenerate: only one coverage level (e.g. all scores equal).
        return float(risk[0]) if len(risk) else float(1.0 - np.mean(correct))
    return float(np.trapz(risk, cov))


def coverage_at_accuracy(
    scores: np.ndarray, correct: np.ndarray, target_accuracy: float
) -> float:
    """
    Maximum coverage achievable while selective accuracy >= target_accuracy.
    Returns 0.0 if no threshold reaches the target. (Higher = better.)
    """
    curve = risk_coverage_curve(scores, correct, thresholds=None)
    ok = curve["selective_accuracy"] >= target_accuracy
    if not ok.any():
        return 0.0
    return float(curve["coverage"][ok].max())


def accuracy_at_coverage(
    scores: np.ndarray, correct: np.ndarray, target_coverage: float
) -> float:
    """
    Selective accuracy at the smallest coverage that is >= target_coverage.
    (i.e. answer at least target_coverage fraction, then read off accuracy.)
    Returns the base accuracy (answer-all) if no threshold reaches the target.
    """
    curve = risk_coverage_curve(scores, correct, thresholds=None)
    ok = curve["coverage"] >= target_coverage
    if not ok.any():
        return float(np.mean(correct))
    # Among coverages >= target, the highest selective accuracy is at the
    # smallest such coverage; pick the row with min coverage among feasible.
    feasible_cov = curve["coverage"][ok]
    idx = np.argmin(feasible_cov)
    return float(curve["selective_accuracy"][ok][idx])


def threshold_for_coverage(scores: np.ndarray, target_coverage: float) -> float:
    """
    The score threshold that yields (approximately) `target_coverage` on these
    scores -- used to pick an operating point on VAL, then apply it to TEST.
    Answer the top `target_coverage` fraction by score.
    NaN scores are ignored when finding the quantile, but properly accounted
    for in the target_coverage denominator.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if target_coverage >= 1.0:
        return float(-np.inf)
    if target_coverage <= 0.0:
        return float(np.inf)

    valid_scores = scores[~np.isnan(scores)]
    n_valid = len(valid_scores)

    target_count = target_coverage * n
    if target_count > n_valid or n_valid == 0:
        return float(-np.inf)

    q = 1.0 - target_count / n_valid
    return float(np.quantile(valid_scores, q))


def oracle_aurc(correct: np.ndarray) -> float:
    """
    Oracle lower bound on AURC: a perfect signal abstains on all wrong answers
    before any correct one. Using c_i itself as the score gives this envelope.
    """
    correct = np.asarray(correct, dtype=float)
    # Score = correctness (+ tiny tie-break so correct sort above wrong).
    return aurc(correct, correct)
