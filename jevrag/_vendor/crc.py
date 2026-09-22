"""
Conformal Risk Control (CRC) Threshold Selection
=================================================
CRC-in-expectation on a monotonized selective-risk surrogate, following
Angelopoulos et al. (2022). Turns "pick a threshold that felt right on val"
into "state an acceptable error rate alpha; get the threshold whose expected
selective risk is bounded by alpha on exchangeable future rows."

Vendored from a separate, private research project (RAG-Gate —
``calibration/calibrate_crc.py``, the pure-math half only) with the author's
own permission, so this package installs and runs standalone. Unchanged from
the original except for this header — see jevrag/eval/crc.py for how it's
used here. RAG-Gate's data-loading half of calibrate_crc.py (val splits,
cached probe scores, hidden-state extraction) is deliberately NOT vendored:
it is RAG-Gate-specific and JevRAG has its own records format.

Method (as stated in the original):
- Order calibration rows by score descending; empirical selective risk
  R-hat(k) = error rate among top-k.
- Monotonize via running maximum over k (non-decreasing upper envelope of
  the empirical curve). On RAG-Gate's real data the raw selective risk is
  NOT monotone in k, so this step is load-bearing, not cosmetic.
- With B=1 (0-1 error indicator), k-hat = largest k with
  (n/(n+1))*R-tilde(k) + B/(n+1) <= alpha; threshold t-hat = k-th largest
  calibration score; if no k satisfies, abstain-all (t-hat = +inf,
  coverage 0).
- Guarantee: E[risk at t-hat] <= alpha on exchangeable future rows
  (in expectation; no delta — LTT rejected).

This module is pure-Python + numpy, no I/O, so it is unit-tested on
synthetic data.
"""

from __future__ import annotations

import numpy as np

B_BOUND = 1.0  # 0-1 error indicator loss is bounded by 1


def empirical_risk_curve(scores, correct):
    """Selective risk among top-k by score desc, k=1..n. Finite scores only."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    y = np.asarray(correct, dtype=float)[order]
    assert np.isfinite(y).all(), "calibration labels must be finite (drop guarded upstream)"
    errors = np.cumsum(1.0 - y)
    return errors / np.arange(1, len(y) + 1)


def monotonize(risks):
    """Running-maximum upper envelope (non-decreasing). The load-bearing step."""
    return np.maximum.accumulate(np.asarray(risks, dtype=float))


def select_khat(mono_risk, alpha, B=B_BOUND):
    """Largest k with (n/(n+1))*R-tilde(k) + B/(n+1) <= alpha; None -> abstain-all."""
    n = len(mono_risk)
    ok = np.flatnonzero((n / (n + 1)) * mono_risk + B / (n + 1) <= alpha)
    return int(ok.max() + 1) if len(ok) else None


def threshold_for_k(scores, k):
    """k-th largest score (k=1 -> max). None k -> +inf (abstain-all)."""
    if k is None:
        return float("inf")
    return float(np.sort(np.asarray(scores, dtype=float))[::-1][k - 1])


def single_threshold_readout(scores, correct, threshold):
    """Coverage + selective risk at one threshold (repo accounting: NaN labels
    excluded from accuracy, counted in the coverage denominator)."""
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=float)
    answered = np.isfinite(s) & (s >= threshold)  # NaN scores never answered
    n = len(s)
    cov = float(answered.sum() / n)
    valid = answered & np.isfinite(c)
    risk = float(1.0 - c[valid].mean()) if valid.sum() else float("nan")
    return {"coverage": cov, "risk": risk, "n_answered": int(answered.sum())}
