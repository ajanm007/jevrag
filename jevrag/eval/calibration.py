"""jevrag.eval.calibration — is the confidence real?

AURC and the risk-coverage curve are imported unchanged from RAG-Gate's
``selective.py`` (tested, NaN-safe, confidence-source-agnostic).

ECE, MCE, and Brier score are the genuinely new code in this file. They answer the
question the whole project hinges on: when the decision backend says "0.9 confident
this evidence is sufficient", is the downstream answer actually right ~90% of the
time? The vendor's calibration claim is a claim; these numbers measure it on *this*
decision, on *this* dataset.

Conventions:
- ``confidence[i]`` is the backend's probability at the decision point, in [0, 1].
- ``correct[i]`` is the binary correctness of the final answer (EM-derived).
- NaN confidences are excluded from ECE/Brier and counted in the summary, matching
  selective.py's explicit-abstention accounting spirit.
"""

from __future__ import annotations

import numpy as np

from .._rag_gate import selective as _load_selective

_sel = _load_selective()

# Re-exported unchanged — do not reimplement.
risk_coverage_curve = _sel.risk_coverage_curve
aurc = _sel.aurc
coverage_at_accuracy = _sel.coverage_at_accuracy
accuracy_at_coverage = _sel.accuracy_at_coverage
threshold_for_coverage = _sel.threshold_for_coverage
oracle_aurc = _sel.oracle_aurc


def _as_arrays(confidence, correct) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.shape != corr.shape:
        raise ValueError(
            f"confidence and correct must have equal shape, got {conf.shape} vs {corr.shape}"
        )
    if conf.size == 0:
        raise ValueError("empty inputs")
    bad = ~np.isnan(conf)
    if bad.any():
        lo, hi = float(conf[bad].min()), float(conf[bad].max())
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                f"confidence must be a probability in [0, 1]; observed [{lo}, {hi}]. "
                "If this came from a backend, the decision path is emitting something that "
                "isn't a calibrated probability — fix it there, don't clip it here."
            )
    return conf, corr


def reliability_bins(confidence, correct, n_bins: int = 10) -> list[dict]:
    """Equal-width binning over [0, 1] — the standard reliability-diagram table.

    One dict per bin: lo, hi, n, mean_confidence, empirical_accuracy, gap
    (|acc - conf|), and contribution to ECE ((n/N) * gap). Empty bins have n=0 and
    None for the statistics, so callers can render them honestly.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf, corr = _as_arrays(confidence, correct)
    mask = ~np.isnan(conf)
    conf, corr = conf[mask], corr[mask]
    n_total = int(conf.size)

    # Bin index: floor(conf * n_bins), with conf == 1.0 landing in the last bin.
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)

    bins: list[dict] = []
    for b in range(n_bins):
        in_bin = idx == b
        n = int(in_bin.sum())
        row: dict = {"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n,
                     "mean_confidence": None, "empirical_accuracy": None,
                     "gap": None, "contribution": 0.0}
        if n > 0:
            mc = float(conf[in_bin].mean())
            acc = float(corr[in_bin].mean())
            gap = abs(acc - mc)
            row.update(mean_confidence=mc, empirical_accuracy=acc, gap=gap,
                       contribution=(n / n_total) * gap if n_total else 0.0)
        bins.append(row)
    return bins


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """ECE: sum over bins of (|B|/N) * |acc(B) - conf(B)|. Lower is better; 0 is perfect."""
    return float(sum(b["contribution"] for b in reliability_bins(confidence, correct, n_bins)))


def maximum_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """MCE: worst per-bin |acc - conf| over non-empty bins. The catastrophic-failure detector."""
    gaps = [b["gap"] for b in reliability_bins(confidence, correct, n_bins) if b["gap"] is not None]
    return float(max(gaps)) if gaps else float("nan")


def brier_score(confidence, correct) -> float:
    """Brier score for a probabilistic binary prediction: mean((p - c)^2).

    Combines calibration and resolution in one number — a signal can be perfectly
    calibrated and useless (always predicts the base rate); Brier catches that,
    ECE doesn't. Lower is better.
    """
    conf, corr = _as_arrays(confidence, correct)
    mask = ~np.isnan(conf)
    if not mask.any():
        return float("nan")
    return float(np.mean((conf[mask] - corr[mask]) ** 2))


def brier_skill_score(confidence, correct) -> float:
    """Skill vs. the base-rate climatology forecast: 1.0 = perfect, 0.0 = no better
    than always predicting the base rate, negative = worse than the base rate."""
    conf, corr = _as_arrays(confidence, correct)
    mask = ~np.isnan(conf)
    corr_valid = corr[mask]
    base = float(corr_valid.mean())
    ref = float(np.mean((base - corr_valid) ** 2))
    if ref == 0.0:
        return float("nan")  # degenerate labels: no reference forecast exists
    return float(1.0 - brier_score(conf, corr) / ref)


def calibration_summary(confidence, correct, n_bins: int = 10) -> dict:
    """Everything the CLI report needs, in one dict.

    AURC context: aurc is the measured signal; oracle_aurc is the perfect-signal
    lower bound; random_aurc (base risk) is what an uninformative signal scores.
    The gap between aurc and those two bounds is the honest summary of the signal.
    """
    conf, corr = _as_arrays(confidence, correct)
    base_rate = float(np.nanmean(corr))
    return {
        "n": int(conf.size),
        "n_nan_confidence": int(np.isnan(conf).sum()),
        "base_rate_correct": base_rate,
        "ece": expected_calibration_error(conf, corr, n_bins),
        "mce": maximum_calibration_error(conf, corr, n_bins),
        "brier": brier_score(conf, corr),
        "brier_skill": brier_skill_score(conf, corr),
        "aurc": aurc(conf, corr),
        "oracle_aurc": oracle_aurc(corr),
        "random_aurc": 1.0 - base_rate,
        "n_bins": n_bins,
    }
