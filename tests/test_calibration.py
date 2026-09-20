"""Tests for jevrag.eval.calibration — ECE/MCE/Brier are the genuinely new code.

AURC/risk-coverage come from RAG-Gate's tested selective.py; these tests pin the
new math against hand-computed cases and known properties.
"""

import numpy as np
import pytest

from jevrag.eval import calibration as cal


def test_ece_zero_when_perfectly_calibrated():
    # 100 points at conf 0.7 with exactly 70 correct, twice (two bins' worth
    # collapsed into one value) -> bin accuracy equals confidence exactly.
    conf = np.array([0.7] * 100)
    corr = np.array([1] * 70 + [0] * 30)
    assert cal.expected_calibration_error(conf, corr) == pytest.approx(0.0)
    assert cal.maximum_calibration_error(conf, corr) == pytest.approx(0.0)


def test_ece_hand_computed_two_bins():
    # Bin [0.0, 0.5): conf 0.4, accuracy 0/2 = 0.0  -> gap 0.4, weight 2/4
    # Bin [0.5, 1.0]: conf 0.8, accuracy 2/2 = 1.0  -> gap 0.2, weight 2/4
    conf = np.array([0.4, 0.4, 0.8, 0.8])
    corr = np.array([0, 0, 1, 1])
    ece = cal.expected_calibration_error(conf, corr, n_bins=2)
    assert ece == pytest.approx(0.5 * 0.4 + 0.5 * 0.2)
    assert cal.maximum_calibration_error(conf, corr, n_bins=2) == pytest.approx(0.4)


def test_confidence_one_lands_in_last_bin():
    conf = np.array([1.0, 1.0])
    corr = np.array([1, 1])
    bins = cal.reliability_bins(conf, corr, n_bins=10)
    assert bins[-1]["n"] == 2
    assert sum(b["n"] for b in bins) == 2
    assert cal.expected_calibration_error(conf, corr) == pytest.approx(0.0)


def test_empty_bins_reported_honestly():
    conf = np.array([0.15, 0.85])
    corr = np.array([0, 1])
    bins = cal.reliability_bins(conf, corr, n_bins=10)
    assert bins[0]["n"] == 0 and bins[0]["gap"] is None
    assert bins[5]["n"] == 0
    assert bins[1]["n"] == 1 and bins[8]["n"] == 1


def test_brier_hand_computed():
    conf = np.array([0.9, 0.1])
    corr = np.array([1, 0])
    # (0.9-1)^2 = 0.01, (0.1-0)^2 = 0.01 -> mean 0.01
    assert cal.brier_score(conf, corr) == pytest.approx(0.01)


def test_brier_skill_perfect_and_base_rate():
    conf = np.array([1.0, 0.0, 1.0, 0.0])
    corr = np.array([1, 0, 1, 0])
    assert cal.brier_skill_score(conf, corr) == pytest.approx(1.0)
    # Always predicting the base rate -> skill exactly 0.
    base = np.array([0.5, 0.5, 0.5, 0.5])
    assert cal.brier_skill_score(base, corr) == pytest.approx(0.0)


def test_nan_confidence_excluded_not_silently_zeroed():
    conf = np.array([0.9, np.nan, 0.1])
    corr = np.array([1, 1, 0])
    summary = cal.calibration_summary(conf, corr)
    assert summary["n_nan_confidence"] == 1
    assert summary["brier"] == pytest.approx((0.01 + 0.01) / 2)


def test_out_of_range_confidence_rejected():
    with pytest.raises(ValueError, match="probability"):
        cal.expected_calibration_error(np.array([1.5]), np.array([1]))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape"):
        cal.brier_score(np.array([0.5, 0.5]), np.array([1]))


def test_summary_aurc_bounds_order_on_good_signal():
    rng = np.random.default_rng(0)
    corr = rng.binomial(1, 0.6, 500).astype(float)
    # Informative signal: high conf on correct, low on wrong, plus noise.
    conf = np.clip(corr * 0.5 + 0.25 + rng.normal(0, 0.1, 500), 0.01, 0.99)
    s = cal.calibration_summary(conf, corr)
    assert s["oracle_aurc"] <= s["aurc"] <= s["random_aurc"]
    assert 0.0 <= s["ece"] <= 1.0


def test_selective_reexports_are_rag_gates():
    # These must be the rag-gate functions, not reimplementations (brief §5).
    assert cal.aurc.__module__ == "rag_gate_selective"
    assert cal.risk_coverage_curve.__module__ == "rag_gate_selective"
    curve = cal.risk_coverage_curve(np.array([0.9, 0.1]), np.array([1, 0]))
    assert set(curve) >= {"coverage", "selective_risk", "threshold"}
