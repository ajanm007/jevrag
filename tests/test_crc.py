"""Tests for jevrag.eval.crc and the vendored jevrag._vendor.crc core.

The math is vendored unchanged from RAG-Gate's calibrate_crc.py (pure-math
half); the first four tests port that file's own self-test so the vendored
copy is pinned against the checks it shipped with. The rest pin the wrapper's
validation, the abstain-all/answer-all edges, the in-sample bound, and the
expectation guarantee by simulation.

These tests touch no data and need no RAG-Gate checkout — they must pass with
RAG_GATE_PATH pointed at a nonexistent path (the same standalone standard the
selective.py vendoring was held to).
"""

import numpy as np
import pytest

from jevrag._vendor import crc as vendored
from jevrag.eval import crc


# --- Ports of the original self_test() --------------------------------------


def test_monotonize_running_max_envelope():
    raw = np.array([0.0, 0.5, 0.25, 0.25])  # non-monotone by construction
    mono = crc.monotonize(raw)
    assert list(mono) == [0.0, 0.5, 0.5, 0.5]
    assert np.all(np.diff(mono) >= 0)


def test_select_khat_hand_computed():
    # n=4, B=1: k-hat at alpha=0.5 -> (4/5)*R-tilde + 1/5 <= .5
    #           -> R-tilde <= .375 -> k=1 only
    mono = np.array([0.0, 0.5, 0.5, 0.5])
    assert crc.select_khat(mono, 0.5) == 1
    # impossible alpha -> abstain-all (None)
    assert crc.select_khat(np.array([0.9, 0.9]), 0.05) is None


def test_threshold_for_k():
    assert crc.threshold_for_k(np.array([0.2, 0.5, 0.9]), 2) == 0.5
    assert crc.threshold_for_k(np.array([0.2]), None) == float("inf")


def test_readout_nan_accounting():
    r = crc.single_threshold_readout(
        np.array([0.9, 0.1, np.nan]), np.array([1.0, 0.0, 1.0]), 0.5
    )
    assert r["coverage"] == pytest.approx(1 / 3)
    assert r["risk"] == 0.0
    assert r["n_answered"] == 1


# --- Vendored-core properties -------------------------------------------------


def test_empirical_risk_curve_hand_computed():
    # score order 0.9>0.8>0.7>0.6 over labels 1,1,0,1 -> errors 0,0,1,1
    risks = crc.empirical_risk_curve(
        np.array([0.6, 0.9, 0.7, 0.8]), np.array([1, 1, 0, 1])
    )
    assert risks == pytest.approx([0.0, 0.0, 1 / 3, 0.25])


def test_crc_reexports_are_vendored():
    # These must be the vendored CRC functions, not local reimplementations.
    for name in (
        "empirical_risk_curve",
        "monotonize",
        "select_khat",
        "threshold_for_k",
        "single_threshold_readout",
    ):
        assert getattr(crc, name).__module__ == "jevrag._vendor.crc"
        assert getattr(crc, name) is getattr(vendored, name)


# --- Wrapper validation -------------------------------------------------------


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape"):
        crc.crc_threshold(np.array([0.5, 0.5]), np.array([1]), 0.2)


def test_empty_inputs_rejected():
    with pytest.raises(ValueError, match="empty"):
        crc.crc_threshold(np.array([]), np.array([]), 0.2)


def test_alpha_out_of_range_rejected():
    for bad in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            crc.crc_threshold(np.array([0.5]), np.array([1]), bad)


def test_non_finite_labels_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        crc.crc_threshold(np.array([0.9, 0.1]), np.array([1.0, np.nan]), 0.2)


def test_out_of_range_labels_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        crc.crc_threshold(np.array([0.9, 0.1]), np.array([1.0, 2.0]), 0.2)


def test_all_nan_scores_rejected():
    with pytest.raises(ValueError, match="NaN"):
        crc.crc_threshold(np.array([np.nan, np.nan]), np.array([1, 0]), 0.2)



def test_nan_scores_excluded_and_counted():
    s = np.array([0.9, np.nan, 0.1])
    c = np.array([1.0, 1.0, 0.0])
    out = crc.crc_calibrate(s, c, 0.5)
    assert out["n_cal"] == 2
    assert out["n_nan_scores"] == 1


# --- Behavior at the edges ----------------------------------------------------


def test_abstain_all_when_signal_cannot_meet_budget():
    # All wrong: R-tilde is flat 1.0, no alpha < 1 can be satisfied.
    s = np.array([0.9, 0.8, 0.1])
    c = np.array([0.0, 0.0, 0.0])
    out = crc.crc_calibrate(s, c, 0.2)
    assert out["abstain_all"] is True
    assert out["k_hat"] is None
    assert out["threshold"] == float("inf")
    ro = crc.single_threshold_readout(s, c, out["threshold"])
    assert ro["coverage"] == 0.0 and np.isnan(ro["risk"])


def test_answer_all_when_perfect_and_budget_loose():
    s = np.array([0.9, 0.5, 0.1])
    c = np.array([1.0, 1.0, 1.0])
    out = crc.crc_calibrate(s, c, 0.9)
    assert out["k_hat"] == 3
    ro = crc.single_threshold_readout(s, c, out["threshold"])
    assert ro["coverage"] == 1.0 and ro["risk"] == 0.0


# --- Tie handling (packet CLINE_23) ------------------------------------------


def _tied_batch(n_high=7, n_mid=23, qid_prefix="q"):
    """A batch with ties at the threshold, laid out like the real records.

    7 rows at 0.98 (the LAST one wrong), 23 at 0.97 (last two wrong), then three
    lower scores. Deterministic on purpose: the tests are about tie handling,
    not about a lucky draw — the packet-22 real data had exactly this shape
    (7 rows at 0.98, one of them an error).
    """
    scores = np.array([0.98] * n_high + [0.97] * n_mid + [0.60, 0.55, 0.50])
    correct = np.ones(scores.size)
    correct[n_high - 1] = 0.0
    correct[n_high + n_mid - 1] = 0.0
    correct[n_high + n_mid - 2] = 0.0
    keys = [f"{qid_prefix}{i}" for i in range(scores.size)]
    return scores, correct, keys


def test_tie_block_overshoots_khat_without_tie_break():
    # The packet-22 bug, pinned: k-hat says 6 rows, the value rule answers all 7
    # rows tied at 0.98 (one of them wrong) -> risk 1/7 = 0.1429 > alpha 0.10.
    s, c, _ = _tied_batch()
    raw = crc.crc_calibrate(s, c, 0.10, tie_break=False)
    assert raw["abstain_all"] is False
    assert raw["k_hat"] == 6
    assert raw["n_answered_cal"] == 7
    assert raw["n_answered_cal"] > raw["k_hat"]
    assert raw["tie_inflated"] is True
    assert raw["tie_break"] == "none"
    ro = crc.single_threshold_readout(s, c, raw["threshold"])
    assert ro["risk"] > 0.10  # the bound as applied, broken


def test_tie_break_answers_exactly_khat():
    s, c, keys = _tied_batch()
    fixed = crc.crc_calibrate(s, c, 0.10, keys=keys)
    assert fixed["tie_break"] == "sha256(key)"
    assert fixed["tie_inflated"] is False
    ro = crc.crc_readout(s, c, keys, fixed["threshold"])
    if fixed["abstain_all"]:
        # The hash dropped the erroneous row to rank 1: k-hat = None, no answer.
        assert fixed["k_hat"] is None
        assert fixed["n_answered_cal"] == 0 and ro["n_answered"] == 0
    else:
        # Answered set is exactly the top-k-hat, so the in-sample bound holds by
        # construction instead of being broken by a tie block.
        assert fixed["n_answered_cal"] == fixed["k_hat"]
        assert ro["n_answered"] == fixed["k_hat"]
        assert ro["risk"] <= 0.10



def test_ties_without_keys_are_refused():
    s, c, _ = _tied_batch()
    with pytest.raises(ValueError, match="tied at the CRC threshold"):
        crc.crc_calibrate(s, c, 0.10)


def test_tie_break_is_order_invariant():
    # A rank/top-k-of-the-batch rule would depend on row order; the hash rule is
    # a per-row function, so shuffling the batch cannot change any decision.
    s, c, keys = _tied_batch()
    fixed = crc.crc_calibrate(s, c, 0.10, keys=keys)
    base = crc.crc_answered(s, keys, fixed["threshold"])
    rng = np.random.default_rng(5)
    for _ in range(5):
        perm = rng.permutation(s.size)
        s2, c2, k2 = s[perm], c[perm], [keys[i] for i in perm]
        other = crc.crc_calibrate(s2, c2, 0.10, keys=k2)
        assert other["k_hat"] == fixed["k_hat"]
        assert other["threshold"] == pytest.approx(fixed["threshold"])
        answered = dict(zip(k2, crc.crc_answered(s2, k2, other["threshold"])))
        for key, flag in zip(keys, base):
            assert answered[key] == flag


def test_fresh_row_rule_is_per_row():
    # Inference-time generalization: deciding one unseen row alone gives the
    # same answer as deciding it inside the batch. This is what rules out
    # "answer the top-k of the batch" as an implementation.
    s, c, keys = _tied_batch()
    fixed = crc.crc_calibrate(s, c, 0.10, keys=keys)
    t = fixed["threshold"]
    batch = crc.crc_answered(s, keys, t)
    alone = np.array(
        [bool(crc.crc_answered([s[i]], [keys[i]], t)[0]) for i in range(s.size)]
    )
    assert np.array_equal(batch, alone)
    # A genuinely new row (unseen key, tied score) is decided by the same rule.
    assert crc.crc_answered([0.98], ["brand-new-question"], t).shape == (1,)


def test_duplicate_keys_rejected():
    s, c, keys = _tied_batch()
    keys = list(keys)
    keys[3] = keys[2]
    with pytest.raises(ValueError, match="unique"):
        crc.crc_calibrate(s, c, 0.10, keys=keys)


def test_key_length_mismatch_rejected():
    with pytest.raises(ValueError, match="equal length"):
        crc.tie_break_scores([0.9, 0.8], ["only-one-key"])


def test_jitter_refuses_scores_finer_than_eps():
    # Sub-epsilon score gaps must be refused, not silently reordered.
    with pytest.raises(ValueError, match="TIE_BREAK_EPS"):
        crc.tie_break_scores([0.5, 0.5 + 1e-9], ["a", "b"])


def test_offsets_deterministic_in_range_and_salt_sensitive():
    keys = ["a", "b", "c"]
    off = crc.crc_offsets(keys)
    assert np.array_equal(off, crc.crc_offsets(keys))
    assert np.all((off >= 0.0) & (off < 1.0))
    assert not np.array_equal(off, crc.crc_offsets(keys, salt="s1"))
    assert not np.array_equal(crc.crc_offsets(["a"]), crc.crc_offsets(["b"]))


def test_no_key_path_unchanged_for_distinct_scores():
    # With genuinely distinct scores the fixed path is a no-op: identical
    # k-hat/threshold/risk to the vendored chain computed by hand.
    rng = np.random.default_rng(13)
    s = rng.uniform(0.05, 0.95, 60)
    c = rng.binomial(1, 0.65, 60).astype(float)
    out = crc.crc_calibrate(s, c, 0.25)
    mono = crc.monotonize(crc.empirical_risk_curve(s, c))
    k = crc.select_khat(mono, 0.25)
    assert out["k_hat"] == k
    assert out["threshold"] == crc.threshold_for_k(s, k)
    assert out["n_answered_cal"] == k
    assert out["tie_inflated"] is False


def test_marginal_risk_is_the_bounded_quantity():
    # marginal = P(answered and wrong) = coverage * selective risk. On fresh
    # data the conditional rate can exceed alpha while this one holds.
    s, c, keys = _tied_batch()
    fixed = crc.crc_calibrate(s, c, 0.10, keys=keys)
    ro = crc.crc_readout(s, c, keys, fixed["threshold"])
    assert ro["marginal_risk"] == pytest.approx(ro["coverage"] * ro["risk"])
    # A holdout draw whose conditional rate exceeds alpha, and whose marginal
    # rate does not — the same shape the real records showed at alpha=0.10.
    s_ho, c_ho, keys_ho = _tied_batch(qid_prefix="h")
    ro_ho = crc.crc_readout(s_ho, c_ho, keys_ho, fixed["threshold"])
    if np.isfinite(ro_ho["risk"]):
        assert ro_ho["marginal_risk"] <= ro_ho["risk"] + 1e-12


def test_guarantee_with_heavy_ties_across_salts():
    # The claim the fix is for: with heavy ties (0.05-quantized scores) and a
    # hash tie-break, E[risk] over tie-break realizations stays under alpha.
    # One salt is one realization; the guarantee is the expectation.
    rng = np.random.default_rng(17)
    alpha = 0.2
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    s_cal = rng.choice(grid, 150)
    c_cal = rng.binomial(1, s_cal).astype(float)
    s_fut = rng.choice(grid, 300)
    c_fut = rng.binomial(1, s_fut).astype(float)
    keys_cal = [f"c{i}" for i in range(150)]
    keys_fut = [f"f{i}" for i in range(300)]
    risks, holds = [], 0
    for t_i in range(200):
        salt = f"s{t_i}"
        calib = crc.crc_calibrate(s_cal, c_cal, alpha, keys=keys_cal, salt=salt)
        if calib["abstain_all"]:
            continue
        ro = crc.crc_readout(s_fut, c_fut, keys_fut, calib["threshold"], salt=salt)
        if np.isfinite(ro["risk"]):
            risks.append(ro["risk"])
            holds += ro["risk"] <= alpha
    assert len(risks) > 150
    assert float(np.mean(risks)) <= alpha + 0.015
    # Not every single realization can hold on a quantized scale — that is the
    # difference between the guarantee (expectation) and a per-draw claim.
    assert holds / len(risks) > 0.6



def test_crc_threshold_matches_crc_calibrate():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 50)
    c = rng.binomial(1, 0.6, 50).astype(float)
    assert crc.crc_threshold(s, c, 0.3) == crc.crc_calibrate(s, c, 0.3)["threshold"]


def test_tighter_alpha_never_gives_looser_threshold():
    # As alpha grows the threshold is non-increasing (more budget, more coverage).
    rng = np.random.default_rng(1)
    c = rng.binomial(1, 0.6, 80).astype(float)
    s = np.clip(c * 0.5 + 0.25 + rng.normal(0, 0.15, 80), 0.01, 0.99)
    ts = [crc.crc_threshold(s, c, a) for a in (0.05, 0.1, 0.2, 0.3, 0.5)]
    assert all(t1 >= t2 for t1, t2 in zip(ts, ts[1:]))


# --- The guarantee ------------------------------------------------------------


def test_in_sample_bound_holds_by_construction():
    # On the calibration rows themselves, risk at the CRC threshold is below
    # alpha: R-hat(k-hat) <= R-tilde(k-hat) <= ((n+1)/n)*alpha - B/n < alpha.
    rng = np.random.default_rng(2)
    for alpha in (0.1, 0.2, 0.3):
        c = rng.binomial(1, 0.6, 100).astype(float)
        s = np.clip(c * 0.5 + 0.25 + rng.normal(0, 0.2, 100), 0.01, 0.99)
        out = crc.crc_calibrate(s, c, alpha)
        if out["abstain_all"]:
            continue
        ro = crc.single_threshold_readout(s, c, out["threshold"])
        assert ro["risk"] <= alpha


def test_expectation_guarantee_by_simulation():
    # E[risk at t-hat] <= alpha on exchangeable future rows. Informative
    # signal (p(correct) = score), n_cal=100, n_future=200, 500 trials.
    rng = np.random.default_rng(3)
    alpha = 0.2
    risks = []
    for _ in range(500):
        s_cal = rng.uniform(0.05, 0.95, 100)
        c_cal = rng.binomial(1, s_cal).astype(float)
        t = crc.crc_threshold(s_cal, c_cal, alpha)
        if not np.isfinite(t):
            continue  # abstain-all trials contribute risk 0 to the expectation
        s_fut = rng.uniform(0.05, 0.95, 200)
        c_fut = rng.binomial(1, s_fut).astype(float)
        ro = crc.single_threshold_readout(s_fut, c_fut, t)
        if np.isfinite(ro["risk"]):
            risks.append(ro["risk"])
    assert len(risks) > 400  # the signal should rarely have to abstain
    assert float(np.mean(risks)) <= alpha + 0.01
