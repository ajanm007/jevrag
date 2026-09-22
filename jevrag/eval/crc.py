"""jevrag.eval.crc — the threshold comes from an error budget, not a guess.

CRC (conformal risk control) threshold selection, imported unchanged from the
pure-math core of RAG-Gate's ``calibrate_crc.py`` (vendored at
``jevrag/_vendor/crc.py``), the same way ``calibration.py`` wraps
``selective.py``.

``calibration.py`` answers "is the confidence real?". This module answers the
follow-up: "given that I tolerate an error rate of ``alpha`` on answered
questions, where does the threshold go?"

Ties, and why this module jitters
---------------------------------
``select_khat`` selects by RANK (the top-k of the calibration order). A plain
``score >= threshold`` read-out answers by VALUE. On continuous scores the two
coincide — the case RAG-Gate's version was written for. JevRAG's confidences
are quantized (2 decimals: 55 distinct values across 180 real sufficiency rows,
7 of them at 0.98), so the value rule answers every tied row and overshoots
k-hat. That overshoot is what broke the bound on real records (packet
CLINE_22), and it is fixed here.

The fix is a randomized tie-break made reproducible by hashing a stable
per-row key::

    scored(x) = score(x) + TIE_BREAK_EPS * offset(key(x))

with ``offset`` in [0, 1) from sha256. Applied to the calibration rows as well,
this makes the order total, so "top-k-hat" and "score >= threshold" mean the
same thing again — and the decision stays a function of one row, which is what
a gate actually needs: a rule phrased as "the top-k of the batch" has no
meaning for a batch of one.

Consequences, stated plainly:

- The value the rule compares against is **not** the raw confidence. It is
  ``score + TIE_BREAK_EPS * offset(key)``. :func:`crc_calibrate` returns both
  ``threshold`` (decision space — what the rule uses) and ``threshold_score``
  (the raw confidence of the k-hat-th row, for display and for humans).
- A tie at the boundary is resolved by the row's key hash: deterministic per
  row, arbitrary with respect to the row's content. That is the standard
  randomized tie-break in conformal prediction, and it carries the assumption
  behind it — the hash behaves like an independent draw, not something
  correlated with score or correctness. :func:`crc_offsets` exposes the
  offsets so that assumption can be checked on real records.
- Jitter changes k-hat selection too, not just the read-out (it permutes the
  order inside tie blocks, and the risk curve is computed on that order). That
  is intended: the previous selection was an artifact of stable-sort tie
  order, not a property of the method.
- Callers with genuinely continuous scores can pass ``tie_break=False`` and get
  the raw behavior. With ties present that is the known-broken path: it is
  reported honestly (``tie_inflated=True``) rather than failing silently, and
  without ``keys`` at all it is refused outright.

Conventions match ``calibration.py`` and ``selective.py``:
- ``scores[i]`` is the gate signal (JevRAG: the decision-point confidence);
  higher = more confident = answered first. NaN scores are excluded from
  calibration and counted, matching the explicit-abstention accounting.
- ``correct[i]`` is the binary (0/1, EM-derived) correctness label; the B=1
  loss bound is only valid for losses in [0, 1], so labels are range-checked.
- ``alpha`` is the caller-chosen accepted error rate on answered questions.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .._rag_gate import crc as _load_crc

_crc = _load_crc()

# Re-exported unchanged — do not reimplement.
empirical_risk_curve = _crc.empirical_risk_curve
monotonize = _crc.monotonize
select_khat = _crc.select_khat
threshold_for_k = _crc.threshold_for_k
single_threshold_readout = _crc.single_threshold_readout

B_BOUND = _crc.B_BOUND

#: Tie-break offset scale. Far below JevRAG's confidence precision (2 decimals,
#: quantum 1e-2), so it can order tied rows without ever crossing a real score
#: gap. :func:`tie_break_scores` proves that per batch and refuses otherwise.
TIE_BREAK_EPS = 1e-6

#: Offsets are 32 bits of sha256 lopped off, divided by 2^32 -> [0, 1) with
#: resolution 2.3e-10. That resolution matters: times TIE_BREAK_EPS it is
#: 2.3e-16, at or above one float ULP across [0, 1), so distinct offsets give
#: distinct jittered scores instead of collapsing into the low bits.
_TWO_POW_32 = float(1 << 32)

def crc_offsets(keys, salt: str = "") -> np.ndarray:
    """Deterministic per-row tie-break offsets in [0, 1), from sha256(key).

    The offset depends only on the row's own stable key (plus an optional
    ``salt``, which exists so a caller can study the tie-break's realization
    noise; the deployed rule uses the default empty salt). It never depends on
    the row's score, its position in the batch, or whether its answer was
    correct — that independence is what makes the tie-break a valid
    randomization rather than a second guess.
    """
    keys = list(keys)
    out = np.empty(len(keys), dtype=float)
    for i, key in enumerate(keys):
        digest = hashlib.sha256(f"{salt}\x1f{key}".encode("utf-8")).digest()
        out[i] = int.from_bytes(digest[:4], "big") / _TWO_POW_32
    return out


def _check_keys_unique(keys) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for key in keys:
        s = str(key)
        if s in seen:
            dupes.append(s)
        seen.add(s)
    if dupes:
        raise ValueError(
            f"tie-break keys must be unique per row; {len(dupes)} duplicate(s), "
            f"first {dupes[0]!r}. Rows sharing a key share an offset and stay "
            "tied; pass a unique per-row key (e.g. question_id, or "
            "'<question_id>:<passage_index>' where one question repeats)."
        )


def tie_break_scores(scores, keys, salt: str = "") -> np.ndarray:
    """``scores + TIE_BREAK_EPS * offset(key)`` — the decision-space score.

    Exactness guards (raise rather than mis-order silently):
    - keys unique per row, else the tie survives;
    - no two distinct input scores closer than 2 * TIE_BREAK_EPS, else jitter
      could swap two genuinely different scores (a hidden behavior change);
    - offsets distinct, and the jittered order equals the lexicographic order
      (score desc, offset desc — a bigger offset answers sooner) — belt and
      braces over the two checks above.
    """
    s = np.asarray(scores, dtype=float)
    keys = list(keys)
    if len(keys) != s.size:
        raise ValueError(
            f"keys and scores must have equal length, got {len(keys)} vs {s.size}"
        )
    _check_keys_unique(keys)
    off = crc_offsets(keys, salt)
    jittered = s + TIE_BREAK_EPS * off
    finite = np.isfinite(s)
    j_f, s_f, o_f = jittered[finite], s[finite], off[finite]
    uniq = np.unique(s_f)
    if uniq.size > 1 and float(np.min(np.diff(uniq))) <= 2 * TIE_BREAK_EPS:
        raise ValueError(
            "two scores are closer than 2 * TIE_BREAK_EPS "
            f"({2 * TIE_BREAK_EPS:g}), so jittering could reorder genuinely "
            "different scores; re-quantize the scores or raise TIE_BREAK_EPS "
            "above the score quantum"
        )
    if len(np.unique(j_f)) != j_f.size:
        raise ValueError(
            "tie-break offsets collided (duplicate jittered scores); pass a "
            "different salt or a different key set"
        )
    if not np.array_equal(np.argsort(-j_f, kind="stable"), np.lexsort((-o_f, -s_f))):
        raise ValueError(
            "jittered order does not match the lexicographic (score desc, "
            "offset asc) order — refusing to decide on an order the guards "
            "cannot account for"
        )
    return jittered


def crc_answered(scores, keys, threshold, salt: str = "") -> np.ndarray:
    """The decision rule, per row: ``score + eps*offset(key) >= threshold``.

    Works for one fresh row (a batch of one) exactly as it does for a whole
    calibration set: the rule never looks at other rows, which is what makes it
    usable at inference time. NaN scores are never answered.
    """
    jittered = tie_break_scores(scores, keys, salt)
    return np.isfinite(np.asarray(scores, dtype=float)) & (jittered >= threshold)


def crc_readout(scores, correct, keys, threshold, salt: str = "") -> dict:
    """Coverage + risk at the tie-broken threshold, with both risk readings.

    Same accounting as the vendored ``single_threshold_readout`` (NaN scores are
    never answered but stay in the coverage denominator; NaN labels lower
    coverage without entering the selective risk), plus the quantity the CRC
    guarantee is actually about:

    - ``selective_risk`` = P(wrong | answered) — the usual report number, and the
      one the bound is proved against *on the calibration rows*;
    - ``marginal_risk`` = P(answered and wrong) = coverage * selective_risk —
      what CRC bounds for a fresh row: E[L(threshold)] <= alpha with
      L = 1{answered} * 1{wrong} (Angelopoulos et al. 2022).

    On fresh data the conditional rate can exceed alpha while the bound still
    holds, because it is a rate over a *different* answered set than the top-k
    the calibration bound was computed on. Read the two together or the check
    will look like a violation when it is a category error.
    """
    jittered = tie_break_scores(scores, keys, salt)
    ro = single_threshold_readout(jittered, correct, threshold)
    ro["marginal_risk"] = (
        ro["coverage"] * ro["risk"] if np.isfinite(ro["risk"]) else float("nan")
    )
    return ro

def _as_arrays(scores, correct, alpha) -> tuple[np.ndarray, np.ndarray, float]:
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=float)
    if s.shape != c.shape:
        raise ValueError(
            f"scores and correct must have equal shape, got {s.shape} vs {c.shape}"
        )
    if s.size == 0:
        raise ValueError("empty inputs")
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(
            f"alpha must be in (0, 1), got {a}; alpha <= 0 is unreachable "
            "(B/(n+1) > 0 always) and alpha >= 1 is answer-all trivially"
        )
    keep = ~np.isnan(s)
    if not keep.any():
        raise ValueError("all scores are NaN — nothing to calibrate on")
    kept_labels = c[keep]
    bad = ~np.isfinite(kept_labels)
    if bad.any():
        raise ValueError(
            f"correct contains {int(bad.sum())} non-finite labels at scored rows; "
            "drop or impute them upstream, don't let them through silently"
        )
    lo, hi = float(kept_labels.min()), float(kept_labels.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"correct must be a loss in [0, 1] (0/1 EM labels here); observed "
            f"[{lo}, {hi}]. With B=1 hardwired, a wider label range silently "
            "breaks the bound — fix the labels, don't clip them."
        )
    return s, c, a


def crc_calibrate(
    scores,
    correct,
    alpha: float,
    keys=None,
    salt: str = "",
    tie_break: bool = True,
) -> dict:
    """Calibrate a CRC threshold at error budget ``alpha``.

    ``keys`` (a unique, stable id per row — ``question_id`` in JevRAG's
    records) turns on the deterministic tie-break, and is the supported path:
    calibration and read-out then run in decision space, the answered set on the
    calibration rows is exactly the top-k-hat, and the bound holds as designed.
    Any batch size works, including one: the rule is per row.

    Without ``keys`` the raw scores are used. That is exact for continuous
    scores; if ties inflate the answered set beyond k-hat, the call is refused
    (the guarantee would not hold for what it returned) with a message naming
    the fix. ``tie_break=False`` opts out of both and instead reports
    ``tie_inflated=True`` — the honest form of the packet-22 behavior, kept for
    callers who want to reproduce it deliberately.

    Returns a dict with the threshold in decision space plus what a report
    needs: ``threshold_score`` (raw confidence of the k-hat-th row), ``k_hat``,
    ``n_answered_cal``, ``tie_inflated``, ``tie_break``, ``salt``, the
    finite-sample correction, calibration-side risks and NaN accounting.
    """
    s, c, a = _as_arrays(scores, correct, alpha)
    keep = ~np.isnan(s)
    keys_seq = None if keys is None else list(keys)
    if keys_seq is not None and len(keys_seq) != s.size:
        raise ValueError(
            f"keys and scores must have equal length, got {len(keys_seq)} vs {s.size}"
        )
    s_k, c_k = s[keep], c[keep]
    n = int(s_k.size)

    if tie_break and keys_seq is not None:
        keys_kept = [k for k, kp in zip(keys_seq, keep) if kp]
        s_dec = tie_break_scores(s_k, keys_kept, salt)
        tie_break_tag = "sha256(key)"
    else:
        s_dec = s_k
        tie_break_tag = "none"

    raw = empirical_risk_curve(s_dec, c_k)
    mono = monotonize(raw)
    k_hat = select_khat(mono, a, B=B_BOUND)
    threshold = float(threshold_for_k(s_dec, k_hat))
    if k_hat is None:
        n_answered_cal = 0
        threshold_score = float("inf")
    else:
        n_answered_cal = int((s_dec >= threshold).sum())
        order = np.argsort(-s_dec, kind="stable")
        threshold_score = float(s_k[order[k_hat - 1]])
    tie_inflated = bool(k_hat is not None and n_answered_cal > k_hat)

    if tie_inflated and tie_break and keys_seq is None:
        raise ValueError(
            f"scores are tied at the CRC threshold: k-hat={k_hat} but "
            f"{n_answered_cal} calibration rows satisfy score >= threshold, so "
            "the answered set is larger than the one the bound was computed "
            "for. Pass keys=<unique id per row> for the deterministic "
            "tie-break, or tie_break=False to accept the unbounded read-out."
        )

    return {
        "alpha": a,
        "n_cal": n,
        "n_nan_scores": int((~keep).sum()),
        "k_hat": k_hat,
        "threshold": threshold,
        "threshold_score": threshold_score,
        "n_answered_cal": n_answered_cal,
        "tie_inflated": tie_inflated,
        "tie_break": tie_break_tag,
        "salt": salt,
        "b_correction": float(B_BOUND / (n + 1)),
        "abstain_all": k_hat is None,
        "cal_risk_at_k": float(raw[k_hat - 1]) if k_hat is not None else None,
        "cal_mono_risk_at_k": float(mono[k_hat - 1]) if k_hat is not None else None,
    }


def crc_threshold(
    scores, correct, alpha: float, keys=None, salt: str = "", tie_break: bool = True
) -> float:
    """The CRC-calibrated threshold in **decision space**.

    Answer iff ``score + TIE_BREAK_EPS * offset(key) >= threshold`` — use
    :func:`crc_answered` / :func:`crc_readout`, not a bare ``score >=`` against
    this number, or tied rows will be under-answered (silently conservative).
    ``threshold_score`` in :func:`crc_calibrate`'s dict is the raw-confidence
    value for display. Returns +inf (abstain-all, coverage 0) when no k satisfies
    the bound at this alpha — the honest answer when the signal cannot meet the
    error budget, matching the original's behavior.
    """
    return crc_calibrate(scores, correct, alpha, keys=keys, salt=salt,
                         tie_break=tie_break)["threshold"]


