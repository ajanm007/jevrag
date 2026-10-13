"""Shift-test proof tests (packet V03-A10). Offline only, no live calls.

Synthetic result files are written directly in the runner's row format into
tmp_path (no repo fixtures, no /mnt or D:\\ literals). Every asserted number
is a literal computed by hand in the test — never via the script's own
helpers (no crc_readout / split_metrics / calibrate_rule imports here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.shift_test import main  # noqa: E402


def _row(qid, arm, split, em, aconf, sconf=None, dataset="tiny"):
    sconf = aconf if sconf is None else sconf
    r = {"question_id": qid, "question": f"q-{qid}?", "gold": "g",
         "outcome": "answer", "answer": "p", "prediction": "p",
         "source": "pipeline", "arm": arm, "split": split,
         "dataset": dataset, "hops": None, "em": em, "f1": 1.0 if em else 0.0,
         "jev_calls": 3, "jev_input_tokens": 100, "selection_calls": 2,
         "selection": {"rounds": [{"jev_calls": 2, "jev_input_tokens": 100,
                                   "new_candidates": ["x"]}], "calls": 2},
         "wall_ms": 1.0}
    if arm == "fixed":
        r["sufficiency"] = None
        r["abstain"] = None
    else:
        r["sufficiency"] = {"record": {"confidence": sconf, "rounds_used": 1}}
        r["abstain"] = {"record": {"confidence": aconf}}
    return r


def _write(tmp_path, name, dataset, rows, alpha=0.3, thresholds=None):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__header__": {"dataset": dataset,
                                           "alpha": alpha}}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if thresholds is not None:
        with open(str(p)[:-len(".jsonl")] + ".trace.json", "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"abst_thresholds": thresholds}))
    return str(p)


def _shift_lines(out, joint):
    return [l for l in out.splitlines() if f"shift:{joint}" in l]


def test_shift_values_hand_computed(tmp_path, capsys):
    # Source calibrate, all correct: abstain confs 0.9/0.8/0.7, suff confs
    # 0.6/0.55/0.5, alpha 0.3. By hand (zero empirical risk at every k):
    # k_hat = 3 = n, abstain threshold ~= 0.7, suff threshold ~= 0.5
    # (tie-break eps 1e-6 cannot cross the >=0.05 gaps).
    src = [_row(f"c{i}", "gated", "calibrate", True, a, s)
           for i, (a, s) in enumerate([(0.9, 0.6), (0.8, 0.55),
                                       (0.7, 0.5)])]
    src.append(_row("k0", "gated_crc", "report", True, 0.9, 0.6,
                    dataset="src"))
    sp = _write(tmp_path, "src.jsonl", "src", src)
    # Target report: abstain confs 0.95(T) 0.6(F) 0.1(F): answered at ~=0.7
    # is only 0.95 -> coverage 1/3, selective 0/1 = 0.0, marginal 0.0.
    # Suff confs 0.45(T) 0.65(F) 0.2(F): answered at ~=0.5 is only 0.65
    # (wrong) -> coverage 1/3, selective 1.0, marginal 1/3.
    tgt = [_row("r0", "gated", "report", True, 0.95, 0.45, dataset="tgt"),
           _row("r1", "gated", "report", False, 0.6, 0.65, dataset="tgt"),
           _row("r2", "gated", "report", False, 0.1, 0.2, dataset="tgt")]
    # In-file CRC context: confs 0.9(T) 0.7(F) at recorded 0.5 -> answered
    # both -> coverage 1.0, selective 1/2, marginal 1/2.
    tgt += [_row("k0", "gated_crc", "report", True, 0.9, dataset="tgt"),
            _row("k1", "gated_crc", "report", False, 0.7, dataset="tgt")]
    tp = _write(tmp_path, "tgt.jsonl", "tgt", tgt,
                thresholds={"gated_crc": 0.5})
    assert main(["--source", sp, "--targets", tp]) == 0
    out = capsys.readouterr().out
    assert "Single-source shift across closed-pool multi-hop datasets" in out
    (aline,) = _shift_lines(out, "answer_abstain")
    assert "| tgt | 3 |" in aline
    assert "| 0.333 | 0.000 | 0.000 |" in aline
    (sline,) = _shift_lines(out, "sufficiency")
    assert "| tgt | 3 |" in sline
    assert "| 0.333 | 1.000 | 0.333 |" in sline
    ctx = [l for l in out.splitlines() if "gated_crc-infile" in l]
    assert len(ctx) == 1 and "| 1.000 | 0.500 | 0.500 |" in ctx[0]


def test_source_abstain_all_covers_nothing(tmp_path, capsys):
    # Source calibrate, all wrong: confs 0.9/0.8, alpha 0.3. By hand:
    # top-1 risk 1.0 -> (2/3)*1 + 1/3 = 1.0 > 0.3 at every k -> k_hat None
    # -> abstain-all (threshold +inf): coverage 0, risk n/a (never 0.0).
    src = [_row("c0", "gated", "calibrate", False, 0.9),
           _row("c1", "gated", "calibrate", False, 0.8),
           _row("k0", "gated_crc", "report", False, 0.9, dataset="src")]
    sp = _write(tmp_path, "src.jsonl", "src", src)
    tgt = [_row("r0", "gated", "report", True, 0.95, dataset="tgt"),
           _row("r1", "gated", "report", False, 0.1, dataset="tgt"),
           _row("k0", "gated_crc", "report", True, 0.9, dataset="tgt")]
    tp = _write(tmp_path, "tgt.jsonl", "tgt", tgt,
                thresholds={"gated_crc": 0.5})
    assert main(["--source", sp, "--targets", tp,
                 "--joint", "answer_abstain"]) == 0
    out = capsys.readouterr().out
    assert "abstain-all" in out
    (aline,) = _shift_lines(out, "answer_abstain")
    assert "| 0.000 | n/a | n/a |" in aline
    assert "abstains on everything" in aline


def test_target_calibrate_sentinel_isolated(tmp_path, capsys):
    # Sentinel target-calibrate rows (extreme confs, all wrong at 0.99)
    # must not reach any figure: calibration comes from the source file.
    src = [_row(f"c{i}", "gated", "calibrate", True, a)
           for i, a in enumerate([0.9, 0.8, 0.7])]
    src.append(_row("k0", "gated_crc", "report", True, 0.9, dataset="src"))
    sp = _write(tmp_path, "src.jsonl", "src", src)
    rep = [_row("r0", "gated", "report", True, 0.95, dataset="tgt"),
           _row("r1", "gated", "report", False, 0.6, dataset="tgt"),
           _row("r2", "gated", "report", False, 0.1, dataset="tgt")]
    sentinel = [_row(f"s{i}", "gated", "calibrate", False, 0.99, dataset="tgt")
                for i in range(3)]
    crc = [_row("k0", "gated_crc", "report", True, 0.9, dataset="tgt")]
    tp = _write(tmp_path, "tgt.jsonl", "tgt", rep + sentinel + crc,
                thresholds={"gated_crc": 0.5})
    assert main(["--source", sp, "--targets", tp,
                 "--joint", "answer_abstain"]) == 0
    out = capsys.readouterr().out
    (aline,) = _shift_lines(out, "answer_abstain")
    # Report-only hand computation: answered = {0.95}, correct.
    assert "| tgt | 3 |" in aline
    assert "| 0.333 | 0.000 | 0.000 |" in aline


def test_missing_file_fails(tmp_path, capsys):
    src = [_row("c0", "gated", "calibrate", True, 0.9),
           _row("k0", "gated_crc", "report", True, 0.9, dataset="src")]
    sp = _write(tmp_path, "src.jsonl", "src", src)
    missing = str(tmp_path / "nope.jsonl")
    assert main(["--source", sp, "--targets", missing]) == 2
    assert "missing file" in capsys.readouterr().err


def test_missing_crc_spec_fails(tmp_path, capsys):
    # Full-file guard passes (gated_crc present) but no calibrate gated
    # rows exist to calibrate from.
    src = [_row("r0", "gated", "report", True, 0.9, dataset="src"),
           _row("k0", "gated_crc", "report", True, 0.9, dataset="src")]
    sp = _write(tmp_path, "src.jsonl", "src", src)
    tgt = [_row("r0", "gated", "report", True, 0.9, dataset="tgt"),
           _row("k0", "gated_crc", "report", True, 0.9, dataset="tgt")]
    tp = _write(tmp_path, "tgt.jsonl", "tgt", tgt)
    assert main(["--source", sp, "--targets", tp]) == 2
    assert "missing CRC spec" in capsys.readouterr().err


def test_checkpoint_target_refused(tmp_path, capsys):
    src = [_row("c0", "gated", "calibrate", True, 0.9),
           _row("k0", "gated_crc", "report", True, 0.9, dataset="src")]
    sp = _write(tmp_path, "src.jsonl", "src", src)
    chk = [_row("r0", "gated", "report", True, 0.9, dataset="tgt"),
           _row("r1", "fixed", "report", True, 0.0, dataset="tgt")]
    cp = _write(tmp_path, "chk.jsonl", "tgt", chk)
    assert main(["--source", sp, "--targets", cp]) == 2
    assert "checkpoint" in capsys.readouterr().err


def test_checkpoint_source_refused(tmp_path, capsys):
    chk = [_row("c0", "gated", "calibrate", True, 0.9, dataset="src")]
    cp = _write(tmp_path, "chk.jsonl", "src", chk)
    assert main(["--source", cp, "--targets", cp]) == 2
    assert "checkpoint" in capsys.readouterr().err


def test_missing_signal_skipped_not_invented(tmp_path, capsys):
    # One target report row lacks the abstain block: skipped, counted in
    # notes, denominator is the 1 scored row (0.95 T -> cov 1.0, risk 0.0).
    src = [_row(f"c{i}", "gated", "calibrate", True, a)
           for i, a in enumerate([0.9, 0.8, 0.7])]
    src.append(_row("k0", "gated_crc", "report", True, 0.9, dataset="src"))
    sp = _write(tmp_path, "src.jsonl", "src", src)
    ok = _row("r0", "gated", "report", True, 0.95, dataset="tgt")
    nosig = _row("r1", "gated", "report", False, 0.6, dataset="tgt")
    nosig["abstain"] = None
    crc = [_row("k0", "gated_crc", "report", True, 0.9, dataset="tgt")]
    tp = _write(tmp_path, "tgt.jsonl", "tgt", [ok, nosig] + crc,
                thresholds={"gated_crc": 0.5})
    assert main(["--source", sp, "--targets", tp,
                 "--joint", "answer_abstain"]) == 0
    out = capsys.readouterr().out
    (aline,) = _shift_lines(out, "answer_abstain")
    assert "| 1.000 | 0.000 | 0.000 |" in aline
    assert "1 rows missing abstain signal skipped" in aline
