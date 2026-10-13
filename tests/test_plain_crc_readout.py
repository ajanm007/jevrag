"""Plain gated_crc readout proof tests (packet V03-A11). Offline, no live calls.

Synthetic result files go to tmp_path (no repo fixtures, no absolute-path
literals). Every asserted number is a hand-computed literal — never produced
via the scripts' own helpers (no crc_readout / split_metrics imports here).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.shift_test import main as shift_main  # noqa: E402
from scripts.suite_table import (  # noqa: E402
    REBUILT_LABEL,
    RECORDED_LABEL,
    markdown_rows,
    table_for_results,
)


def _row(qid, arm, split, em, conf, outcome=None, dataset="tiny"):
    outcome = ("answer" if em else "no_answer") if outcome is None else outcome
    if arm == "gated_crc" and outcome not in ("answer", "no_answer"):
        raise ValueError("gated_crc test rows need a live outcome")
    r = {"question_id": qid, "question": f"q-{qid}?", "gold": "g",
         "outcome": outcome if arm != "fixed" else "answer",
         "answer": "p", "prediction": "p", "source": "pipeline",
         "arm": arm, "split": split, "dataset": dataset, "hops": None,
         "em": em, "f1": 1.0 if em else 0.0,
         "jev_calls": 3, "jev_input_tokens": 100, "selection_calls": 2,
         "selection": {"rounds": [{"jev_calls": 2, "jev_input_tokens": 100,
                                   "new_candidates": ["x"]}], "calls": 2},
         "wall_ms": 1.0}
    if arm == "fixed":
        r["sufficiency"] = None
        r["abstain"] = None
    else:
        r["sufficiency"] = {"record": {"confidence": conf, "rounds_used": 1}}
        r["abstain"] = {"record": {"confidence": conf}}
    return r


def _write(tmp_path, name, dataset, rows, alpha=0.3, thresholds=None):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__header__": {"dataset": dataset,
                                           "alpha": alpha}}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(str(p)[:-len(".jsonl")] + ".trace.json", "w",
              encoding="utf-8") as f:
        f.write(json.dumps({"abst_thresholds": thresholds or {}}))
    return str(p)


def _crc_line(md):
    return next(l for l in md.splitlines() if "| gated_crc |" in l)


def test_sub_eps_ties_yield_plain_numbers(tmp_path):
    # 0.5000001 vs 0.5 differ by 1e-7 < 2*TIE_BREAK_EPS: crc_readout refuses
    # these rows today. Plain readout at recorded 0.5: answered = {0.6,
    # 0.5000001} -> coverage 2/3, selective 1/2, marginal 1/3.
    rows = [_row("k0", "gated_crc", "report", True, 0.6, "answer"),
            _row("k1", "gated_crc", "report", False, 0.5000001, "answer"),
            _row("k2", "gated_crc", "report", False, 0.4, "no_answer")]
    p = _write(tmp_path, "r.jsonl", "tiny", rows,
               thresholds={"gated_crc": 0.5})
    table = {(t["arm"], t["split"]): t for t in table_for_results(p)}
    m = table[("gated_crc", "report")]["metrics"]
    assert (m["coverage"], m["selective_risk"], m["marginal_risk"]) == (
        2 / 3, 1 / 2, 1 / 3)
    assert m["risk_source"] == RECORDED_LABEL
    line = _crc_line(markdown_rows([table_for_results(p)]))
    assert "| 0.667 | 0.500 | 0.333 |" in line
    assert RECORDED_LABEL in line


def test_threshold_boundary_is_answered(tmp_path):
    # Rule: answered iff recorded confidence >= threshold. >= matches the
    # live arm (answer_abstain.action_for passes at equality, as does the
    # CRC adapter's decision-space comparison). Rows at exactly 0.5 answer;
    # 0.49 does not -> coverage 2/3, selective 1/2, marginal 1/3.
    rows = [_row("k0", "gated_crc", "report", True, 0.5, "answer"),
            _row("k1", "gated_crc", "report", False, 0.5, "answer"),
            _row("k2", "gated_crc", "report", True, 0.49, "no_answer")]
    p = _write(tmp_path, "r.jsonl", "tiny", rows,
               thresholds={"gated_crc": 0.5})
    line = _crc_line(markdown_rows([table_for_results(p)]))
    assert "| 0.667 | 0.500 | 0.333 |" in line


def test_zero_answered_risks_na_never_zero(tmp_path):
    # Threshold 0.9 answers nothing: coverage 0.000, risks n/a (never 0.0).
    rows = [_row("k0", "gated_crc", "report", False, 0.5, "no_answer"),
            _row("k1", "gated_crc", "report", True, 0.1, "no_answer")]
    p = _write(tmp_path, "r.jsonl", "tiny", rows,
               thresholds={"gated_crc": 0.9})
    line = _crc_line(markdown_rows([table_for_results(p)]))
    assert "| 0.000 | n/a | n/a |" in line


def test_stale_trace_rebuild_verified(tmp_path):
    # Halted-file shape: trace gated_crc None. Calibrate all-correct at
    # 0.9/0.8/0.7 -> k_hat=3, threshold ~=0.7 (gaps >> eps). Report rows:
    # 0.95(T,answer) 0.6(F,no_answer) 0.1(F,no_answer): rebuilt predicts 1
    # served, rows show 1 -> row prints, coverage 1/3, risks 0.0.
    cal = [_row(f"c{i}", "gated", "calibrate", True, a)
           for i, a in enumerate([0.9, 0.8, 0.7])]
    rep = [_row("k0", "gated_crc", "report", True, 0.95, "answer"),
           _row("k1", "gated_crc", "report", False, 0.6, "no_answer"),
           _row("k2", "gated_crc", "report", False, 0.1, "no_answer")]
    p = _write(tmp_path, "r.jsonl", "tiny", cal + rep,
               thresholds={"gated": 0.5, "gated_crc": None})
    line = _crc_line(markdown_rows([table_for_results(p)]))
    assert "| 0.333 | 0.000 | 0.000 |" in line
    assert REBUILT_LABEL in line


def test_rebuild_mismatch_prints_no_row(tmp_path):
    # Same calibration, but the rows' live outcomes contradict the rebuilt
    # threshold (0.6 row served although below ~=0.7): predicted 1 served
    # vs 2 shown -> no CRC row, n/a with the reason, never invented numbers.
    cal = [_row(f"c{i}", "gated", "calibrate", True, a)
           for i, a in enumerate([0.9, 0.8, 0.7])]
    rep = [_row("k0", "gated_crc", "report", True, 0.95, "answer"),
           _row("k1", "gated_crc", "report", False, 0.6, "answer"),
           _row("k2", "gated_crc", "report", False, 0.1, "no_answer")]
    p = _write(tmp_path, "r.jsonl", "tiny", cal + rep,
               thresholds={"gated": 0.5, "gated_crc": None})
    line = _crc_line(markdown_rows([table_for_results(p)]))
    assert "| n/a | n/a | n/a |" in line
    assert "no CRC row" in line


def test_rebuild_abstain_all_zero_served(tmp_path):
    # Calibrate all-wrong at 0.9/0.8 -> abstain-all (+inf): predicts 0
    # served, rows show 0 -> row prints, coverage 0.000, risks n/a.
    cal = [_row("c0", "gated", "calibrate", False, 0.9),
           _row("c1", "gated", "calibrate", False, 0.8)]
    rep = [_row("k0", "gated_crc", "report", False, 0.5, "no_answer"),
           _row("k1", "gated_crc", "report", True, 0.1, "no_answer")]
    p = _write(tmp_path, "r.jsonl", "tiny", cal + rep,
               thresholds={"gated": 0.5, "gated_crc": None})
    line = _crc_line(markdown_rows([table_for_results(p)]))
    assert "| 0.000 | n/a | n/a |" in line
    assert REBUILT_LABEL in line


def test_partial_run_note_and_actual_n(tmp_path):
    # Halted shape (150 calibrate, 2 report): the table prints the actual
    # report n and names the shortfall; a complete file gets no such note.
    cal = [_row(f"c{i}", "gated", "calibrate", True, 0.9) for i in range(3)]
    rep = [_row("r0", "gated", "report", True, 0.9),
           _row("r1", "fixed", "report", True, 0.0)]
    p = _write(tmp_path, "h.jsonl", "tiny", cal + rep,
               thresholds={"gated": 0.5})
    md = markdown_rows([table_for_results(p)])
    assert "partial run: report n=1 < calibrate n=3" in md
    p2 = _write(tmp_path, "f.jsonl", "full",
                [_row("c0", "gated", "calibrate", True, 0.9),
                 _row("r0", "gated", "report", True, 0.9)],
                thresholds={"gated": 0.5})
    assert "partial run" not in markdown_rows([table_for_results(p2)])


def test_shift_context_uses_plain_readout(tmp_path, capsys):
    # End to end through shift_test: a stale-trace target's context row uses
    # the rebuilt plain readout (same numbers as the main table).
    src = [_row(f"c{i}", "gated", "calibrate", True, a, dataset="src")
           for i, a in enumerate([0.9, 0.8, 0.7])]
    src.append(_row("k0", "gated_crc", "report", True, 0.9, "answer",
                    dataset="src"))
    sp = _write(tmp_path, "src.jsonl", "src", src,
                thresholds={"gated_crc": 0.5})
    tgt = [_row(f"c{i}", "gated", "calibrate", True, a, dataset="tgt")
           for i, a in enumerate([0.9, 0.8, 0.7])]
    tgt += [_row("r0", "gated", "report", True, 0.95, "answer",
                 dataset="tgt"),
            _row("k0", "gated_crc", "report", True, 0.95, "answer",
                 dataset="tgt"),
            _row("k1", "gated_crc", "report", False, 0.6, "no_answer",
                 dataset="tgt"),
            _row("k2", "gated_crc", "report", False, 0.1, "no_answer",
                 dataset="tgt")]
    tp = _write(tmp_path, "tgt.jsonl", "tgt", tgt,
                thresholds={"gated": 0.5, "gated_crc": None})
    assert shift_main(["--source", sp, "--targets", tp,
                       "--joint", "answer_abstain"]) == 0
    out = capsys.readouterr().out
    ctx = [l for l in out.splitlines() if "gated_crc-infile" in l]
    assert len(ctx) == 1
    assert "| 0.333 | 0.000 | 0.000 |" in ctx[0]
    assert REBUILT_LABEL in ctx[0]
