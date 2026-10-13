"""Aggregator proof tests (packet V03-A9). Offline only, no live calls.

Synthetic result files are written directly in the runner's row format into
tmp_path (no repo fixtures, no /mnt or D:\\ literals). Every asserted number
is a literal computed by hand in the test — never by calling the
aggregator's own helpers. AURC/ECE floats are asserted as present-or-n/a
patterns only (they come from vendored/tested code with its own tests);
all count-based cells are exact literals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.suite_table import markdown_rows, table_for_results  # noqa: E402


def _row(qid, arm, split, pred, gold, em, f1, conf, rounds=1,
         outcome="answer", sel=2, toks=100, dataset="tiny"):
    r = {"question_id": qid, "question": "q?", "gold": gold,
         "outcome": outcome, "answer": pred, "prediction": pred,
         "source": "pipeline", "arm": arm, "split": split,
         "dataset": dataset, "hops": None, "em": em, "f1": f1,
         "jev_calls": 3, "jev_input_tokens": toks,
         "selection_calls": sel,
         "selection": {"rounds": [{"jev_calls": sel,
                                   "jev_input_tokens": toks,
                                   "new_candidates": ["x"]}],
                        "calls": sel},
         "wall_ms": 1.0}
    if arm == "fixed":
        r["sufficiency"] = None
        r["abstain"] = None
        r["rounds_used"] = rounds
    else:
        r["sufficiency"] = {"record": {"confidence": conf,
                                       "rounds_used": rounds}}
        r["abstain"] = {"record": {"confidence": conf}}
    return r


def _write(tmp_path, name, dataset, rows, thresholds=None):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__header__": {"dataset": dataset}}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if thresholds is not None:
        with open(str(p)[:-len(".jsonl")] + ".trace.json", "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"abst_thresholds": thresholds}))
    return str(p)


def test_per_arm_aggregation_hand_computed(tmp_path):
    # Gated report x4: confs 0.9(T) 0.8(F) 0.2(F) 0.1(F), threshold 0.5.
    # By hand: answered = first two -> coverage 1/2; wrong among answered
    # = 1/2 -> selective 0.5; marginal = 0.5*0.5 = 0.25. EM 1, F1 mean
    # (1+0+0+0)/4 = 0.25. Rounds 1+1+2+3 = 7. Calls 4*3 = 12, toks 400,
    # $ = 400e-6*0.041 = 0.0000164 -> "0.0000". Selection live 4*2 = 8.
    gated = [
        _row("g0", "gated", "report", "a", "a", True, 1.0, 0.9, rounds=1),
        _row("g1", "gated", "report", "b", "B!", False, 0.0, 0.8, rounds=1),
        _row("g2", "gated", "report", "c", "C!", False, 0.0, 0.2, rounds=2),
        _row("g3", "gated", "report", "d", "D!", False, 0.0, 0.1, rounds=3),
    ]
    # Fixed report x2, one wrong: coverage 1.0, sel = marg = 1/2, no confidences.
    fixed = [
        _row("f0", "fixed", "report", "a", "a", True, 1.0, 0.0, rounds=3),
        _row("f1", "fixed", "report", "b", "B!", False, 0.0, 0.0, rounds=3),
    ]
    p = _write(tmp_path, "r.jsonl", "tiny", gated + fixed,
               {"gated": 0.5})
    table = { (t["arm"], t["split"]): t for t in table_for_results(p) }
    g = table[("gated", "report")]["metrics"]
    assert (g["n"], g["n_served"], g["em_overall"]) == (4, 4, 1)
    assert g["f1_mean"] == pytest.approx(0.25)
    assert g["coverage"] == pytest.approx(0.5)
    assert g["selective_risk"] == pytest.approx(0.5)
    assert g["marginal_risk"] == pytest.approx(0.25)
    assert g["rounds"] == 7
    assert (g["jev_calls"], g["jev_input_tokens"]) == (12, 400)
    assert g["sel_live"] == 8 and g["risk_source"] == "crc_readout"
    assert g["aurc"] is not None and g["ece"] is not None
    f = table[("fixed", "report")]["metrics"]
    assert (f["n"], f["em_overall"]) == (2, 1)
    assert f["coverage"] == pytest.approx(1.0)
    assert f["selective_risk"] == pytest.approx(0.5)
    assert f["marginal_risk"] == pytest.approx(0.5)
    assert f["risk_source"] == "definitional_no_gate"
    md = markdown_rows([table_for_results(p)])
    assert "| 0.500 |" in md  # coverage + selective risk cells
    assert "| 0.250 |" in md  # marginal risk cell
    assert "| 0.0000 |" in md  # $400 toks at suite rate
    assert "n/a" in md  # fixed AURC/ECE cells say so


def test_calibrate_report_never_mixed(tmp_path):
    # Sentinel: calibrate rows all EM, report rows all wrong. The report
    # cell must read 0/2 regardless of the calibrate content.
    cals = [_row(f"c{i}", "gated", "calibrate", "a", "a", True, 1.0, 0.9)
            for i in range(2)]
    reps = [_row(f"r{i}", "gated", "report", "b", "B!", False, 0.0, 0.1)
            for i in range(2)]
    p = _write(tmp_path, "r.jsonl", "tiny", cals + reps, {"gated": 0.5})
    md = markdown_rows([table_for_results(p)])
    cal_line = next(l for l in md.splitlines() if "| calibrate |" in l)
    rep_line = next(l for l in md.splitlines() if "| report |" in l)
    assert "2/2" in cal_line and "0/2" in rep_line


def test_crc_rows_and_shift_files(tmp_path):
    # CRC lives in the SAME file under its own arm (no separate-run path
    # exists in this design): confs 0.9(T) 0.7(F) at 0.5 -> answered both,
    # coverage 1.0, selective = marginal... wrong|answered = 1/2,
    # marginal = 1.0*0.5 = 0.5.
    crc = [_row("k0", "gated_crc", "report", "a", "a", True, 1.0, 0.9),
           _row("k1", "gated_crc", "report", "b", "B!", False, 0.0, 0.7)]
    p = _write(tmp_path, "r.jsonl", "tiny", crc,
               {"gated_crc": 0.5})
    table = {(t["arm"], t["split"]): t for t in table_for_results(p)}
    m = table[("gated_crc", "report")]["metrics"]
    assert (m["coverage"], m["selective_risk"], m["marginal_risk"]) == \
        pytest.approx((1.0, 0.5, 0.5))
    # Checkpoint-style file without CRC: no gated_crc row printed (absence,
    # never zeros).
    p2 = _write(tmp_path, "c.jsonl", "tiny",
                [_row("g0", "gated", "report", "a", "a", True, 1.0, 0.9)],
                {"gated": 0.5})
    assert "gated_crc" not in markdown_rows([table_for_results(p2)])
    # Shift-test shape = two files aggregated: both datasets appear.
    p3 = _write(tmp_path, "s.jsonl", "shifted",
                [_row("s0", "gated", "calibrate", "a", "a", True, 1.0, 0.9)],
                {"gated": 0.5})
    md = markdown_rows([table_for_results(p), table_for_results(p3)])
    assert "| tiny | gated_crc | report |" in md
    assert "| shifted | gated | calibrate |" in md


def test_missing_arm_is_absent_not_zero(tmp_path):
    p = _write(tmp_path, "r.jsonl", "tiny",
               [_row("g0", "gated", "report", "a", "a", True, 1.0, 0.9)],
               {"gated": 0.5})
    md = markdown_rows([table_for_results(p)])
    assert "fixed" not in md  # omitted, never a silent 0 row


def test_truncated_file_fails_loudly(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"__header__": {}}) + '\n{"question_id": ',
                 encoding="utf-8")
    with pytest.raises(ValueError):
        table_for_results(str(p))


def test_checkpoint_where_full_expected(tmp_path):
    p = _write(tmp_path, "r.jsonl", "tiny",
               [_row("c0", "gated", "calibrate", "a", "a", True, 1.0, 0.9)],
               {"gated": 0.5})
    md = markdown_rows([table_for_results(p)])
    assert "report" not in md  # no report rows exist to print


def test_empty_file_prints_no_numbers(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("", encoding="utf-8")
    md = markdown_rows([table_for_results(str(p))])
    assert len(md.splitlines()) == 2  # header lines only, zero data rows


def test_error_rows_noted_not_hidden(tmp_path):
    ok = _row("g0", "gated", "report", "a", "a", True, 1.0, 0.9)
    bad = dict(ok)
    bad.update({"question_id": "g1", "outcome": "error", "answer": "",
                "prediction": "", "em": False, "f1": 0.0,
                "sufficiency": None, "abstain": None,
                "jev_calls": 0, "jev_input_tokens": 0, "selection_calls": 0})
    p = _write(tmp_path, "r.jsonl", "tiny", [ok, bad], {"gated": 0.5})
    md = markdown_rows([table_for_results(p)])
    assert "1 errors" in md and "50.0" in md and "1/2" in md


def test_duplicate_ids_probe(tmp_path):
    r1 = _row("q0", "gated", "report", "WRONG", "a", False, 0.0, 0.1)
    r2 = _row("q0", "gated", "report", "a", "a", True, 1.0, 0.9)
    p = _write(tmp_path, "r.jsonl", "tiny", [r1, r2], {"gated": 0.5})
    table = {(t["arm"], t["split"]): t for t in table_for_results(p)}
    # Correct (last-wins, mirroring run_suite): n == 1.
    assert table[("gated", "report")]["metrics"]["n"] == 1


def test_redone_row_later_answer_wins(tmp_path):
    # A redone row that changes the answer: the LATER row wins, n stays 1.
    r1 = _row("q0", "gated", "report", "WRONG", "a", False, 0.0, 0.1)
    r2 = _row("q0", "gated", "report", "a", "a", True, 1.0, 0.9)
    p = _write(tmp_path, "r.jsonl", "tiny", [r1, r2], {"gated": 0.5})
    m = {(t["arm"], t["split"]): t
         for t in table_for_results(p)}[("gated", "report")]["metrics"]
    assert m["n"] == 1
    assert m["em_overall"] == 1  # the later (correct) row, not the first
    md = markdown_rows([table_for_results(p)])
    assert "1/1" in md
