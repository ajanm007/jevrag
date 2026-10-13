"""Degeneracy-guard tests (packet V03-A5 §3). Offline only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_hotpotqa_arms import degeneracy_fraction  # noqa: E402


def _row(new_lists):
    return {"selection": {"rounds": [{"new_candidates": n} for n in new_lists]}}


def test_all_later_rounds_add_something():
    rows = [_row([["a"], ["b"], ["c"]]), _row([["x"], ["y"]])]
    assert degeneracy_fraction(rows) == 0.0


def test_fully_degenerate_flags_one():
    rows = [_row([["a"], [], []]), _row([["x"], [], []])]
    assert degeneracy_fraction(rows) == 1.0


def test_single_round_questions_do_not_count():
    assert degeneracy_fraction([_row([["a"]])]) == 0.0
    assert degeneracy_fraction([]) == 0.0


def test_partial_degeneracy_fraction():
    rows = [_row([["a"], ["b"], []])]  # 1 of 2 later rounds empty
    assert degeneracy_fraction(rows) == 0.5
