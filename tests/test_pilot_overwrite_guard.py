"""Tests for the pilot runner's overwrite guard + fail-closed generator.

Offline only: no network, no Jev, no OpenRouter. Covers packet V03-A3 §1
(the A2 overwrite cost 47% of that packet's spend) and §2 (fail closed on
the first generator error, never silent fallback).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline_docbench_pilot import (  # noqa: E402
    GeneratorHalted,
    OverwriteRefused,
    checked_out_paths,
    fail_closed_answer_fn,
)


def test_guard_refuses_nonempty_outputs_without_overwrite(tmp_path):
    rows = tmp_path / "run.jsonl"
    rows.write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(OverwriteRefused, match="refusing to overwrite"):
        checked_out_paths(str(tmp_path / "run"), False)


def test_guard_allows_overwrite_flag(tmp_path):
    rows = tmp_path / "run.jsonl"
    rows.write_text('{"a": 1}\n', encoding="utf-8")
    paths = checked_out_paths(str(tmp_path / "run"), True)
    assert paths["rows"] == rows


def test_guard_allows_missing_and_empty_files(tmp_path):
    paths = checked_out_paths(str(tmp_path / "fresh"), False)
    assert not paths["rows"].exists()
    empty = tmp_path / "e.jsonl"
    empty.write_text("", encoding="utf-8")
    checked_out_paths(str(tmp_path / "e"), False)  # empty: crash remnant, ok


def test_fail_closed_raises_on_first_error_with_status():
    class E402(Exception):
        status_code = 402

    def boom(question, evidence):
        raise E402("Insufficient credits")

    fn = fail_closed_answer_fn(boom, tag="openrouter:test")
    with pytest.raises(GeneratorHalted, match="402"):
        fn("q?", [])


def test_fail_closed_raises_on_any_generator_error():
    def boom(question, evidence):
        raise RuntimeError("connection reset")

    fn = fail_closed_answer_fn(boom, tag="openrouter:test")
    with pytest.raises(GeneratorHalted, match="halted"):
        fn("q?", [])


def test_fail_closed_passes_through_success():
    def ok(question, evidence):
        return "an answer", {"input_tokens": 5, "output_tokens": 3}

    fn = fail_closed_answer_fn(ok, tag="openrouter:test")
    assert fn("q?", []) == ("an answer",
                            {"input_tokens": 5, "output_tokens": 3})
