"""Tests for jevrag.benchmarks.hotpotqa — data loading and correctness labels."""

import pytest

from jevrag.benchmarks import hotpotqa


def test_dataset_loads_with_frozen_splits():
    questions = hotpotqa.load_questions()
    assert len(questions) == 1000
    splits = {q["split"] for q in questions}
    assert splits == {"val", "test"}
    types = {q["type"] for q in questions}
    assert types <= {"bridge", "comparison"}


def test_split_filtering():
    val = hotpotqa.load_questions(split="val")
    test = hotpotqa.load_questions(split="test")
    assert all(q["split"] == "val" for q in val)
    assert all(q["split"] == "test" for q in test)
    assert len(val) + len(test) == 1000


def test_split_lookup_covers_every_question():
    lookup = hotpotqa.split_lookup()
    assert len(lookup) == 1000
    assert set(lookup.values()) == {"val", "test"}


def test_label_record_em_and_f1():
    rec = {"prediction": "Yes, they were.", "gold": "yes"}
    out = hotpotqa.label_record(rec)
    assert out["correct"] == 0  # EM is exact after normalization
    assert out["f1"] > 0.0

    rec2 = {"prediction": "The Beatles", "gold": "beatles"}
    out2 = hotpotqa.label_record(rec2)
    assert out2["correct"] == 1  # articles + case normalized away
    assert out2["f1"] == pytest.approx(1.0)


def test_label_record_does_not_mutate():
    rec = {"prediction": "a", "gold": "a"}
    hotpotqa.label_record(rec)
    assert "correct" not in rec


def test_evaluator_reexports_are_rag_gates():
    assert hotpotqa.exact_match.__module__ == "rag_gate_evaluator"
    assert hotpotqa.f1_score.__module__ == "rag_gate_evaluator"
