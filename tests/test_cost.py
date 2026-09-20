"""Tests for jevrag.eval.cost — token/latency accounting."""

import pytest

from jevrag.eval.cost import JEV_USD_PER_MTOK_INPUT, format_cost_table, summarize_cost


def _rec(lat, tok, rounds):
    return {"latency_ms": lat, "input_tokens": tok, "rounds_used": rounds}


def test_summarize_basic():
    s = summarize_cost([_rec(100.0, 1000, 1), _rec(300.0, 3000, 3)])
    assert s["n_records"] == 2
    assert s["latency_ms"]["mean"] == pytest.approx(200.0)
    assert s["input_tokens"]["total"] == 4000
    assert s["input_tokens"]["per_question"]["mean"] == pytest.approx(2000.0)
    assert s["rounds_used"]["max"] == pytest.approx(3.0)
    assert s["jev_cost_usd"]["total"] == pytest.approx(
        4000 / 1e6 * JEV_USD_PER_MTOK_INPUT
    )


def test_missing_fields_tolerated_and_counted():
    # Gate-free baseline records may lack token/latency detail; aggregates must
    # report coverage rather than silently computing over a subset.
    s = summarize_cost([
        {"latency_ms": None, "input_tokens": None, "rounds_used": 3},
        _rec(50.0, 500, 1),
    ])
    assert s["n_records"] == 2
    assert s["n_with_latency"] == 1
    assert s["n_with_tokens"] == 1
    assert s["latency_ms"]["mean"] == pytest.approx(50.0)


def test_empty_records():
    s = summarize_cost([])
    assert s["n_records"] == 0
    assert s["latency_ms"]["mean"] is None
    assert s["jev_cost_usd"]["total"] == 0.0


def test_format_table_renders():
    s = summarize_cost([_rec(100.0, 1000, 2)])
    table = format_cost_table(s)
    assert "latency_ms" in table and "input_tokens" in table
    assert "est. Jev cost" in table
