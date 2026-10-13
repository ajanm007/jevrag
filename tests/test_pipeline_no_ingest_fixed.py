"""Ingestion-free builds + fixed-round arm (packet V03-A4, suite prerequisite).

Offline only: stub Decisions, stub generator, tiny in-memory paragraph pool.
No network, no dataset files.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from jevrag.decision import DecisionResult, StubDecision, TypedQuestion
from jevrag.pipeline import (
    PipelineConfig,
    answer_question_fixed,
    build_pipeline,
    build_pipeline_from_chunks,
    run_pipeline,
    run_pipeline_fixed,
)

PAGES = [
    {"page": 1, "text": "Alpha one.\n\nAlpha two talks about retrievers."},
    {"page": 2, "text": "Beta one covers generators.\n\nBeta two ends it."},
]

POOL = [
    {"title": f"p{i}", "text": text}
    for i, text in enumerate([
        "keep retrievers here", "drop off-topic noise",
        "keep generators here", "drop random filler",
        "keep both retrievers and generators", "drop more noise"])]

GATED_KEYS = {"question_id", "question", "gold", "outcome", "answer",
              "prediction", "source", "sufficiency", "abstain", "selection",
              "cache", "selection_calls", "jev_calls", "jev_input_tokens",
              "jev_cost_usd", "thresholds", "wall_ms"}


class SplitOnC(StubDecision):
    def ask(self, state: Any, questions: list[TypedQuestion]) -> DecisionResult:
        after = str(state.get("after", ""))
        conf = 0.9 if after.startswith("Beta one") else 0.1
        return StubDecision(confidences={q.name: conf
                                         for q in questions}).ask(state, questions)


class SelectKeep(StubDecision):
    def ask(self, state: Any, questions: list[TypedQuestion]) -> DecisionResult:
        conf = 0.9 if "keep" in str(state.get("passage", "")) else 0.1
        return StubDecision(confidences={q.name: conf
                                         for q in questions}).ask(state, questions)


class Counting(StubDecision):
    def __init__(self, inner: StubDecision) -> None:
        super().__init__(confidences=dict(inner.confidences))
        self._inner = inner
        self.calls = 0

    def ask(self, state: Any, questions: list[TypedQuestion]) -> DecisionResult:
        self.calls += 1
        return self._inner.ask(state, questions)


def scripted(conf: float) -> StubDecision:
    return StubDecision(confidences={"sufficient": conf, "grounded": conf,
                                      "relevant": conf, "split": conf})


def _config(**over: Any) -> PipelineConfig:
    base: dict[str, Any] = {
        "chunk_decision": SplitOnC(),
        "selection_decision": SelectKeep(),
        "sufficiency_decision": scripted(over.pop("suff_conf", 0.95)),
        "abstain_decision": scripted(0.8),
        "cache_decision": scripted(0.1),
        "answer_fn": lambda q, ev: "test answer",
        "per_round_k": 2,
        "max_rounds": 3,
    }
    base.update(over)
    return PipelineConfig(**base)


def _qs():
    return [
        {"id": "q1", "question": "What talks about retrievers?",
         "answer": "Alpha two.", "type": "bridge"},
        {"id": "q2", "question": "What covers generators today?",
         "answer": "Beta one.", "type": "bridge"},
    ]


def test_from_chunks_makes_zero_chunk_boundary_calls():
    counting = Counting(SplitOnC())
    cfg = _config(chunk_decision=counting)
    ctx = build_pipeline_from_chunks(cfg, "pool", copy.deepcopy(POOL))
    assert counting.calls == 0
    assert ctx.ingestion_trace["mode"] == "ingestion_skipped_paragraph_pool"
    assert ctx.ingestion_trace["jev_calls"] == 0
    assert ctx.ingestion_trace["jev_input_tokens"] == 0
    assert len(ctx.chunks) == len(POOL)


def test_from_chunks_rejects_bad_pool_entries():
    with pytest.raises(ValueError, match="empty"):
        build_pipeline_from_chunks(_config(), "pool", [])
    with pytest.raises(ValueError, match="title, text"):
        build_pipeline_from_chunks(_config(), "pool",
                                   [{"title": "no-text"}])


def test_index_built_once_shared_across_questions():
    ctx = build_pipeline_from_chunks(_config(), "pool",
                                     copy.deepcopy(POOL))
    first_index = ctx.index
    outs = run_pipeline(ctx, _qs())
    assert len(outs) == 2
    assert ctx.index is first_index
    assert set(ctx.evidence_log) == {q["question"] for q in _qs()}


def test_gated_and_fixed_rows_same_shape():
    ctx = build_pipeline_from_chunks(_config(), "pool",
                                     copy.deepcopy(POOL))
    gated = run_pipeline(ctx, _qs())
    fixed = run_pipeline_fixed(ctx, _qs(), n_rounds=3,
                               generator_tag="stub:test")
    assert len(gated) == len(fixed) == 2
    for g, f in zip(gated, fixed):
        assert GATED_KEYS <= set(g), GATED_KEYS - set(g)
        assert GATED_KEYS <= set(f), GATED_KEYS - set(f)
        assert f["outcome"] == "answer" and f["answer"] == f["prediction"]
        assert f["source"] == "pipeline-fixed"
        assert f["sufficiency"] is None and f["abstain"] is None
        assert f["rounds_used"] == 3 and f["generator"] == "stub:test"
        assert f["fixed"] == {"n_rounds": 3, "rounds_used": 3}


def test_fixed_arm_always_runs_exactly_n_rounds():
    # Gate would stop after round 1 (0.99 >> threshold); fixed still runs N.
    ctx = build_pipeline_from_chunks(_config(suff_conf=0.99), "pool",
                                     copy.deepcopy(POOL))
    gated = run_pipeline(ctx, _qs()[:1])
    assert gated[0]["sufficiency"]["record"]["rounds_used"] == 1
    for n in (1, 2, 3):
        ctx = build_pipeline_from_chunks(_config(suff_conf=0.99), "pool",
                                         copy.deepcopy(POOL))
        out = run_pipeline_fixed(ctx, _qs()[:1], n_rounds=n)[0]
        assert out["rounds_used"] == n
        assert len(ctx.selection_trace) == n  # one retrieve call per round
    with pytest.raises(ValueError, match="n_rounds"):
        answer_question_fixed(_qs()[0], ctx, n_rounds=0)


def _scrub(node: Any) -> Any:
    drop = {"latency_ms", "wall_ms", "input_tokens", "jev_calls",
            "jev_input_tokens", "jev_cost_usd", "selection_calls", "calls",
            "cache_hits", "cache_hit", "cache_enabled", "n_cache_hits",
            "n_live_calls", "n_votes", "confidences", "pool_ids", "cache"}
    if isinstance(node, dict):
        return {k: _scrub(v) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_scrub(v) for v in node]
    return node


def test_memo_equivalence_in_chunks_mode():
    off = build_pipeline_from_chunks(_config(), "pool",
                                     copy.deepcopy(POOL))
    on = build_pipeline_from_chunks(
        _config(memoize_selection=True, memoize_chunk=True), "pool",
        copy.deepcopy(POOL))
    assert _scrub(run_pipeline(off, _qs())) == \
        _scrub(run_pipeline(on, _qs()))
    off_calls = sum(o["selection_calls"] for o in run_pipeline(off, _qs()))
    on_calls = sum(o["selection_calls"] for o in run_pipeline(on, _qs()))
    assert on_calls < off_calls
    # Fixed arm agrees too (same retriever, same memo store semantics).
    assert _scrub(run_pipeline_fixed(off, _qs())) == \
        _scrub(run_pipeline_fixed(on, _qs()))


def test_arm_caches_are_independent_across_contexts():
    # I-5 guard, found live in A4: two arms sharing one config lets the
    # second arm free-ride the first arm's selection cache (fixed showed 0
    # live calls after gated). Separate configs over one shared index keep
    # per-arm costs independent; the index object itself is shared.
    from jevrag.benchmarks.docbench import build_bm25_index
    pool = copy.deepcopy(POOL)
    index = build_bm25_index(pool)
    shared = build_pipeline_from_chunks(
        _config(memoize_selection=True), "pool", pool, index=index)
    run_pipeline(shared, _qs())
    free_ride = run_pipeline_fixed(shared, _qs()[:1], n_rounds=3)[0]
    # Partial free-ride, documented exactly: gated q1 stopped after round 1
    # (4 cached), so fixed reuses those 4 and pays only rounds 2-3's 2 new
    # chunks (live 2 of 6). On a saturated pool (live HotpotQA: pool_n=20
    # every round) the free-ride is total — 0 live calls.
    assert free_ride["selection_calls"] == 2

    fresh = build_pipeline_from_chunks(
        _config(memoize_selection=True), "pool", pool, index=index)
    assert fresh.index is shared.index  # index built once, shared
    independent = run_pipeline_fixed(fresh, _qs()[:1], n_rounds=3)[0]
    assert independent["selection_calls"] == 6
    assert _scrub({k: v for k, v in free_ride.items()
                   if k != "selection_calls"}) == _scrub(
        {k: v for k, v in independent.items() if k != "selection_calls"})


def test_trace_rows_carry_new_candidates_and_evidence_ids():
    # V03-A5, both paths: round 1 judges 4 (pool_n=4 over 6 docs), round 2
    # sees 6 with 2 new, round 3 repeats all 6; evidence_ids name what the
    # loop actually kept each round.
    for memo, expected_live in ((False, 16), (True, 6)):
        ctx = build_pipeline_from_chunks(
            _config(memoize_selection=memo, suff_conf=0.1), "pool",
            copy.deepcopy(POOL))
        out = run_pipeline(ctx, _qs()[:1])[0]
        rows = out["selection"]["rounds"]
        assert [len(r["new_candidates"]) for r in rows] == [4, 2, 0]
        seen: set[str] = set()
        for r in rows:
            assert isinstance(r["new_candidates"], list)
            assert isinstance(r["evidence_ids"], list)
            assert not (set(r["new_candidates"]) & seen)
            seen.update(r["new_candidates"])
            assert set(r["evidence_ids"]) <= seen
        assert out["selection_calls"] == expected_live


def test_default_path_golden_after_refactor():
    # Pins the _assemble_context extraction (§20): the ingestion path's
    # outputs on the tiny fixture must equal these exact values.
    # outputs on the tiny fixture must equal these exact values.
    ctx = build_pipeline(_config(), "d", copy.deepcopy(PAGES))
    assert [c["text"] for c in ctx.chunks] == [
        "Alpha one.\n\nAlpha two talks about retrievers.",
        "Beta one covers generators.\n\nBeta two ends it."]
    assert ctx.ingestion_trace["splits"] == [1]
    assert ctx.per_round_k == 2  # explicit config wins
    modes = ctx.joint_modes
    assert modes["sufficiency"] == {"mode": "default", "threshold": 0.7}
    assert modes["chunk_boundary"]["threshold"] == 0.5
    out = run_pipeline(ctx, [_qs()[0]])[0]
    assert out["outcome"] == "answer" and out["source"] == "pipeline"
