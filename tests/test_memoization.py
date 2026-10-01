"""Memoization equivalence (packet V03-A1, tracker item I-2).

With a deterministic fake Decision, pipeline outputs (outcomes, answers,
thresholds, selected evidence) are identical with memoization on vs. off;
only call counts differ. Zero network; all stubs.
"""

from __future__ import annotations

import copy
from typing import Any

from jevrag.decision import DecisionResult, StubDecision, TypedQuestion
from jevrag.pipeline import (
    PipelineConfig,
    answer_question,
    build_pipeline,
    ingest_document,
    load_memo_cache_jsonl,
    make_filtered_retrieve_fn,
    run_pipeline,
    save_memo_cache_jsonl,
)

PAGES = [
    {"page": 1, "text": "Alpha one.\n\nAlpha two talks about retrievers."},
    {"page": 2, "text": "Beta one covers generators.\n\nBeta two ends it."},
    {"page": 3, "text": "Gamma one keeps retrievers.\n\nGamma two keeps generators."},
    {"page": 4, "text": "Delta one drops topic.\n\nDelta two ends it."},
]


class SplitOnBeta(StubDecision):
    """Content-aware stub: split only before the 'Beta one' paragraph."""

    def ask(self, state: Any, questions: list[TypedQuestion]) -> DecisionResult:
        after = str(state.get("after", ""))
        conf = 0.9 if after.startswith("Beta one") else 0.1
        return StubDecision(confidences={q.name: conf
                                         for q in questions}).ask(state, questions)


class SelectKeep(StubDecision):
    """Content-aware stub: relevant iff 'keep' appears in the passage."""

    def ask(self, state: Any, questions: list[TypedQuestion]) -> DecisionResult:
        conf = 0.9 if "keep" in str(state.get("passage", "")) else 0.1
        return StubDecision(confidences={q.name: conf
                                         for q in questions}).ask(state, questions)


class Counting(StubDecision):
    """Wraps another stub's verdicts while counting live backend calls."""

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


def _questions():
    return [
        {"id": "q1", "question": "What talks about retrievers?",
         "answer": "Alpha two.", "type": "text-only"},
        {"id": "q2", "question": "What keeps generators running here?",
         "answer": "Gamma two.", "type": "text-only"},
    ]


def _config(memo: bool, selection: StubDecision, **over: Any) -> PipelineConfig:
    return PipelineConfig(
        chunk_decision=SplitOnBeta(),
        selection_decision=selection,
        # 0.1 < 0.7: the gate never fires early -> all max_rounds run,
        # so round 2+ pools overlap round 1's and the cache has work to do.
        sufficiency_decision=scripted(0.1),
        abstain_decision=scripted(0.8),
        cache_decision=scripted(0.1),
        answer_fn=lambda q, ev: "test answer",
        per_round_k=2,
        max_rounds=3,
        memoize_selection=memo,
        memoize_chunk=memo,
        **over,
    )


#: Accounting/timing keys that MUST differ (or may differ) between memo
#: on/off — everything else must be identical.
_SCRUB_KEYS = frozenset({
    "latency_ms", "wall_ms", "input_tokens", "output_tokens", "cost_usd",
    "jev_calls", "jev_input_tokens", "jev_cost_usd", "selection_calls",
    "calls", "cache_hits", "cache_hit", "cache_enabled", "n_cache_hits",
    "n_live_calls", "n_votes", "confidences", "pool_ids", "cache",
})


def _scrub(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _scrub(v) for k, v in node.items() if k not in _SCRUB_KEYS}
    if isinstance(node, list):
        return [_scrub(v) for v in node]
    return node


def test_pipeline_outputs_identical_memo_on_vs_off():
    sel_off: StubDecision = SelectKeep()
    sel_on = SelectKeep()
    ctx_off = build_pipeline(_config(False, sel_off), "d", copy.deepcopy(PAGES))
    ctx_on = build_pipeline(_config(True, sel_on), "d", copy.deepcopy(PAGES))
    out_off = run_pipeline(ctx_off, _questions())
    out_on = run_pipeline(ctx_on, _questions())
    assert len(out_off) == len(out_on) == 2
    for a, b in zip(out_off, out_on):
        assert _scrub(a) == _scrub(b)
    # Same selected evidence stashed for the abstain handoff.
    assert _scrub(ctx_off.evidence_log) == _scrub(ctx_on.evidence_log)
    # Only call counts differ — and memoization strictly saves calls here:
    # every round re-judges the same small chunk pool.
    calls_off = sum(o["selection_calls"] for o in out_off)
    calls_on = sum(o["selection_calls"] for o in out_on)
    assert calls_on < calls_off, (calls_on, calls_off)
    hits = sum(o["selection"]["cache_hits"] for o in out_on)
    assert hits > 0
    assert calls_off - calls_on == hits
    # Ingestion verdicts identical too (cold chunk cache: same live cost).
    assert _scrub(ctx_off.chunks) == _scrub(ctx_on.chunks)
    assert ctx_off.ingestion_trace["splits"] == ctx_on.ingestion_trace["splits"]


def test_selection_backend_calls_drop_by_cache_hits():
    counting_off = Counting(SelectKeep())
    counting_on = Counting(SelectKeep())
    ctx_off = build_pipeline(_config(False, counting_off), "d",
                             copy.deepcopy(PAGES))
    ctx_on = build_pipeline(_config(True, counting_on), "d",
                            copy.deepcopy(PAGES))
    run_pipeline(ctx_off, _questions())
    run_pipeline(ctx_on, _questions())
    assert counting_on.calls < counting_off.calls
    hits = sum(r.get("n_cache_hits", 0) for r in ctx_on.selection_trace)
    assert hits > 0
    assert counting_off.calls - counting_on.calls == hits


def test_selection_trace_rows_show_hits_vs_live_calls():
    ctx = build_pipeline(_config(True, SelectKeep()), "d",
                         copy.deepcopy(PAGES))
    run_pipeline(ctx, _questions())
    rows = ctx.selection_trace
    assert rows, "expected selection trace rows"
    assert all(r.get("cache_enabled") is True for r in rows)
    # Round 1 per question is all-live; every later round is all-hits,
    # because the cumulative top-k pool can only repeat chunks.
    first_round_rows = [r for i, r in enumerate(rows) if i % 3 == 0]
    later_rows = [r for i, r in enumerate(rows) if i % 3 != 0]
    assert all(r["n_cache_hits"] == 0 for r in first_round_rows)
    assert all(r["n_cache_hits"] == r["n_judged"] > 0 for r in later_rows)
    for r in rows:
        assert r["jev_calls"] == r["n_live_calls"] == r["n_judged"] - r["n_cache_hits"]
        # Overlap proof: later rounds judge zero chunks not seen before.
        assert r["n_judged"] == r["pool_n"] or r["n_judged"] < r["pool_n"]


def test_memo_off_trace_shape_has_no_cache_keys():
    ctx = build_pipeline(_config(False, SelectKeep()), "d",
                         copy.deepcopy(PAGES))
    run_pipeline(ctx, _questions())
    for r in ctx.selection_trace:
        assert "cache_enabled" not in r and "n_cache_hits" not in r
        assert "pool_ids" not in r
    assert "cache_enabled" not in ctx.ingestion_trace
    out = answer_question(
        {"id": "qx", "question": "Fresh question about retrievers?",
         "answer": "Alpha.", "type": "text-only"}, ctx)
    assert "cache_hits" not in out["selection"]


def test_chunk_reingest_costs_zero_calls_with_identical_chunks():
    cache: dict[str, dict[str, Any]] = {}
    chunks1, trace1 = ingest_document(
        copy.deepcopy(PAGES), doc_id="d", chunk_decision=SplitOnBeta(),
        repeats=2, chunk_cache=cache, memoize_chunk=True)
    assert trace1["jev_calls"] == trace1["n_candidates"] * 2
    assert trace1["n_cache_hits"] == 0
    chunks2, trace2 = ingest_document(
        copy.deepcopy(PAGES), doc_id="d", chunk_decision=SplitOnBeta(),
        repeats=2, chunk_cache=cache, memoize_chunk=True)
    assert chunks2 == chunks1
    assert trace2["splits"] == trace1["splits"]
    assert trace2["jev_calls"] == 0
    assert trace2["jev_input_tokens"] == 0
    assert trace2["n_cache_hits"] == trace2["n_candidates"] > 0
    # And the memoized chunks match a memo-off ingest exactly.
    chunks0, _ = ingest_document(
        copy.deepcopy(PAGES), doc_id="d", chunk_decision=SplitOnBeta(),
        repeats=2)
    assert chunks0 == chunks1


def test_chunk_cache_jsonl_roundtrip(tmp_path):
    cache: dict[str, dict[str, Any]] = {}
    ingest_document(copy.deepcopy(PAGES), doc_id="d",
                    chunk_decision=SplitOnBeta(), repeats=1,
                    chunk_cache=cache, memoize_chunk=True)
    path = str(tmp_path / "cb_cache.jsonl")
    save_memo_cache_jsonl(cache, path)
    reloaded = load_memo_cache_jsonl(path)
    assert reloaded == cache
    chunks, trace = ingest_document(
        copy.deepcopy(PAGES), doc_id="d", chunk_decision=SplitOnBeta(),
        repeats=1, chunk_cache=reloaded, memoize_chunk=True)
    assert trace["jev_calls"] == 0
    assert chunks  # non-empty, reassembled from cached verdicts
    # Missing file is the cold-cache shape, not an error.
    assert load_memo_cache_jsonl(str(tmp_path / "nope.jsonl")) == {}


def test_filtered_retrieve_fn_pool_overlap_measured_offline():
    # Direct seam-level proof: cumulative top-k pools overlap by
    # construction, so round 2+ re-judges round 1's chunks.
    docs = [{"title": f"c{i}", "text": f"keep chunk number {i}"}
            for i in range(6)]

    def raw(question: str, n: int) -> list[dict[str, Any]]:
        return docs[:n]  # deterministic top-k prefix, like BM25

    log: dict = {}
    trace: list = []
    fn = make_filtered_retrieve_fn(
        raw, selection_decision=SelectKeep(), selection_threshold=0.5,
        candidate_pool_cap=20, oversample=2, evidence_log=log, trace=trace,
        selection_cache={}, memoize_selection=True)
    pools = []
    for rnd in (1, 2, 3):
        fn("q", rnd * 2)  # sufficiency's cumulative counts, per_round_k=2
        pools.append(set(trace[-1].get("pool_ids", [])))
    assert pools[0] and pools[0] <= pools[1] <= pools[2]
    # Round 1 judges 4 (pool_n = 2*2); round 2's pool of 6 contains those 4
    # plus 2 new; round 3 re-judges nothing new — 100% hits.
    assert trace[0]["n_judged"] == 4 and trace[0]["n_cache_hits"] == 0
    assert trace[1]["n_judged"] == 6 and trace[1]["n_cache_hits"] == 4
    assert trace[2]["n_judged"] == 6 and trace[2]["n_cache_hits"] == 6
    assert pools[1] == pools[2] and len(pools[1]) == 6
