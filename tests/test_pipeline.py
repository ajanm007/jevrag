"""Tests for jevrag.pipeline — joint-by-joint proof, then the chain.

Per the prove-then-chain discipline: each handoff is proven on its own with
StubDecision (zero cost, offline) before any live run. No test here touches
the network; live runs live in scripts/run_pipeline_doc.py.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from jevrag.decision import Decision, DecisionResult, StubDecision, TypedQuestion
from jevrag.eval.crc import crc_answered, crc_calibrate
from jevrag.pipeline import (
    JOINT_ABSTAIN,
    JOINT_SUFFICIENCY,
    PipelineConfig,
    _CrcShiftDecision,
    answer_question,
    build_pipeline,
    calibrate_rule,
    check_cache_branch,
    ingest_document,
    make_filtered_retrieve_fn,
    normalize_query,
    run_pipeline,
    split_paragraphs,
)
from jevrag.pipeline import CrcSpec


PAGES = [
    {"page": 1, "text": "Alpha one.\n\nAlpha two talks about retrievers."},
    {"page": 2, "text": "Beta one covers generators.\n\nBeta two ends it."},
]


class SplitOnC(StubDecision):
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


def scripted(conf: float) -> StubDecision:
    return StubDecision(confidences={"sufficient": conf, "grounded": conf,
                                     "relevant": conf, "split": conf})


# --- joint 1: chunk-boundary output becomes the index ---

def test_split_paragraphs_keeps_pages():
    paras = split_paragraphs(PAGES)
    assert [p["text"] for p in paras] == [
        "Alpha one.", "Alpha two talks about retrievers.",
        "Beta one covers generators.", "Beta two ends it."]
    assert [p["page"] for p in paras] == [1, 1, 2, 2]


def test_ingest_reassembles_chunks_from_split_labels():
    chunks, trace = ingest_document(PAGES, doc_id="d", chunk_decision=SplitOnC())
    assert trace["n_paragraphs"] == 4
    assert trace["n_candidates"] == 3
    assert trace["n_splits"] == 1
    assert trace["n_chunks"] == 2
    assert chunks[0]["text"] == "Alpha one.\n\nAlpha two talks about retrievers."
    assert chunks[1]["text"] == "Beta one covers generators.\n\nBeta two ends it."
    assert all(set(c) == {"title", "text", "page"} for c in chunks)
    # Coverage: every paragraph present exactly once, in order.
    flat = "\n\n".join(c["text"] for c in chunks).split("\n\n")
    assert flat == ["Alpha one.", "Alpha two talks about retrievers.",
                    "Beta one covers generators.", "Beta two ends it."]


def test_ingest_force_split_caps_chunk_size():
    chunks, trace = ingest_document(
        PAGES, doc_id="d", chunk_decision=scripted(0.1),  # merge everything
        max_chunk_chars=30)
    assert trace["force_splits"] >= 1
    assert all(len(c["text"]) <= 60 for c in chunks)  # paragraph granularity kept
    flat = "\n\n".join(c["text"] for c in chunks).split("\n\n")
    assert len(flat) == 4  # nothing lost


def test_ingest_single_paragraph_needs_no_candidates():
    chunks, trace = ingest_document(
        [{"page": 1, "text": "Only para."}], doc_id="d",
        chunk_decision=scripted(0.9))
    assert trace["n_candidates"] == 0 and len(chunks) == 1
    assert chunks[0]["text"] == "Only para."


# --- joint 2: context-selection inside the retrieve_fn seam ---

def pool_docs():
    return [
        {"title": f"doc-{i}", "text": text}
        for i, text in enumerate([
            "keep this retriever passage", "drop this off-topic passage",
            "keep this generator passage", "drop this random passage"])]


def test_filtered_retrieve_returns_only_selected():
    log: dict = {}
    trace: list = []
    fn = make_filtered_retrieve_fn(
        lambda q, n: pool_docs()[:n], selection_decision=SelectKeep(),
        selection_threshold=0.5, candidate_pool_cap=4, oversample=2,
        evidence_log=log, trace=trace)
    out = fn("q", 2)
    assert [d["title"] for d in out] == ["doc-0", "doc-2"]
    assert log["q"] == out
    assert trace[0]["n_judged"] == 4 and trace[0]["n_selected"] == 2
    assert trace[0]["fallback_top1"] is False


def test_filtered_retrieve_fails_open_on_empty_selection():
    log: dict = {}
    trace: list = []
    fn = make_filtered_retrieve_fn(
        lambda q, n: pool_docs()[:n], selection_decision=scripted(0.0),
        selection_threshold=0.5, candidate_pool_cap=4, oversample=2,
        evidence_log=log, trace=trace)
    out = fn("q", 2)
    assert [d["title"] for d in out] == ["doc-0"]  # top-1 raw, never empty
    assert trace[0]["fallback_top1"] is True


def test_filtered_retrieve_matches_run_sufficiency_contract():
    # run_sufficiency calls retrieve_fn(question, round_no * per_round_k):
    # positional (str, int), list of {title, text} out. Prove the seam fits
    # by driving the real loop through it.
    from jevrag.primitives.sufficiency import run_sufficiency

    log: dict = {}
    trace: list = []
    fn = make_filtered_retrieve_fn(
        lambda q, n: pool_docs()[:min(n, 4)], selection_decision=SelectKeep(),
        candidate_pool_cap=4, oversample=2,
        evidence_log=log, trace=trace)
    rec = run_sufficiency(
        question_id="q", question="q", gold="g", qtype="t",
        decision=scripted(0.95), retrieve_fn=fn,
        answer_fn=lambda q, ev: "ans", threshold=0.7, max_rounds=2,
        per_round_k=2)
    assert rec["rounds_used"] == 1 and rec["prediction"] == "ans"
    assert "q" in log  # the abstain handoff's evidence is stashed


# --- joint 3+4: sufficiency record -> abstain verdict -> outcome ---

def _stub_ctx(**over):
    cache_decision = over.pop(
        "cache_decision",
        StubDecision(confidences={"serve_from_cache": 0.1,
                                  "staleness_risk": 0.1}))
    cfg = PipelineConfig(
        chunk_decision=SplitOnC(),
        selection_decision=SelectKeep(),
        sufficiency_decision=scripted(0.95),
        abstain_decision=scripted(over.pop("abstain_conf", 0.8)),
        cache_decision=cache_decision,
        answer_fn=lambda q, ev: "test answer",
        per_round_k=2,
        **over,
    )
    return build_pipeline(cfg, "d", PAGES)


def _question(qid="q1", text="What talks about retrievers?"):
    return {"id": qid, "question": text, "answer": "Alpha two.",
            "type": "text-only"}


def test_answer_question_pass_produces_answer():
    out = answer_question(_question(), _stub_ctx())
    assert out["outcome"] == "answer" and out["source"] == "pipeline"
    assert out["answer"] == "test answer"
    assert out["sufficiency"]["record"]["rounds_used"] >= 1
    assert out["abstain"]["record"]["action"] == "pass"
    assert out["selection_calls"] >= 1
    assert out["thresholds"]["sufficiency"]["mode"] == "default"


def test_answer_question_abstain_produces_no_answer_with_prediction_kept():
    out = answer_question(_question(), _stub_ctx(abstain_conf=0.1))
    assert out["outcome"] == "no_answer"
    assert out["answer"] == ""
    assert out["prediction"] == "test answer"  # kept for scoring, not served


# --- joint 5: cache short-circuit ---

def test_cache_hit_serves_without_pipeline():
    served, trace = check_cache_branch(
        query="Who wrote it?", query_id="q",
        cache_entries={normalize_query("Who wrote it?"): {"answer": "Ann"}},
        cache_decision=StubDecision(confidences={
            "serve_from_cache": 0.9, "staleness_risk": 0.1}))
    assert trace["hit"] is True
    assert served is not None and served["outcome"] == "answer"
    assert served["source"] == "cache" and served["answer"] == "Ann"


def test_cache_miss_and_regenerate_enter_pipeline():
    served, trace = check_cache_branch(
        query="Unseen question?", query_id="q", cache_entries={},
        cache_decision=scripted(0.9))
    assert served is None and trace["hit"] is False
    served, trace = check_cache_branch(
        query="Stale entry?", query_id="q",
        cache_entries={normalize_query("Stale entry?"): {"answer": "Old"}},
        cache_decision=StubDecision(confidences={
            "serve_from_cache": 0.1, "staleness_risk": 0.9}))
    assert served is None and trace["action"] == "regenerate"


def test_full_chain_cache_hit_short_circuits():
    ctx = _stub_ctx(
        cache_entries={normalize_query("What talks about retrievers?"):
                       {"answer": "Cached!"}},
        cache_decision=StubDecision(confidences={"serve_from_cache": 0.9,
                                                 "staleness_risk": 0.1}))
    out = answer_question(_question(), ctx)
    assert out["source"] == "cache" and out["answer"] == "Cached!"
    assert out["sufficiency"] is None  # the loop never ran


# --- CRC adapter: the decision-space rule, exactly ---

def _calibration():
    scores = [0.95, 0.9, 0.7, 0.4, 0.2]
    correct = [1.0, 1.0, 0.0, 0.0, 0.0]
    keys = [f"q{i}" for i in range(5)]
    return scores, correct, keys


def test_crc_adapter_matches_crc_answered_row_for_row():
    scores, correct, keys = _calibration()
    cal = crc_calibrate(scores, correct, 0.4, keys=keys)
    thr = float(cal["threshold"])
    for raw, key in zip(scores, keys):
        inner = StubDecision(confidences={"sufficient": raw})
        adapted = _CrcShiftDecision(inner, thr, lambda s, qs: s["question"])
        res = adapted.ask({"question": key}, [
            __import__("jevrag.decision", fromlist=["TypedQuestion"])
            .TypedQuestion(name="sufficient", kind="boolean",
                           instructions="x")])
        expected = bool(crc_answered([raw], [key], thr)[0])
        assert (float(res.confidence) >= thr) == expected
        assert res.metadata["crc_raw_confidence"] == pytest.approx(raw)


def test_crc_adapter_refuses_multi_question():
    inner = StubDecision(confidences={"a": 0.9, "b": 0.1})
    adapted = _CrcShiftDecision(inner, 0.5, lambda s, qs: "k")
    from jevrag.decision import TypedQuestion as TQ
    with pytest.raises(ValueError, match="single-question"):
        adapted.ask({}, [TQ(name="a", kind="boolean", instructions="x"),
                         TQ(name="b", kind="boolean", instructions="x")])


def test_calibrate_rule_rejects_non_crc_joint():
    scores, correct, keys = _calibration()
    with pytest.raises(ValueError, match="CRC opt-in is supported"):
        calibrate_rule("context_selection",
                       CrcSpec(alpha=0.3, scores=scores, correct=correct,
                               keys=keys))


def test_crc_abstain_all_sufficiency_runs_all_rounds_raw():
    scores = [0.96, 0.9]
    correct = [0.0, 0.0]  # even the top row is wrong: no threshold exists
    keys = ["qa", "qb"]
    rule = calibrate_rule(
        JOINT_SUFFICIENCY,
        CrcSpec(alpha=0.3, scores=scores, correct=correct, keys=keys))
    assert rule.abstain_all
    cfg = PipelineConfig(
        chunk_decision=SplitOnC(), selection_decision=SelectKeep(),
        sufficiency_decision=scripted(0.99), abstain_decision=scripted(0.99),
        cache_decision=scripted(0.1), answer_fn=lambda q, ev: "ans",
        per_round_k=2, max_rounds=2,
        crc_specs={JOINT_SUFFICIENCY: CrcSpec(
            alpha=0.3, scores=scores, correct=correct, keys=keys)})
    ctx = build_pipeline(cfg, "d", PAGES)
    assert ctx.joint_modes[JOINT_SUFFICIENCY]["mode"] == "crc-abstain-all"
    out = answer_question(_question(), ctx)
    assert out["sufficiency"]["record"]["rounds_used"] == 2  # full run, still answers
    assert out["outcome"] == "answer"


def test_run_pipeline_rejects_duplicate_question_texts():
    ctx = _stub_ctx()
    q = _question()
    with pytest.raises(ValueError, match="must be unique"):
        run_pipeline(ctx, [q, dict(q, id="q2")])


def test_ingestion_repeats_pass_through_to_chunk_decisions():
    from jevrag.decision import DecisionResult, TypedQuestion

    calls = []

    class Counting(StubDecision):
        def ask(self, state, questions):
            calls.append(state.get("boundary_index"))
            return super().ask(state, questions)

    cfg = PipelineConfig(
        chunk_decision=Counting(confidences={"split": 0.9}),
        selection_decision=SelectKeep(),
        sufficiency_decision=scripted(0.95),
        abstain_decision=scripted(0.8),
        cache_decision=scripted(0.1),
        answer_fn=lambda q, ev: "test answer",
        per_round_k=2,
        chunk_repeats=3,
    )
    ctx = build_pipeline(cfg, "d", PAGES)
    assert ctx.ingestion_trace["repeats"] == 3
    # 3 paragraph breaks x 3 votes each.
    assert len(calls) == 9
    assert sorted(calls) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


# --- acceptance: controller calls primitives, rewrites none ---

def test_acceptance_pipeline_import_surface():
    path = Path(__file__).resolve().parent.parent / "jevrag" / "pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
    jevrag_imports = {m for m in imports if m.startswith("jevrag.")}
    # Measurement stays out of the controller: no calibration/cost/vendor,
    # no HotpotQA wiring, no scripts. Benchmarks.docbench (BM25 + PDF
    # extraction helpers) and eval.crc (the mandated threshold layer) are
    # the only non-primitive, non-decision modules it may touch.
    banned = {m for m in jevrag_imports
              if m.startswith(("jevrag.eval.calibration", "jevrag.eval.cost",
                                "jevrag._vendor", "jevrag.benchmarks.hotpotqa",
                                "jevrag.adapters", "jevrag.backends"))
              or m == "jevrag.eval" or m.startswith("scripts")}
    assert not banned, f"controller reaches into measurement/vendor: {banned}"
    assert "jevrag.primitives.sufficiency" in jevrag_imports
    assert isinstance(build_pipeline, object) and isinstance(answer_question, object)
