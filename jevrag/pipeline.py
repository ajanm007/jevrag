"""jevrag.pipeline — the real end-to-end RAG controller (v0.2.0 item 1).

The locked architecture, as actual callable code::

    DOCUMENT INGESTION -> Chunk-Boundary -> INDEX -> QUERY -> RETRIEVAL ->
    Context-Selection -> Sufficiency (loop: retrieve again on NO) -> GENERATE
    -> Answer-Abstain -> ANSWER / NO ANSWER

    (separate, parallel path:)
    QUERY -> CACHE LOOKUP -> Cache-Trust -> SAFE: cached answer /
                                               UNSAFE: enter the pipeline above

What already existed vs. what this module adds (both blocking decisions were
closed before this packet — verified in code, not re-derived):
- Max-rounds exit: ``run_sufficiency`` already generates anyway after the
  final round ("Always answers after the final round regardless of
  confidence"). The gap was that nothing wired that answer anywhere — joint
  4 below closes it, no primitive change.
- Per-decision thresholds: each joint takes an optional ``CrcSpec``. If
  given, its threshold comes from the real CRC layer (``crc_calibrate`` with
  real keys); if omitted, the joint uses its primitive's own default
  threshold, unchanged. No joint requires CRC.

The four previously-nonexistent handoffs, each its own function:
1. ``ingest_document`` — chunk-boundary runs once at ingestion; its
   split/merge labels reassemble paragraphs into the chunks a BM25 index is
   built over (via ``decide_document``, unmodified).
2. ``make_filtered_retrieve_fn`` — context-selection filters every
   sufficiency round's candidates *before* the loop asks "enough?". Plugs
   into ``run_sufficiency``'s existing ``retrieve_fn(question, n_docs)``
   seam, which fits exactly (verified against the call site: cumulative
   counts, ``{title, text}`` dicts out). No memoization across rounds — the
   filter literally re-runs each round, and the extra Jev calls are the
   reported cost finding, not an estimate.
3. Sufficiency's output flows into ``check_answer`` as the real terminal
   step; the abstain verdict — not sufficiency's confidence — produces
   ANSWER or NO ANSWER. Bridge detail: the handoff record carries no
   evidence, so the controller's retrieve wrapper stashes each round's
   evidence itself (it already sees every set). Proposed, not made: carry
   evidence ids in the handoff record.
4. ``check_cache_branch`` — cache-trust stays the parallel short-circuit. A
   toy in-memory dict keyed by normalized query text (explicitly not real
   cache infrastructure); hit → real ``check_cache`` call → serve or fall
   through to the pipeline.

CRC-in-the-chain interface finding (reported, not routed around): the
primitives compare ``confidence >= threshold`` internally, while a CRC
threshold lives in *decision space* (``score + eps*offset(key)``). These
compose through :class:`_CrcShiftDecision`, a thin Decision wrapper that
shifts the returned confidence by the row's deterministic offset so the
primitive's plain comparison implements the CRC rule exactly. Raw confidence
is preserved in metadata; shifts are <= 1e-6 (below reporting precision).
No primitive signature changes. Genuine abstain-all calibrations
(threshold +inf) are handled without the adapter: sufficiency maps to
``threshold=None`` (run all rounds — the loop's own gate-off shape, still
answering), abstain maps to an always-abstain threshold. Both are traced
loudly as ``crc-abstain-all``, never silently.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from jevrag.benchmarks.docbench import build_bm25_index
from jevrag.decision import Decision, DecisionResult, TypedQuestion, estimate_cost_usd
from jevrag.eval.crc import TIE_BREAK_EPS, crc_calibrate, crc_offsets
from jevrag.primitives.answer_abstain import (
    DEFAULT_ABSTAIN_THRESHOLD,
    action_for as abstain_action_for,
    check_answer,
)
from jevrag.primitives.cache_trust import (
    DEFAULT_SERVE_THRESHOLD,
    DEFAULT_STALENESS_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TTL_SECONDS,
    CacheTrustState,
    check_cache,
)
from jevrag.primitives.chunk_boundary import (
    DEFAULT_SPLIT_THRESHOLD,
    decide_document,
)
from jevrag.primitives.context_selection import (
    DEFAULT_INCLUDE_THRESHOLD,
    decide_query,
)
from jevrag.primitives.sufficiency import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_THRESHOLD,
    run_sufficiency_with_trace,
)

#: Joint names, used as CrcSpec keys and in traces.
JOINT_CHUNK = "chunk_boundary"
JOINT_SELECTION = "context_selection"
JOINT_SUFFICIENCY = "sufficiency"
JOINT_ABSTAIN = "answer_abstain"
JOINT_CACHE = "cache_trust"

#: CRC opt-in is supported at these joints (terminal loop + terminal verdict —
#: where an error budget means something). Selection/chunk/cache stay on
#: their default thresholds; selection-as-CRC is a noted alternative, not built.
CRC_JOINTS = (JOINT_SUFFICIENCY, JOINT_ABSTAIN)

#: Candidate pool ceiling per retrieval call (budget-derived: the filter asks
#: one Jev call per candidate, so the pool is the per-round cost dial).
DEFAULT_CANDIDATE_POOL_CAP = 20

#: Raw over-retrieval multiplier: pool = min(round_k * oversample, cap).
DEFAULT_OVERSAMPLE = 2

#: Force-split merged chunks past this size (controller policy: an unbounded
#: merged chunk would blow the downstream Jev state budget).
DEFAULT_MAX_CHUNK_CHARS = 4000


@dataclass
class CrcSpec:
    """Error-budget calibration data for one joint: alpha + a labeled prior run."""

    alpha: float
    scores: list[float]
    correct: list[float]
    keys: list[str]


@dataclass
class CrcRule:
    """A calibrated joint: decision-space threshold + its provenance."""

    joint: str
    alpha: float
    threshold: float  # decision space; +inf iff abstain_all
    threshold_score: float
    k_hat: int | None
    n_cal: int
    abstain_all: bool


@dataclass
class PipelineConfig:
    """Everything the controller needs, decided up front."""

    chunk_decision: Decision
    selection_decision: Decision
    sufficiency_decision: Decision
    abstain_decision: Decision
    cache_decision: Decision
    answer_fn: Callable[..., Any]
    chunk_threshold: float = DEFAULT_SPLIT_THRESHOLD
    selection_threshold: float = DEFAULT_INCLUDE_THRESHOLD
    sufficiency_threshold: float | None = DEFAULT_THRESHOLD
    abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD
    crc_specs: dict[str, CrcSpec] = field(default_factory=dict)
    cache_entries: dict[str, dict[str, str]] = field(default_factory=dict)
    max_rounds: int = DEFAULT_MAX_ROUNDS
    per_round_k: int | None = None  # None -> default_k_for_doc(n_chunks)
    candidate_pool_cap: int = DEFAULT_CANDIDATE_POOL_CAP
    oversample: int = DEFAULT_OVERSAMPLE
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    #: Ingestion votes per boundary (packet 29): the backend's own output
    #: jitters ~±0.03 call-to-call near the threshold, so the chain judges
    #: each candidate 3x on the mean by default. Once per document, so the
    #: 3x call cost is negligible and traced.
    chunk_repeats: int = 3


@dataclass
class PipelineContext:
    """Built once by :func:`build_pipeline`: index, rules, live traces."""

    config: PipelineConfig
    doc_id: str
    chunks: list[dict[str, Any]]
    index: Any
    per_round_k: int
    crc_rules: dict[str, CrcRule] = field(default_factory=dict)
    joint_modes: dict[str, dict[str, Any]] = field(default_factory=dict)
    sufficiency_decision: Decision | None = None  # CRC-wrapped or raw
    abstain_decision: Decision | None = None
    ingestion_trace: dict[str, Any] = field(default_factory=dict)
    evidence_log: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    selection_trace: list[dict[str, Any]] = field(default_factory=list)


def calibrate_rule(joint: str, spec: CrcSpec) -> CrcRule:
    """Run the real CRC calibration for one joint (keys required, always)."""
    if joint not in CRC_JOINTS:
        raise ValueError(
            f"CRC opt-in is supported at {list(CRC_JOINTS)}; "
            f"joint {joint!r} stays on its default threshold."
        )
    cal = crc_calibrate(spec.scores, spec.correct, spec.alpha, keys=spec.keys)
    return CrcRule(
        joint=joint,
        alpha=float(spec.alpha),
        threshold=float(cal["threshold"]),
        threshold_score=float(cal["threshold_score"]),
        k_hat=cal["k_hat"],
        n_cal=int(cal["n_cal"]),
        abstain_all=bool(cal["abstain_all"]),
    )


class _CrcShiftDecision:
    """Decision wrapper mapping raw confidence into CRC decision space.

    Returns the inner result with confidence (and, for single-question
    boolean calls, result) shifted by ``TIE_BREAK_EPS * offset(key)`` so a
    primitive's internal ``confidence >= threshold`` check implements
    ``crc_answered`` exactly when ``threshold`` is the calibrated
    decision-space value. Raw confidence is kept in the returned
    ``DecisionResult``'s ``metadata["crc_raw_confidence"]`` — with one
    precise boundary: primitives build their output records from
    ``result.confidence``, so downstream records carry the decision-space
    value, which differs from raw by <= 1e-6 (below every reporting
    precision in this project; the harness cannot see it). Single-question
    calls only — a dict confidence has no one threshold to be shifted
    against, refused loudly. Finite thresholds only; abstain-all is handled
    by the caller without this wrapper (see module docstring).
    """

    def __init__(
        self,
        inner: Decision,
        threshold: float,
        key_fn: Callable[[Any, list[TypedQuestion]], str],
    ) -> None:
        if not callable(getattr(inner, "ask", None)):
            raise TypeError(
                "CRC adapter wraps a Decision backend; "
                f"got {type(inner).__name__}."
            )
        self._inner = inner
        self.threshold = float(threshold)
        self._key_fn = key_fn

    def _key(self, state: Any, questions: list[TypedQuestion]) -> str:
        key = str(self._key_fn(state, questions))
        if not key:
            raise ValueError("CRC adapter key_fn returned an empty key.")
        return key

    def ask(
        self, state: Any, questions: list[TypedQuestion]
    ) -> DecisionResult:
        result = self._inner.ask(state, questions)
        if isinstance(result.confidence, dict):
            raise ValueError(
                "CRC adapter supports single-question calls only; "
                "this joint needs default thresholds."
            )
        raw = float(result.confidence)
        offset = float(crc_offsets([self._key(state, questions)])[0])
        shifted = raw + TIE_BREAK_EPS * offset
        md = dict(result.metadata) if isinstance(result.metadata, dict) else {}
        md["crc_raw_confidence"] = raw
        md["crc_threshold"] = self.threshold
        return DecisionResult(result=shifted, confidence=shifted, metadata=md)


def _question_key(state: Any, questions: list[TypedQuestion]) -> str:
    """CRC row key: the question text. Stable across rounds and joints;
    unique per question within a run (asserted at build)."""
    _ = questions
    if not isinstance(state, dict) or "question" not in state:
        raise ValueError(
            "CRC adapter expects primitive state dicts carrying 'question'."
        )
    return str(state["question"])


def normalize_query(query: str) -> str:
    """Toy-cache key: lowercase alphanumeric tokens, joined. Same tokenization
    the BM25 path uses (cited, not reinvented) — adequate for an explicit
    stand-in, not a production key."""
    return " ".join(re.findall(r"[a-z0-9]+", query.lower()))


def split_paragraphs(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Page texts -> ordered paragraphs with page numbers. Blank-line split;
    empties dropped. Controller policy: chunk-boundary judges paragraph
    breaks, so paragraphs are the base unit the index is reassembled from."""
    paras = []
    for p in pages:
        for block in str(p.get("text", "")).split("\n\n"):
            text = block.strip()
            if text:
                paras.append({"text": text, "page": int(p.get("page", 0))})
    if not paras:
        raise ValueError("no paragraphs extracted — nothing to ingest")
    return paras


def ingest_document(
    pages: list[dict[str, Any]],
    *,
    doc_id: str,
    chunk_decision: Decision,
    chunk_threshold: float = DEFAULT_SPLIT_THRESHOLD,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    max_chars: int = 2000,
    repeats: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Joint 1: paragraphs -> chunk-boundary decisions -> indexable chunks.

    Every paragraph break becomes one candidate (before/after = adjacent
    paragraphs); ``decide_document`` judges each once; "split" closes the
    running chunk. Merged text past ``max_chunk_chars`` force-splits (traced).
    Returns (chunks, trace); chunks are HotpotQA-shaped {title, text, page}.
    """
    paras = split_paragraphs(pages)
    candidates = [
        {
            "doc_id": doc_id,
            "boundary_index": i,
            "before": paras[i]["text"],
            "after": paras[i + 1]["text"],
        }
        for i in range(len(paras) - 1)
    ]
    wall_start = time.perf_counter()
    if candidates:
        records, cb_trace = decide_document(
            candidates, chunk_decision, threshold=chunk_threshold,
            max_chars=max_chars, repeats=repeats)
    else:
        records, cb_trace = [], []
    by_idx = {r["boundary_index"]: r for r in records}
    assert len(by_idx) == len(candidates), "boundary records != candidates"

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [paras[0]] if paras else []
    force_splits = 0
    chunk_meta: list[dict[str, Any]] = []

    def close_chunk() -> None:
        text = "\n\n".join(p["text"] for p in current)
        pages_span = [p["page"] for p in current]
        n = len(chunks)
        chunks.append({
            "title": f"{doc_id}-c{n:02d}",
            "text": text,
            "page": pages_span[0],
        })
        chunk_meta.append({"title": f"{doc_id}-c{n:02d}",
                           "n_paras": len(current),
                           "page_end": pages_span[-1]})

    for i in range(len(paras) - 1):
        rec = by_idx[i]
        if rec["result"] == "split":
            close_chunk()
            current = [paras[i + 1]]
        else:
            tentative = "\n\n".join([*(p["text"] for p in current),
                                     paras[i + 1]["text"]])
            if len(tentative) > max_chunk_chars and len(current) >= 1:
                close_chunk()
                current = [paras[i + 1]]
                force_splits += 1
            else:
                current.append(paras[i + 1])
    if current:
        # A single paragraph already over budget still becomes one chunk:
        # force-splitting mid-paragraph would invent boundaries the gate
        # never judged. Traced, not silently capped.
        if len("\n\n".join(p["text"] for p in current)) > max_chunk_chars:
            force_splits += 1
        close_chunk()

    # Coverage proof: every paragraph lands in exactly one chunk, in order
    # (chunks are concatenations of whole paragraphs joined by "\n\n").
    flat_paras: list[str] = []
    for c in chunks:
        flat_paras.extend(c["text"].split("\n\n"))
    assert [p.strip() for p in flat_paras] == [p["text"] for p in paras], \
        "ingestion lost or reordered paragraph text"
    trace = {
        "doc_id": doc_id,
        "n_paragraphs": len(paras),
        "n_candidates": len(candidates),
        "repeats": repeats,
        "n_splits": sum(1 for r in records if r["result"] == "split"),
        # Boundary-level identity: which candidates split, not just how
        # many (packet 29 — count equality hid real flips).
        "splits": sorted(int(r["boundary_index"]) for r in records
                         if r["result"] == "split"),
        "n_chunks": len(chunks),
        "chunk_meta": chunk_meta,
        "force_splits": force_splits,
        # Jev calls, not candidates: repeats votes per candidate (packet 29).
        "jev_calls": sum(int(t.get("n_votes", 1)) for t in cb_trace),
        "jev_input_tokens": sum(int(r["input_tokens"]) for r in records),
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
    }
    return chunks, trace


def make_filtered_retrieve_fn(
    raw_retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    *,
    selection_decision: Decision,
    selection_threshold: float = DEFAULT_INCLUDE_THRESHOLD,
    candidate_pool_cap: int = DEFAULT_CANDIDATE_POOL_CAP,
    oversample: int = DEFAULT_OVERSAMPLE,
    evidence_log: dict[str, list[dict[str, Any]]],
    trace: list[dict[str, Any]],
) -> Callable[[str, int], list[dict[str, Any]]]:
    """Joint 2: context-selection filtering inside the retrieve_fn seam.

    ``retrieve_fn(question, n_docs)`` over-retrieves
    (``min(n_docs * oversample, cap)`` raw), judges every candidate with one
    ``decide_query`` call, and returns selected passages capped at ``n_docs``
    (shape-preserving: the loop's cumulative top-n contract holds). Empty
    selection fails open to top-1 raw — an empty evidence set would
    guarantee a wasted gate call and a vacuous generation — traced as
    ``fallback_top1``. Every round's evidence is stashed in
    ``evidence_log[question]`` for the abstain handoff (joint 4's bridge).
    """

    def retrieve_fn(question: str, n_docs: int) -> list[dict[str, Any]]:
        pool_n = min(max(int(n_docs) * int(oversample), 1),
                     int(candidate_pool_cap))
        pool = list(raw_retrieve_fn(question, pool_n))
        cands = [{
            "query": question,
            "query_id": "",
            "passage": d.get("text", ""),
            "passage_id": str(d.get("title", f"pool-{i}")),
            "passage_title": str(d.get("title", "")),
        } for i, d in enumerate(pool)]
        sel_records, _ = decide_query(
            cands, selection_decision, threshold=selection_threshold)
        selected_ids = {r["passage_id"] for r in sel_records if r["selected"]}
        evidence = [d for d in pool
                    if str(d.get("title", "")) in selected_ids][:int(n_docs)]
        fallback = False
        if not evidence and pool:
            evidence = pool[:1]
            fallback = True
        evidence_log[question] = list(evidence)
        trace.append({
            "n_requested": int(n_docs),
            "pool_n": pool_n,
            "n_judged": len(cands),
            "n_selected": len(selected_ids),
            "n_returned": len(evidence),
            "fallback_top1": fallback,
            "jev_calls": len(cands),
            "jev_input_tokens": sum(int(r["input_tokens"]) for r in sel_records),
        })
        return evidence

    return retrieve_fn


def check_cache_branch(
    *,
    query: str,
    query_id: str,
    cache_entries: dict[str, dict[str, str]],
    cache_decision: Decision,
    serve_threshold: float = DEFAULT_SERVE_THRESHOLD,
    staleness_threshold: float = DEFAULT_STALENESS_THRESHOLD,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Joint 5 (parallel path): toy lookup -> real cache-trust call.

    Returns (outcome, trace). Outcome is set only on SERVE (short-circuit);
    miss or regenerate verdict returns None (enter the main pipeline).
    The toy assigns similarity 1.0 on exact normalized match, age 60s,
    conversation-scoped — disclosed stand-in values, not measurements.
    """
    key = normalize_query(query)
    trace: dict[str, Any] = {"cache_key": key, "hit": key in cache_entries,
                             "toy": True}
    if key not in cache_entries:
        return None, trace
    entry = cache_entries[key]
    state = CacheTrustState.make(
        similarity_score=1.0,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        cache_age_seconds=60.0,
        cache_ttl_seconds=DEFAULT_TTL_SECONDS,
        entry_scoped_to_conversation=True,
        query_length_chars=len(query),
        answer_length_chars=len(entry.get("answer", "")),
    )
    record, ctrace = check_cache(
        query_id=query_id, query=query,
        cached_answer=entry.get("answer", ""), state=state,
        decision=cache_decision, serve_threshold=serve_threshold,
        staleness_threshold=staleness_threshold)
    _ = ctrace  # diagnostic only; the record carries what the trace needs
    trace.update({
        "action": record["action"],
        "confidence": record["confidence"],
        "staleness_risk": record["staleness_risk"],
        "fallback_used": record["fallback_used"],
        "latency_ms": record["latency_ms"],
        "input_tokens": record["input_tokens"],
    })
    if record["action"] == "serve":
        return {
            "outcome": "answer",
            "answer": entry.get("answer", ""),
            "prediction": entry.get("answer", ""),
            "source": "cache",
            "cache": trace,
        }, trace
    return None, trace


def build_pipeline(config: PipelineConfig, doc_id: str,
                   pages: list[dict[str, Any]]) -> PipelineContext:
    """Build once: ingest -> index -> CRC rules -> wrapped joints.

    Raises loudly on duplicate question-text keys later (CRC offsets need
    unique rows) — checked per run in :func:`run_pipeline`, not here, since
    questions arrive there.
    """
    for joint, spec in config.crc_specs.items():
        if joint not in CRC_JOINTS:  # fail fast via calibrate_rule's guard
            calibrate_rule(joint, spec)
    crc_rules = {j: calibrate_rule(j, s)
                 for j, s in config.crc_specs.items()}

    chunks, ingestion_trace = ingest_document(
        pages, doc_id=doc_id, chunk_decision=config.chunk_decision,
        chunk_threshold=config.chunk_threshold,
        max_chunk_chars=config.max_chunk_chars,
        repeats=config.chunk_repeats)
    index = build_bm25_index(chunks)

    from jevrag.benchmarks.docbench import default_k_for_doc
    per_round_k = config.per_round_k or default_k_for_doc(len(chunks))

    joint_modes: dict[str, dict[str, Any]] = {
        JOINT_CHUNK: {"mode": "default",
                      "threshold": config.chunk_threshold},
        JOINT_SELECTION: {"mode": "default",
                          "threshold": config.selection_threshold},
        JOINT_CACHE: {"mode": "default",
                      "serve_threshold": DEFAULT_SERVE_THRESHOLD,
                      "staleness_threshold": DEFAULT_STALENESS_THRESHOLD},
    }
    suff_decision: Decision = config.sufficiency_decision
    if JOINT_SUFFICIENCY in crc_rules:
        rule = crc_rules[JOINT_SUFFICIENCY]
        if rule.abstain_all:
            # No stop-early threshold exists: run all rounds (the loop's own
            # gate-off shape with the backend still asking), keep raw
            # confidences in records. Traced, not silent.
            joint_modes[JOINT_SUFFICIENCY] = {
                "mode": "crc-abstain-all", "alpha": rule.alpha,
                "threshold": None, "k_hat": None, "n_cal": rule.n_cal,
            }
            suff_threshold = None
        else:
            suff_decision = _CrcShiftDecision(
                config.sufficiency_decision, rule.threshold, _question_key)
            joint_modes[JOINT_SUFFICIENCY] = {
                "mode": "crc", "alpha": rule.alpha,
                "threshold": rule.threshold,
                "threshold_score": rule.threshold_score,
                "k_hat": rule.k_hat, "n_cal": rule.n_cal,
            }
            suff_threshold = rule.threshold
    else:
        joint_modes[JOINT_SUFFICIENCY] = {
            "mode": "default", "threshold": config.sufficiency_threshold}
        suff_threshold = config.sufficiency_threshold

    abst_decision: Decision = config.abstain_decision
    if JOINT_ABSTAIN in crc_rules:
        rule = crc_rules[JOINT_ABSTAIN]
        if rule.abstain_all:
            joint_modes[JOINT_ABSTAIN] = {
                "mode": "crc-abstain-all", "alpha": rule.alpha,
                "threshold": None, "k_hat": None, "n_cal": rule.n_cal,
            }
            abst_threshold: float = float("inf")  # nothing clears it
        else:
            abst_decision = _CrcShiftDecision(
                config.abstain_decision, rule.threshold, _question_key)
            joint_modes[JOINT_ABSTAIN] = {
                "mode": "crc", "alpha": rule.alpha,
                "threshold": rule.threshold,
                "threshold_score": rule.threshold_score,
                "k_hat": rule.k_hat, "n_cal": rule.n_cal,
            }
            abst_threshold = rule.threshold
    else:
        joint_modes[JOINT_ABSTAIN] = {
            "mode": "default", "threshold": config.abstain_threshold}
        abst_threshold = config.abstain_threshold

    ctx = PipelineContext(
        config=config, doc_id=doc_id, chunks=chunks, index=index,
        per_round_k=per_round_k, crc_rules=crc_rules,
        joint_modes=joint_modes, sufficiency_decision=suff_decision,
        abstain_decision=abst_decision, ingestion_trace=ingestion_trace)
    ctx_cache: dict[str, Any] = {"suff_threshold": suff_threshold,
                                 "abst_threshold": abst_threshold}
    ctx.joint_modes["_resolved_thresholds"] = ctx_cache
    return ctx


def answer_question(question: dict[str, Any],
                    ctx: PipelineContext) -> dict[str, Any]:
    """One query through the full chain: cache? -> loop -> generate ->
    abstain -> ANSWER / NO ANSWER. Returns a JSON-serializable outcome with
    every joint's trace kept."""
    qid = str(question["id"])
    qtext = str(question["question"])
    gold = str(question.get("answer", ""))
    qtype = str(question.get("type", "unknown"))
    cfg = ctx.config
    wall_start = time.perf_counter()

    served, cache_trace = check_cache_branch(
        query=qtext, query_id=qid, cache_entries=cfg.cache_entries,
        cache_decision=cfg.cache_decision)
    if served is not None:
        served.update({
            "question_id": qid, "question": qtext, "gold": gold,
            "sufficiency": None, "abstain": None,
            "selection_calls": 0,
            "thresholds": {k: v for k, v in ctx.joint_modes.items()
                           if not k.startswith("_")},
            "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
        })
        return served

    from jevrag.benchmarks.docbench import bm25_retrieve

    def raw_retrieve(q: str, n: int) -> list[dict[str, Any]]:
        return bm25_retrieve(q, ctx.index, ctx.chunks, top_k=n)["chunks"]

    sel_trace_start = len(ctx.selection_trace)
    retrieve_fn = make_filtered_retrieve_fn(
        raw_retrieve, selection_decision=cfg.selection_decision,
        selection_threshold=cfg.selection_threshold,
        candidate_pool_cap=cfg.candidate_pool_cap,
        oversample=cfg.oversample, evidence_log=ctx.evidence_log,
        trace=ctx.selection_trace)
    thresholds = ctx.joint_modes["_resolved_thresholds"]
    record, loop_trace = run_sufficiency_with_trace(
        question_id=qid, question=qtext, gold=gold, qtype=qtype,
        decision=ctx.sufficiency_decision, retrieve_fn=retrieve_fn,
        answer_fn=cfg.answer_fn, threshold=thresholds["suff_threshold"],
        max_rounds=cfg.max_rounds, per_round_k=ctx.per_round_k)
    if qtext not in ctx.evidence_log:
        raise RuntimeError(
            f"evidence handoff broken for {qid!r}: the retrieve wrapper "
            "never saw this question — the abstain joint has no evidence.")
    evidence = ctx.evidence_log[qtext]

    from jevrag.primitives.answer_abstain import check_answer as _check
    arec, atrace = _check(
        question_id=qid, question=qtext, gold=gold, evidence=evidence,
        prediction=record["prediction"], decision=ctx.abstain_decision,
        threshold=thresholds["abst_threshold"])
    action = abstain_action_for(float(arec["confidence"]),
                                thresholds["abst_threshold"])
    # NOTE: check_answer already applies the same rule internally; recomputed
    # here only to name the outcome — the two cannot disagree by construction
    # (same confidence, same threshold), asserted below.
    assert action == arec["action"], "abstain action recompute diverged"
    outcome = "answer" if action == "pass" else "no_answer"
    sel_rows = ctx.selection_trace[sel_trace_start:]
    jev_calls = (len(loop_trace)
                 + sum(int(r["jev_calls"]) for r in sel_rows) + 1)
    jev_tokens = (sum(int(t["input_tokens"]) for t in loop_trace)
                  + sum(int(r["jev_input_tokens"]) for r in sel_rows)
                  + int(arec["input_tokens"]))
    return {
        "question_id": qid,
        "question": qtext,
        "gold": gold,
        "outcome": outcome,
        "answer": record["prediction"] if outcome == "answer" else "",
        "prediction": record["prediction"],
        "source": "pipeline",
        "sufficiency": {"record": record, "trace": loop_trace},
        "abstain": {"record": arec, "trace": atrace},
        "selection": {"rounds": sel_rows,
                      "calls": sum(int(r["jev_calls"]) for r in sel_rows)},
        "cache": cache_trace,
        "selection_calls": sum(int(r["jev_calls"]) for r in sel_rows),
        "jev_calls": jev_calls,
        "jev_input_tokens": jev_tokens,
        "jev_cost_usd": estimate_cost_usd(jev_tokens),
        "thresholds": {k: v for k, v in ctx.joint_modes.items()
                       if not k.startswith("_")},
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
    }


def run_pipeline(ctx: PipelineContext,
                 questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every question through :func:`answer_question`, in order."""
    texts = [str(q["question"]) for q in questions]
    dupes = sorted({t for t in texts if texts.count(t) > 1})
    if dupes:
        raise ValueError(
            "CRC row keys (question texts) must be unique per run; "
            f"duplicates: {dupes[:3]}"
        )
    return [answer_question(q, ctx) for q in questions]
