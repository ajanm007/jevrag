"""Live memo-on pipeline run (packet V03-A1, tracker item I-2).

Same 7-question DocBench run as packets 28/29 (P19-1598.pdf, default
thresholds, seeded toy cache on doc0:4, OpenRouter 4o-mini generation +
judge), with PipelineConfig(memoize_selection=True, memoize_chunk=True).

Writes (v03_a1 prefix — this packet's own files):
  outputs/v03_a1_pipeline_doc_memo.jsonl
  outputs/v03_a1_pipeline_doc_memo.trace.json

Off baseline for comparison: outputs/pipeline_doc29_default.trace.json
(selection 96 calls, 146 total, $0.0072). Live Jev nondeterminism means
small off-vs-on deltas beyond cache savings; the deterministic
equivalence proof is tests/test_memoization.py, not this run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402

from jevrag._rag_gate import jevrag_kaggle_root, rag_gate_root  # noqa: E402
from jevrag.benchmarks.docbench import (  # noqa: E402
    LICENSE_NOTE,
    TEXT_ONLY,
    extract_pdf_pages,
    load_docbench_questions,
)
from jevrag.benchmarks.hotpotqa import f1_score  # noqa: E402
from jevrag.benchmarks.llm_judge import JUDGE_MODEL, make_judge_fn  # noqa: E402
from jevrag.decision import JevDecision, estimate_cost_usd  # noqa: E402
from jevrag.pipeline import (  # noqa: E402
    PipelineConfig,
    build_pipeline,
    normalize_query,
    run_pipeline,
)

DOC_DIR = jevrag_kaggle_root() / "docbench-input" / "0"
PDF_NAME = "P19-1598.pdf"
QA_NAME = "0_qa.jsonl"
DOC_ID = "doc0"
PROMPT_TEMPLATE_FILE = rag_gate_root() / "data" / "prompt_template.txt"
GENERATOR_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SEED_CACHE_QID = "doc0:4"
ALL_TYPES = (TEXT_ONLY, "multimodal-t", "meta-data")
OUT_PREFIX = "outputs/v03_a1_pipeline_doc_memo"


def main() -> int:
    print(f"license: {LICENSE_NOTE}", flush=True)
    from openai import OpenAI
    from scripts.produce_records import make_answer_fn

    pages = extract_pdf_pages(DOC_DIR / PDF_NAME)
    questions = load_docbench_questions(DOC_ID, DOC_DIR / QA_NAME,
                                        types=ALL_TYPES)
    print(f"doc: {PDF_NAME}: {len(pages)} pages, questions: {len(questions)}",
          flush=True)

    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    answer_fn = make_answer_fn(or_client, GENERATOR_MODEL, template)
    judge_fn = make_judge_fn(or_client, JUDGE_MODEL)
    jev_key = load_jev_key()
    backends = {j: JevDecision(api_key=jev_key)
                for j in ("chunk", "selection", "sufficiency", "abstain",
                          "cache")}

    seed_q = next(q for q in questions if q["id"] == SEED_CACHE_QID)
    cache_entries = {normalize_query(seed_q["question"]):
                     {"answer": seed_q["answer"], "gold": seed_q["answer"]}}

    config = PipelineConfig(
        chunk_decision=backends["chunk"],
        selection_decision=backends["selection"],
        sufficiency_decision=backends["sufficiency"],
        abstain_decision=backends["abstain"],
        cache_decision=backends["cache"],
        answer_fn=answer_fn,
        cache_entries=cache_entries,
        max_chunk_chars=9000,
        memoize_selection=True,
        memoize_chunk=True,
    )
    t0 = time.perf_counter()
    ctx = build_pipeline(config, DOC_ID, pages)
    ing = ctx.ingestion_trace
    print(f"ingestion (cold chunk cache): {ing['n_paragraphs']} paragraphs -> "
          f"{ing['n_candidates']} candidates -> {ing['n_splits']} splits "
          f"(+{ing['force_splits']} force) -> {ing['n_chunks']} chunks; "
          f"jev_calls={ing['jev_calls']} cache_hits={ing.get('n_cache_hits')} "
          f"per_round_k={ctx.per_round_k}", flush=True)

    # Chunk-cache proof live: re-ingest the same document through the same
    # config (same store). No new Jev chunk calls may occur, splits identical
    # by construction (cached verdicts, no fresh jitter).
    ctx2 = build_pipeline(config, DOC_ID, pages)
    ing2 = ctx2.ingestion_trace
    print(f"re-ingest (warm chunk cache): jev_calls={ing2['jev_calls']} "
          f"cache_hits={ing2.get('n_cache_hits')} "
          f"splits={ing2.get('splits')} (first: {ing.get('splits')})",
          flush=True)

    outcomes = run_pipeline(ctx, questions)
    judge_tokens = 0
    for i, (out, q) in enumerate(zip(outcomes, questions), 1):
        served = out["answer"] if out["outcome"] == "answer" else ""
        if served:
            j = judge_fn(q["question"], q["answer"], served)
            judge_tokens += (j["usage"]["input_tokens"]
                             + j["usage"]["output_tokens"])
        else:
            j = {"verdict": False,
                 "reason": "pipeline returned NO ANSWER — nothing served",
                 "raw": None, "model": JUDGE_MODEL,
                 "usage": {"input_tokens": 0, "output_tokens": 0}}
        out["judge_verdict"] = j["verdict"]
        out["judge_reason"] = j["reason"]
        out["f1"] = f1_score(out.get("prediction", ""), q["answer"])
        sel = out.get("selection") or {}
        print(f"  [{i}/{len(questions)}] {out['question_id']} "
              f"src={out['source']} "
              f"rounds={out['sufficiency']['record']['rounds_used'] if out['sufficiency'] else '-'} "
              f"sel_live={out['selection_calls']} "
              f"sel_hits={sel.get('cache_hits', 0)} "
              f"-> {out['outcome']} judge={j['verdict']} "
              f"f1={out['f1']:.3f}", flush=True)

    out_path = REPO_ROOT / f"{OUT_PREFIX}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for out in outcomes:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    jev_calls = ing["jev_calls"] + sum(
        o["jev_calls"] for o in outcomes if o["source"] == "pipeline")
    jev_calls += sum(1 for o in outcomes if o["cache"].get("hit"))
    jev_toks = (ing["jev_input_tokens"]
                + sum(o["jev_input_tokens"] for o in outcomes
                      if o["source"] == "pipeline")
                + sum(int(o["cache"].get("input_tokens") or 0)
                      for o in outcomes))
    sel_calls = sum(o["selection_calls"] for o in outcomes)
    sel_hits = sum((o.get("selection") or {}).get("cache_hits", 0)
                   for o in outcomes)
    n_answer = sum(1 for o in outcomes if o["outcome"] == "answer")
    n_correct = sum(1 for o in outcomes if o["judge_verdict"] is True)
    trace_doc = {
        "packet": "V03-A1",
        "doc": PDF_NAME,
        "doc_id": DOC_ID,
        "memoize_selection": True,
        "memoize_chunk": True,
        "ingestion": ing,
        "reingest_ingestion": ing2,
        "per_round_k": ctx.per_round_k,
        "n_questions": len(outcomes),
        "n_answered": n_answer,
        "n_judge_correct": n_correct,
        "jev_calls_total": jev_calls,
        "jev_calls_ingestion": ing["jev_calls"],
        "jev_calls_selection_live": sel_calls,
        "jev_calls_selection_hits": sel_hits,
        "jev_calls_rest": jev_calls - ing["jev_calls"] - sel_calls,
        "jev_input_tokens": jev_toks,
        "jev_cost_usd": estimate_cost_usd(jev_toks),
        "judge_tokens": judge_tokens,
        "outcomes_path": str(out_path),
    }
    trace_path = REPO_ROOT / f"{OUT_PREFIX}.trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_doc, f, indent=2)
    print(f"\nselection: live={sel_calls} cache_hits={sel_hits} "
          f"(off baseline pkt29: 96 live, 0 hits)", flush=True)
    print(f"Jev total: {jev_calls} calls, {jev_toks} tokens "
          f"(est. ${estimate_cost_usd(jev_toks):.4f})", flush=True)
    print(f"outcomes: {n_answer}/{len(outcomes)} answered, "
          f"{n_correct}/{len(outcomes)} judge-correct", flush=True)
    print(f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
