"""Live end-to-end pipeline run on a real DocBench document (packet 28).

Ingestion (chunk-boundary -> BM25 index) once, then every question through
cache? -> filtered-retrieve sufficiency loop -> generate -> answer-abstain.
Scoring (judge + F1) is measurement-side, in this script — the controller
(jevrag.pipeline) stays pure retrieval-to-verdict.

    py scripts/run_pipeline_doc.py --out-prefix outputs/pipeline_doc_default
    py scripts/run_pipeline_doc.py --crc-joints sufficiency abstain --alpha 0.3 \\
        --out-prefix outputs/pipeline_doc_crc

CRC calibration records are prior runs on the SAME document/generator/Jev
(outputs/doc20_*.jsonl) — the most exchangeable calibration available;
n=7 is thin and is stated wherever the thresholds surface. Live spend:
real Jev calls + OpenRouter generation/judge, reported at the end.
"""

from __future__ import annotations

import argparse
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
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.pipeline import (  # noqa: E402
    JOINT_ABSTAIN,
    JOINT_SUFFICIENCY,
    CrcSpec,
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
GENERATOR_MODEL = "openai/gpt-4o-mini"  # proven low-volume OpenRouter path
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CALIB_DIR = REPO_ROOT / "outputs"

#: Which question (by doc id) gets a seeded toy-cache entry. doc0:4 ("Who is
#: the last author of the paper?") — a stable single-fact answer, the honest
#: case for a cache hit.
SEED_CACHE_QID = "doc0:4"

ALL_TYPES = (TEXT_ONLY, "multimodal-t", "meta-data")


def load_calibration(joint: str, alpha: float) -> CrcSpec:
    """Prior same-doc run -> CrcSpec (scores, judge labels, question-text keys)."""
    name = "abstain" if joint == JOINT_ABSTAIN else "sufficiency"
    rows = [json.loads(l) for l in
            open(CALIB_DIR / f"doc20_{name}.jsonl", encoding="utf-8")
            if l.strip()]
    scores = [float(r["confidence"]) for r in rows]
    correct = [1.0 if r.get("judge_verdict") else 0.0 for r in rows]
    keys = [str(r["question"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"calibration keys not unique for {joint}")
    print(f"  calib {joint}: n={len(rows)} base_rate={sum(correct)/len(correct):.3f} "
          f"alpha={alpha}", flush=True)
    return CrcSpec(alpha=alpha, scores=scores, correct=correct, keys=keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crc-joints", nargs="*", default=[],
                    choices=[JOINT_SUFFICIENCY, JOINT_ABSTAIN],
                    help="joints with CRC opt-in (default: all-default thresholds)")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="error budget for CRC joints")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-prefix", default="outputs/pipeline_doc_default")
    args = ap.parse_args()
    print(f"license: {LICENSE_NOTE}", flush=True)

    from openai import OpenAI
    from scripts.produce_records import make_answer_fn

    pages = extract_pdf_pages(DOC_DIR / PDF_NAME)
    all_questions = load_docbench_questions(DOC_ID, DOC_DIR / QA_NAME,
                                            types=ALL_TYPES)
    questions = all_questions[:args.limit] if args.limit else all_questions
    print(f"doc: {PDF_NAME}: {len(pages)} pages, "
          f"questions: {len(questions)}", flush=True)

    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    answer_fn = make_answer_fn(or_client, GENERATOR_MODEL, template)
    judge_fn = make_judge_fn(or_client, JUDGE_MODEL)
    jev_key = load_jev_key()
    backends = {j: JevDecision(api_key=jev_key)
                for j in ("chunk", "selection", "sufficiency", "abstain",
                          "cache")}

    seed_q = next(q for q in all_questions if q["id"] == SEED_CACHE_QID)
    cache_entries = {normalize_query(seed_q["question"]):
                     {"answer": seed_q["answer"], "gold": seed_q["answer"]}}
    print(f"toy cache: 1 seeded entry ({SEED_CACHE_QID}; "
          "exact-normalized-match only, disclosed stand-in)", flush=True)

    crc_specs = {j: load_calibration(j, args.alpha) for j in args.crc_joints}
    config = PipelineConfig(
        chunk_decision=backends["chunk"],
        selection_decision=backends["selection"],
        sufficiency_decision=backends["sufficiency"],
        abstain_decision=backends["abstain"],
        cache_decision=backends["cache"],
        answer_fn=answer_fn,
        crc_specs=crc_specs,
        cache_entries=cache_entries,
        max_chunk_chars=9000,
    )
    t0 = time.perf_counter()
    ctx = build_pipeline(config, DOC_ID, pages)
    ing = ctx.ingestion_trace
    print(f"ingestion: {ing['n_paragraphs']} paragraphs -> "
          f"{ing['n_candidates']} CB candidates -> {ing['n_splits']} splits "
          f"(+{ing['force_splits']} force) -> {ing['n_chunks']} chunks; "
          f"per_round_k={ctx.per_round_k}", flush=True)
    for joint, mode in ctx.joint_modes.items():
        if joint.startswith("_"):
            continue
        print(f"  joint {joint}: {mode}", flush=True)

    outcomes = run_pipeline(ctx, questions)
    judge_tokens = 0
    for i, (out, q) in enumerate(zip(outcomes, questions), 1):
        served = out["answer"] if out["outcome"] == "answer" else ""
        # Judge the served answer when there is one (uniform scoring); a
        # NO ANSWER has nothing to judge and is deterministically incorrect
        # as a served response (same rule as llm_judge's empty-prediction).
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
        gen_toks = (out["sufficiency"]["record"]["input_tokens"]
                    - sum(t["input_tokens"]
                          for t in out["sufficiency"]["trace"])) \
            if out["sufficiency"] else 0
        print(f"  [{i}/{len(questions)}] {out['question_id']} "
              f"src={out['source']} rounds={out['sufficiency']['record']['rounds_used'] if out['sufficiency'] else '-'} "
              f"sel_calls={out['selection_calls']} "
              f"abst_conf={out['abstain']['record']['confidence'] if out['abstain'] else '-'} "
              f"-> {out['outcome']} judge={j['verdict']} "
              f"f1={out['f1']:.3f}", flush=True)

    out_path = REPO_ROOT / f"{args.out_prefix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for out in outcomes:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    jev_calls = ctx.ingestion_trace["jev_calls"] + sum(
        o["jev_calls"] for o in outcomes if o["source"] == "pipeline")
    jev_calls += sum(1 for o in outcomes if o["cache"].get("hit"))
    jev_toks = (ctx.ingestion_trace["jev_input_tokens"]
                + sum(o["jev_input_tokens"] for o in outcomes
                      if o["source"] == "pipeline")
                + sum(int(o["cache"].get("input_tokens") or 0)
                      for o in outcomes))
    sel_calls = sum(o["selection_calls"] for o in outcomes)
    n_answer = sum(1 for o in outcomes if o["outcome"] == "answer")
    n_correct = sum(1 for o in outcomes if o["judge_verdict"] is True)
    trace_doc = {
        "doc": PDF_NAME,
        "doc_id": DOC_ID,
        "crc_joints": args.crc_joints,
        "alpha": args.alpha,
        "ingestion": ctx.ingestion_trace,
        "joint_modes": {k: v for k, v in ctx.joint_modes.items()
                        if not k.startswith("_")},
        "per_round_k": ctx.per_round_k,
        "n_questions": len(outcomes),
        "n_answered": n_answer,
        "n_judge_correct": n_correct,
        "jev_calls_total": jev_calls,
        "jev_calls_ingestion": ctx.ingestion_trace["jev_calls"],
        "jev_calls_selection": sel_calls,
        "jev_calls_rest": (jev_calls - ctx.ingestion_trace["jev_calls"]
                           - sel_calls),
        "jev_input_tokens": jev_toks,
        "jev_cost_usd": jev_toks / 1e6 * 0.042,
        "judge_tokens": judge_tokens,
        "outcomes_path": str(out_path),
    }
    trace_path = REPO_ROOT / f"{args.out_prefix}.trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_doc, f, indent=2)
    print(f"\ntrace written to {trace_path}", flush=True)
    ing = ctx.ingestion_trace
    print(f"\noutcomes: {n_answer}/{len(outcomes)} answered, "
          f"{n_correct}/{len(outcomes)} judge-correct", flush=True)
    print(f"Jev calls: total={jev_calls} "
          f"(ingestion={ing['jev_calls']} votes over "
          f"{ing['n_candidates']} candidates, "
          f"selection={sel_calls}, "
          f"sufficiency+abstain+cache={jev_calls - ing['jev_calls'] - sel_calls})",
          flush=True)
    print(f"Jev input tokens: {jev_toks} "
          f"(est. ${jev_toks / 1e6 * 0.042:.4f} @ $0.042/Mtok)", flush=True)
    print(f"judge tokens (measurement, not pipeline): {judge_tokens}",
          flush=True)
    print(f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
