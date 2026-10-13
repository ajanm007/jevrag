"""Two-arm HotpotQA smoke (packet V03-A4, suite prerequisite).

Gated pipeline vs fixed-round baseline on 20 HotpotQA val questions through
ONE shared BM25 index built once over hotpotqa_corpus.json (9,769 docs),
via build_pipeline_from_chunks (no ingestion) + answer_question_fixed.
Real Jev + openai/gpt-4o-mini, fail-closed on the first generator error
(reuses the A3 guard + fail-closed wrapper; the key is re-tested live with
one tiny call before running, never assumed).

Writes (v03_a4_ prefix; existing non-empty outputs refused):
  outputs/v03_a4_hotpotqa_gated.jsonl / _fixed.jsonl (one row per question)
  outputs/v03_a4_hotpotqa.trace.json

Wiring check on 20 questions, not an accuracy claim.
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
from scripts.run_pipeline_docbench_pilot import (  # noqa: E402
    GeneratorHalted,
    checked_out_paths,
    fail_closed_answer_fn,
)

from jevrag._rag_gate import rag_gate_root  # noqa: E402
from jevrag.benchmarks.docbench import build_bm25_index  # noqa: E402
from jevrag.benchmarks.hotpotqa import exact_match, f1_score  # noqa: E402
from jevrag.benchmarks.hotpotqa import load_questions  # noqa: E402
from jevrag.decision import JevDecision, estimate_cost_usd  # noqa: E402
from jevrag.pipeline import (  # noqa: E402
    PipelineConfig,
    answer_question_fixed,
    build_pipeline_from_chunks,
    run_pipeline,
    run_pipeline_fixed,
)

PROMPT_TEMPLATE_FILE = rag_gate_root() / "data" / "prompt_template.txt"
CORPUS_FILE = rag_gate_root() / "data" / "hotpotqa_corpus.json"
GENERATOR_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
N_QUESTIONS = 20
N_ROUNDS = 3
#: V03-A5: k=5 with oversample 2 over 3 rounds gives pool_n = 10/20/30, so
#: every round surfaces new candidates (cap >= 30). A4's k=20/cap=20 judged
#: the same 20 every round — degenerate by construction.
PER_ROUND_K = 5
CANDIDATE_POOL_CAP = 30
OUT_PREFIX = "outputs/v03_a5_hotpotqa"


def degeneracy_fraction(rows: list[dict]) -> float:
    """Fraction of rounds 2+ with zero new candidates (packet V03-A5 §3).

    Pure function over outcome rows; unit-tested. Round index is the
    position in the per-question selection trace (1-based).
    """
    later = [r for row in rows for r in
             (row.get("selection") or {}).get("rounds", [])[1:]]
    if not later:
        return 0.0
    return sum(1 for r in later if not r.get("new_candidates")) / len(later)


def pick_questions(n: int) -> list[dict]:
    """First ``n`` val questions with unique texts (file order, stated).

    Uniqueness is required by run_pipeline's CRC-key guard; filtering keeps
    the smoke deterministic without hand-picking.
    """
    picked, seen = [], set()
    for q in load_questions("val"):
        t = str(q["question"])
        if t in seen:
            continue
        seen.add(t)
        picked.append({"id": str(q["id"]), "question": t,
                       "answer": str(q.get("answer", "")),
                       "type": str(q.get("type", "unknown"))})
        if len(picked) == n:
            break
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} unique-text val questions")
    return picked


def score(rows: list[dict]) -> dict:
    for r in rows:
        r["em"] = bool(exact_match(r.get("prediction", ""), r.get("gold", "")))
        r["f1"] = f1_score(r.get("prediction", ""), r.get("gold", ""))
    served = [r for r in rows if r["outcome"] == "answer"]
    withheld = [r for r in rows if r["outcome"] != "answer"]
    return {
        "n": len(rows), "n_served": len(served),
        "em_served": sum(1 for r in served if r["em"]),
        "em_withheld": sum(1 for r in withheld if r["em"]),
        "em_overall": sum(1 for r in rows if r["em"]),
        "f1_mean": sum(float(r["f1"]) for r in rows) / len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_QUESTIONS)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--out-prefix", default=OUT_PREFIX)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-degenerate", action="store_true",
                    help="do not fail when >50% of rounds 2+ add no new "
                    "candidates (degeneracy guard, V03-A5 §3)")
    args = ap.parse_args()

    from openai import OpenAI
    from scripts.produce_records import make_answer_fn

    paths = checked_out_paths(
        args.out_prefix, args.overwrite,
        extra_suffixes=("_gated.jsonl", "_fixed.jsonl"))
    gated_path = REPO_ROOT / f"{args.out_prefix}_gated.jsonl"
    fixed_path = REPO_ROOT / f"{args.out_prefix}_fixed.jsonl"
    trace_path = paths["trace"]

    generator_tag = f"openrouter:{GENERATOR_MODEL}"
    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    answer_fn = fail_closed_answer_fn(
        make_answer_fn(or_client, GENERATOR_MODEL, template), generator_tag)
    jev_key = load_jev_key()
    backends = {j: JevDecision(api_key=jev_key)
                for j in ("chunk", "selection", "sufficiency", "abstain",
                          "cache")}

    questions = pick_questions(args.n)
    print(f"questions: {len(questions)} val (first unique-text, file order)",
          flush=True)
    corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))["documents"]
    chunks = [{"title": str(d["title"]), "text": str(d["text"])}
              for d in corpus]
    print(f"corpus: {len(chunks)} docs; building ONE shared BM25 index ...",
          flush=True)
    t0 = time.perf_counter()
    index = build_bm25_index(chunks)

    def fresh_ctx() -> object:
        # Per-arm config (separate memo stores!): sharing one config across
        # arms lets the second arm free-ride the first arm's selection cache
        # (found live: fixed showed 0 live calls after gated). The INDEX is
        # the thing built once (§2); caches are per-arm by construction.
        arm_config = PipelineConfig(
            chunk_decision=backends["chunk"],
            selection_decision=backends["selection"],
            sufficiency_decision=backends["sufficiency"],
            abstain_decision=backends["abstain"],
            cache_decision=backends["cache"],
            answer_fn=answer_fn,
            memoize_selection=True,
            memoize_chunk=True,
            per_round_k=PER_ROUND_K,
            candidate_pool_cap=CANDIDATE_POOL_CAP,
        )
        return build_pipeline_from_chunks(arm_config, "hotpotqa-val20",
                                          chunks, index=index)

    ctx = fresh_ctx()
    print(f"built in {(time.perf_counter() - t0):.0f}s; "
          f"ingestion mode={ctx.ingestion_trace['mode']} "
          f"calls={ctx.ingestion_trace['jev_calls']} "
          f"per_round_k={ctx.per_round_k}", flush=True)

    try:
        gated_rows = run_pipeline(ctx, questions)
        for r in gated_rows:
            r["arm"] = "gated"
            r["generator"] = generator_tag
        ctx_fixed = fresh_ctx()
        fixed_rows = run_pipeline_fixed(ctx_fixed, questions,
                                        n_rounds=args.rounds,
                                        generator_tag=generator_tag)
        for r in fixed_rows:
            r["arm"] = "fixed"
    except GeneratorHalted as exc:
        print(f"\nHALTED: {exc}\nno fallback ran; no outputs written.",
              flush=True)
        return 1

    g_score, f_score = score(gated_rows), score(fixed_rows)
    with open(gated_path, "w", encoding="utf-8") as f:
        for r in gated_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(fixed_path, "w", encoding="utf-8") as f:
        for r in fixed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def costs(rows):
        calls = sum(int(r.get("jev_calls", 0)) for r in rows)
        toks = sum(int(r.get("jev_input_tokens", 0)) for r in rows)
        return calls, toks

    g_calls, g_toks = costs(gated_rows)
    f_calls, f_toks = costs(fixed_rows)
    trace = {
        "packet": "V03-A5", "n_questions": len(questions),
        "n_rounds_fixed": args.rounds, "generator": generator_tag,
        "ingestion": ctx.ingestion_trace,
        "per_round_k": ctx.per_round_k,
        "gated": {"score": g_score, "jev_calls": g_calls,
                  "jev_input_tokens": g_toks,
                  "jev_cost_usd": estimate_cost_usd(g_toks)},
        "fixed": {"score": f_score, "jev_calls": f_calls,
                  "jev_input_tokens": f_toks,
                  "jev_cost_usd": estimate_cost_usd(f_toks)},
        "jev_cost_usd_total": estimate_cost_usd(g_toks + f_toks),
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    for arm, s, c, t in (("gated", g_score, g_calls, g_toks),
                         ("fixed", f_score, f_calls, f_toks)):
        print(f"{arm}: served {s['n_served']}/{s['n']} "
              f"(EM served {s['em_served']}, withheld {s['em_withheld']}, "
              f"overall {s['em_overall']}, F1 {s['f1_mean']:.3f}); "
              f"Jev {c} calls, {t} toks "
              f"(${estimate_cost_usd(t):.4f})", flush=True)
    deg = degeneracy_fraction(gated_rows + fixed_rows)
    print(f"degeneracy: fraction of rounds 2+ with zero new candidates = "
          f"{deg:.3f}", flush=True)
    print(f"wall {(time.perf_counter() - t0):.0f}s", flush=True)
    if deg > 0.5 and not args.allow_degenerate:
        print("DEGENERATE: >50% of rounds 2+ added nothing — the config "
              "cannot separate the arms (see A4). Rerun with "
              "--allow-degenerate to accept this.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
