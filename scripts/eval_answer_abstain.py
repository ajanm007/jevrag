"""Eval: answer-abstain grounding check on HotpotQA val (first cut, n=30).

Per question: run the sufficiency loop (read-only reuse — captures the
evidence actually used at the stopping point + the generated prediction),
re-derive that round's evidence deterministically via BM25, then ask the
abstain question ("is this answer supported by this evidence?").

Ground-truth disclosure: no (answer, grounded) dataset exists.
Proxy labels — EM correctness as the harness label ("wrong" correlates with
"ungrounded", not identical), plus the stronger failure-mode split (wrong
prediction's normalized content absent from vs present in evidence).
Stated with every number; conflating them would be the overclaim §8 bans.

Usage: py -3.13 scripts/eval_answer_abstain.py [--n 30] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(r"D:\Research\RAG-Gate")))

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402
from scripts.produce_records import (  # noqa: E402 — reuse, not duplication
    GENERATOR_MODEL,
    OPENROUTER_BASE_URL,
    PROMPT_TEMPLATE_FILE,
    make_answer_fn,
    make_retrieve_fn,
)

from jevrag.benchmarks.hotpotqa import exact_match, load_questions, normalize_answer  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import calibration_summary  # noqa: E402
from jevrag.primitives.answer_abstain import (  # noqa: E402
    DEFAULT_ABSTAIN_THRESHOLD,
    check_answer,
)
from jevrag.primitives.sufficiency import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    DEFAULT_THRESHOLD,
    run_sufficiency_with_trace,
)


def prediction_in_evidence(prediction: str, evidence: list[dict]) -> bool:
    """Normalized prediction substring of normalized evidence text?"""
    pred = normalize_answer(prediction or "")
    if not pred:
        return False
    hay = normalize_answer(" ".join(str(d.get("text", "")) for d in evidence))
    return pred in hay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threshold", type=float,
                    default=DEFAULT_ABSTAIN_THRESHOLD)
    ap.add_argument("--per-round-k", type=int, default=5)
    ap.add_argument("--out", default="outputs/answer_abstain_val.jsonl")
    args = ap.parse_args()

    from openai import OpenAI

    questions = load_questions(split="val")[:args.n]
    jev = JevDecision(api_key=load_jev_key())
    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = Path(str(PROMPT_TEMPLATE_FILE)).read_text(encoding="utf-8")
    retrieve_fn = make_retrieve_fn()
    answer_fn = make_answer_fn(or_client, GENERATOR_MODEL, template)

    rows, t0 = [], time.perf_counter()
    for i, q in enumerate(questions, 1):
        qid, question, gold = str(q["id"]), q["question"], q["answer"]
        rec, _ = run_sufficiency_with_trace(
            question_id=qid, question=question, gold=gold,
            qtype=q.get("type", "unknown"), decision=jev,
            retrieve_fn=retrieve_fn, answer_fn=answer_fn,
            threshold=DEFAULT_THRESHOLD, max_rounds=DEFAULT_MAX_ROUNDS,
            per_round_k=args.per_round_k)
        # Re-derive the stopping round's evidence (BM25 deterministic → exact).
        evidence = list(retrieve_fn(question, rec["rounds_used"] * args.per_round_k))
        arec, _ = check_answer(
            question_id=qid, question=question, gold=gold, evidence=evidence,
            prediction=rec["prediction"], decision=jev,
            threshold=args.threshold)
        em = exact_match(rec["prediction"], gold)
        rows.append({**arec, "type": q.get("type", "unknown"),
                     "correct": em, "rounds_used": rec["rounds_used"],
                     "in_evidence": prediction_in_evidence(rec["prediction"],
                                                           evidence)})
        print(f"[{i}/{len(questions)}] em={em} conf={arec['confidence']:.2f} "
              f"action={arec['action']} in_ev={rows[-1]['in_evidence']}",
              flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import numpy as np
    conf = np.array([r["confidence"] for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=float)
    cal = calibration_summary(conf, corr, n_bins=5)
    n_abstain = sum(1 for r in rows if r["action"] == "abstain")
    passed = [r for r in rows if r["action"] == "pass"]
    wrong = [r for r in rows if not r["correct"]]
    absent = [r for r in wrong if not r["in_evidence"]]
    print(f"\nn={len(rows)} abstain_rate={n_abstain / len(rows):.3f}")
    print(f"passed EM: {sum(r['correct'] for r in passed) / max(len(passed), 1):.4f} "
          f"(n={len(passed)}) vs overall EM {corr.mean():.4f}")
    print(f"wrong answers: {len(wrong)}, of which prediction absent from "
          f"evidence: {len(absent)} (cleaner ungrounded cases)")
    print(f"abstain-confidence: AURC {cal['aurc']:.4f} (oracle "
          f"{cal['oracle_aurc']:.4f}, random {cal['random_aurc']:.4f}), "
          f"ECE {cal['ece']:.4f}, Brier {cal['brier']:.4f} "
          f"(skill {cal['brier_skill']:.4f})")
    print(f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
