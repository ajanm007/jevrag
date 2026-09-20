"""Eval: `rag-jev` relevance judgments through our calibration harness.

Ground truth: HotpotQA supporting facts (offline, indexed, strong labels).
Per question the 10 context paragraphs are the candidates; a paragraph is
relevant iff its title is in the question's supporting-facts titles (the 8
distractors are negatives). Title-level granularity — stated, since
supporting facts are sentence-level within gold paras.

What this measures, plainly: multi-hop QA passage relevance — NOT
`rag-jev`'s published SciFact claim-verification number. SciFact would need
a new download + loader for a different task; HotpotQA is already here with
gold relevance labels, so this is the honest small-scale first pass. Do not
quote these numbers against their SciFact result.

Usage: py -3.13 scripts/eval_rag_jev.py [--n 20] [--min-relevance 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key  # noqa: E402

from jevrag.adapters.rag_jev_selector import (  # noqa: E402
    DEFAULT_MIN_RELEVANCE,
    select_passages,
)
from jevrag.benchmarks.hotpotqa import load_questions  # noqa: E402
from jevrag.eval.calibration import calibration_summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-relevance", type=float,
                    default=DEFAULT_MIN_RELEVANCE)
    ap.add_argument("--out",
                    default="outputs/rag_jev_hotpotqa_val.jsonl")
    args = ap.parse_args()

    questions = load_questions(split="val")[:args.n]
    api_key = load_jev_key()
    all_rows, t0 = [], time.perf_counter()
    for qi, q in enumerate(questions, 1):
        titles = q["context"]["titles"]
        sents = q["context"]["sentences"]
        gold = set(q["supporting_facts"]["titles"])
        passages = [{"id": f"{q['id']}:{i}", "text": f"{t}\n{' '.join(s)}"}
                    for i, (t, s) in enumerate(zip(titles, sents))]
        out = select_passages(
            q["question"], passages, api_key=api_key,
            min_relevance=args.min_relevance)
        for rec, title in zip(out["records"], titles):
            all_rows.append({**rec, "question_id": q["id"], "title": title,
                             "correct": int(title in gold)})
        print(f"[{qi}/{len(questions)}] status={out['status']} "
              f"elapsed={out['elapsed_ms']:.0f}ms", flush=True)

    import numpy as np
    conf = np.array([np.nan if r["confidence"] is None else r["confidence"]
                     for r in all_rows])
    corr = np.array([r["correct"] for r in all_rows], dtype=float)
    cal = calibration_summary(conf, corr)
    n_pos = int(corr.sum())
    print(f"\nrag-jev relevance on HotpotQA val contexts "
          f"(n={len(all_rows)}, positives={n_pos}, "
          f"min_relevance={args.min_relevance})")
    print(f"  AURC {cal['aurc']:.4f} (oracle {cal['oracle_aurc']:.4f}, "
          f"random {cal['random_aurc']:.4f})")
    print(f"  ECE {cal['ece']:.4f}  Brier {cal['brier']:.4f} "
          f"(skill {cal['brier_skill']:.4f})")
    tok_in = sum(r["input_tokens"] for r in all_rows)
    print(f"  tokens: {tok_in} in (Jev-side, via rag-jev usage), "
          f"wall {(time.perf_counter() - t0):.0f}s total")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
