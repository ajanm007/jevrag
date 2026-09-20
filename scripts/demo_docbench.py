"""Live unscored demo: the actual sufficiency loop running over one PDF.

    py -3.13 scripts/demo_docbench.py [--doc DIR] [--question N]

Watches (not measures): prints each round's retrieved docs and Jev gate
confidence, then the final answer against gold, with F1 and one LLM-judge
verdict for chain confirmation. No baseline, no CLI eval, no records file —
the judge call at the end exists to confirm the scoring half of the chain
works, not to produce a metric.

DocBench license disclosure: DocBench's licensing is unclarified — no
LICENSE file exists in its repository and an open issue asking the authors
is unanswered as of 2026-09-20. Used here for research/evaluation only.
The PDF stays outside the repo (default: <JEVRAG_KAGGLE_PATH>\\docbench-sample,
JEVRAG_KAGGLE_PATH itself defaulting to D:\\JevRAG-kaggle).

Design note: the generator is the SAME OpenRouter closure as the HotpotQA
runs (scripts.produce_records.make_answer_fn) — one low-volume use, a few
calls for one question. Jev decides; OpenRouter generates and judges.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402

from jevrag._rag_gate import jevrag_kaggle_root, rag_gate_root  # noqa: E402
from jevrag.benchmarks.docbench import (  # noqa: E402
    LICENSE_NOTE,
    TEXT_ONLY,
    build_doc_index,
    default_k_for_doc,
    load_docbench_questions,
    make_retrieve_fn,
)
from jevrag.benchmarks.hotpotqa import f1_score  # noqa: E402
from jevrag.benchmarks.llm_judge import JUDGE_MODEL, make_judge_fn  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.primitives.sufficiency import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    DEFAULT_THRESHOLD,
    run_sufficiency_with_trace,
)

SAMPLE_DIR = jevrag_kaggle_root() / "docbench-sample"
PDF_NAME = "P19-1598.pdf"
QA_NAME = "0_qa.jsonl"
PROMPT_TEMPLATE_FILE = rag_gate_root() / "data" / "prompt_template.txt"

GENERATOR_MODEL = "qwen/qwen3-8b"  # same closure as the HotpotQA runs
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", default=str(SAMPLE_DIR))
    ap.add_argument("--pdf", default=PDF_NAME)
    ap.add_argument("--qa", default=QA_NAME)
    ap.add_argument("--doc-id", default="doc0")
    ap.add_argument("--question", type=int, default=0,
                    help="index into the doc's text-only questions")
    ap.add_argument("--per-round-k", type=int, default=None,
                    help="docs added per retrieval round (default: auto from "
                         "chunk count via default_k_for_doc — round-1 evidence "
                         "for this doc's Q sits at rank ~6-15, below the "
                         "HotpotQA-tuned 5)")
    args = ap.parse_args()

    from openai import OpenAI
    from scripts.produce_records import make_answer_fn

    doc_dir = Path(args.doc)
    print(f"license: {LICENSE_NOTE}")
    chunks, index = build_doc_index(doc_dir / args.pdf)
    print(f"doc: {args.pdf}: {len(chunks)} chunks indexed")
    retrieve_fn = make_retrieve_fn(index, chunks)
    per_round_k = (args.per_round_k if args.per_round_k is not None
                   else default_k_for_doc(len(chunks)))
    print(f"per_round_k={per_round_k} (auto from {len(chunks)} chunks)" if args.per_round_k is None else f"per_round_k={per_round_k} (explicit)")

    questions = load_docbench_questions(args.doc_id, doc_dir / args.qa,
                                        types=(TEXT_ONLY,))
    q = questions[args.question]
    print(f"\nQ: {q['question']}\n")

    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    answer_fn = make_answer_fn(or_client, GENERATOR_MODEL, template)
    jev = JevDecision(api_key=load_jev_key())

    record, trace = run_sufficiency_with_trace(
        question_id=q["id"], question=q["question"], gold=q["answer"],
        qtype=q["type"], decision=jev, retrieve_fn=retrieve_fn,
        answer_fn=answer_fn, threshold=DEFAULT_THRESHOLD,
        max_rounds=DEFAULT_MAX_ROUNDS, per_round_k=per_round_k,
    )
    for t in trace:
        print(f"round {t['round']}: {t['n_docs']} docs, "
              f"sufficiency p={t['confidence']:.3f} "
              f"({t['latency_ms']:.0f}ms, {t['input_tokens']} tok)")
    print(f"\nstopped after {record['rounds_used']} round(s)")
    print(f"prediction: {record['prediction']}")
    print(f"gold:       {q['answer']}")
    print(f"F1: {f1_score(record['prediction'], q['answer']):.4f} "
          "(partial credit; EM is meaningless on long-form golds)")

    judge = make_judge_fn(or_client, JUDGE_MODEL)
    j = judge(q["question"], q["answer"], record["prediction"])
    print(f"judge ({j['model']}): "
          f"{'CORRECT' if j['verdict'] else 'INCORRECT' if j['verdict'] is False else 'UNPARSEABLE'}"
          f" — {j['reason']}")
    print(f"judge tokens: {j['usage']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
