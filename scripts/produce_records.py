"""Produce decision-path records for the eval bar (live run, Jev + generator).

    py -3.13 scripts/produce_records.py --split val
    py -3.13 scripts/produce_records.py --split test

This is the portable reference path. The equivalently-shaped Kaggle port
(local Qwen3-8B on 2xT4 instead of the OpenRouter client, because the
OpenRouter account has no purchased credits) lives at
``kaggle/jevrag_records_kaggle.ipynb`` — gitignored, so it is not part of the
tracked repo, but it lives in this working tree. It imports the same ``jevrag``
loop functions and copies ``build_generator_context`` verbatim, so it runs the
same loop rather than a parallel one.

For each question in the frozen split, produces TWO records from the SAME
retrieval (BM25, prebuilt RAG-Gate index) and the SAME generator (one OpenRouter
model, temperature 0, RAG-Gate's prompt template) — only the stopping policy
differs:

- gated:    run_question(gate=True)  -> Jev "is this enough?" per round
- baseline: run_fixed_iteration(n_rounds=3) -> no gate, always 3 rounds

Generator parity between arms is a hard guard — satisfied here
structurally: both arms call the same ``answer_fn`` closure over the same client
and model, and the generator id is recorded on every record.

Checkpointed after every question: records append to disk, so a throttled or
interrupted run resumes where it left off (RAG-Gate checkpointer discipline).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevrag._rag_gate import rag_gate_root  # noqa: E402

sys.path.insert(0, str(rag_gate_root()))  # src.bm25_retriever

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402

from jevrag.baselines.fixed_iteration import run_fixed_iteration  # noqa: E402
from jevrag.benchmarks.hotpotqa import load_questions  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.primitives.sufficiency import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    DEFAULT_THRESHOLD,
    run_question,
    validate_handoff_record,
)

ASSETS = rag_gate_root() / "hotpot qa"  # prebuilt bm25_index.pkl + corpus
PROMPT_TEMPLATE_FILE = rag_gate_root() / "data" / "prompt_template.txt"

GENERATOR_MODEL = "qwen/qwen3-8b"  # via OpenRouter; temperature 0
GENERATOR_ID = f"openrouter:{GENERATOR_MODEL}"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Generator prompt budget (chars). OpenRouter enforces a per-request
#: prompt-token ceiling tied to remaining credits — observed 1096 tokens at
#: ~$0.04 balance, and round-3 prompts (15 docs, 6000-char cap) hit 1317.
#: ~2500 chars ~= 600-650 tokens, leaving headroom for the template and
#: question even under a squeezed ceiling. This caps ONLY what goes to the
#: generator: retrieval and the Jev state are untouched, so the gate keeps
#: seeing full evidence and existing records stay comparable.
GENERATOR_TOTAL_BUDGET_CHARS = 2500
#: Per-document cap inside the generator budget. Whole documents only.
GENERATOR_PER_DOC_CHARS = 300


def build_generator_context(
    evidence: list[dict],
    per_doc_chars: int = GENERATOR_PER_DOC_CHARS,
    total_budget_chars: int = GENERATOR_TOTAL_BUDGET_CHARS,
) -> str:
    """Assemble generator context doc-by-doc within a character budget.

    Each doc is trimmed to ``per_doc_chars`` (at a word boundary, with an
    ellipsis marker), then docs are added until ``total_budget_chars`` is
    exhausted. A doc that does not fit is dropped whole — the context never
    ends mid-document at an arbitrary concatenation boundary (the old
    ``[:max_context_chars]`` did, handing the model a trailing fragment).
    """
    parts: list[str] = []
    used = 0
    for d in evidence:
        text = str(d.get("text", ""))
        if len(text) > per_doc_chars:
            cut = text[:per_doc_chars].rsplit(" ", 1)[0]
            text = (cut if cut else text[:per_doc_chars]) + "…"
        if parts and used + len(text) > total_budget_chars:
            break
        parts.append(f"[{d.get('title', '')}] {text}")
        used += len(text)
    return "\n\n".join(parts)


def make_retrieve_fn():
    """BM25 retrieval over the prebuilt RAG-Gate index; cumulative top-n."""
    from src.bm25_retriever import bm25_retrieve, load_bm25_index
    from src.data_loader import load_corpus

    bm25 = load_bm25_index(ASSETS / "bm25_index.pkl")
    documents = load_corpus(ASSETS / "corpus.json")
    print(f"retriever ready: BM25 over {len(documents)} documents")

    def retrieve_fn(question: str, n_docs: int) -> list[dict]:
        return bm25_retrieve(question, bm25, documents, top_k=n_docs)["chunks"]

    return retrieve_fn


def make_answer_fn(client, model: str, template: str,
                   per_doc_chars: int = GENERATOR_PER_DOC_CHARS,
                   total_budget_chars: int = GENERATOR_TOTAL_BUDGET_CHARS):
    """One generator closure shared by both arms. Returns (answer, usage dict).

    Usage-dict form (fixed 2026-09-19, Cline seat): ``_call_answer_fn`` in
    ``sufficiency.py`` only counts dict-form usage, so this closure returns
    ``(answer, {"input_tokens": n, "output_tokens": m})`` and generation tokens
    land in ``record["input_tokens"]`` for BOTH arms. It previously returned a
    bare int, which meant both arms silently recorded Jev-only tokens (the
    baseline's 0 meant "unmeasured", not "free").

    Provenance note: the 180 pre-existing val records were written under the old
    bare-int regime, so their ``input_tokens`` are Jev-only. Val is therefore
    mixed-accounting across the 180/120 boundary; the test split is written
    entirely under this regime and its cost table is internally consistent.
    """

    def answer_fn(question: str, evidence: list[dict]) -> tuple[str, dict]:
        context = build_generator_context(
            evidence, per_doc_chars=per_doc_chars,
            total_budget_chars=total_budget_chars,
        )
        prompt = template.replace("{context}", context).replace("{question}", question)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=32,
        )
        answer = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        return answer, {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

    return answer_fn


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["question_id"])
    return done


def append_record(path: Path, record: dict) -> None:
    validate_handoff_record(record)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke)")
    args = ap.parse_args()

    records_dir = REPO_ROOT / "records"
    records_dir.mkdir(exist_ok=True)
    gated_path = records_dir / f"sufficiency_hotpotqa_{args.split}.jsonl"
    baseline_path = records_dir / f"fixed_iter3_hotpotqa_{args.split}.jsonl"

    questions = load_questions(split=args.split)
    if args.limit:
        questions = questions[: args.limit]

    gated_done = load_done_ids(gated_path)
    baseline_done = load_done_ids(baseline_path)
    todo = [q for q in questions if q["id"] not in gated_done or q["id"] not in baseline_done]
    print(f"split={args.split}: {len(questions)} questions, "
          f"{len(questions) - len(todo)} already done, {len(todo)} to run")

    if not todo:
        print("nothing to do")
        return 0

    from openai import OpenAI

    jev = JevDecision(api_key=load_jev_key())
    or_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=load_openrouter_key())
    template = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    retrieve_fn = make_retrieve_fn()
    answer_fn = make_answer_fn(or_client, GENERATOR_MODEL, template)

    t0 = time.perf_counter()
    for i, q in enumerate(todo, 1):
        qid = str(q["id"])
        if qid not in gated_done:
            rec = run_question(
                q, decision=jev, retrieve_fn=retrieve_fn, answer_fn=answer_fn,
                threshold=args.threshold, max_rounds=DEFAULT_MAX_ROUNDS,
            )
            rec["generator"] = GENERATOR_ID
            append_record(gated_path, rec)
        if qid not in baseline_done:
            base = run_fixed_iteration(
                q, n_rounds=DEFAULT_MAX_ROUNDS, retrieve_fn=retrieve_fn,
                answer_fn=answer_fn,
            )
            base["generator"] = GENERATOR_ID
            append_record(baseline_path, base)
        if i % 10 == 0 or i == len(todo):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            print(f"  [{i}/{len(todo)}] {rate:.2f} q/s", flush=True)

    print(f"done: {gated_path} + {baseline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
