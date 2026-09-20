"""Packet 20: four primitives, one shared real document, INDEPENDENT runs.

Document: D:\\JevRAG-kaggle\\docbench-input\\0\\P19-1598.pdf (41 chunks) with
0_qa.jsonl (7 questions: 1 text-only, 3 multimodal-t, 3 meta-data). No live
chaining between primitives — each runs on its own inputs, exactly as today.

- sufficiency: loop over all 7 questions (mechanism for all, judge-scored);
  correctness via llm_judge (EM is meaningless on long-form golds).
- chunk-boundary: real paragraph breaks of the PDF text as weak positives +
  seeded within-paragraph negatives (same discipline as the Wikipedia eval),
  vs the same cosine baseline. Page-break subset reported separately.
- context-selection: top-20 BM25 chunks as candidates for the text-only Q
  (gold evidence span available); labels = evidence-substring match,
  disclosed as constructed-from-gold.
- answer-abstain: grounding check on the generated answer per question,
  correctness via llm_judge verdict (EM doesn't fit long-form prose golds).
- cache-trust: SKIPPED here — a constructed cache scenario on this document
  would repeat the existing hardening eval's construction without new
  signal, and judge-labels on paraphrases would be trivially all-correct.
  Stated, not forced.

Usage: py -3.13 scripts/eval_docbench_primitives.py [--only sufficiency ...]
"""

from __future__ import annotations

import argparse
import json
import random
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
    build_doc_index,
    default_k_for_doc,
    load_docbench_questions,
    make_retrieve_fn,
)
from jevrag.benchmarks.hotpotqa import f1_score  # noqa: E402
from jevrag.benchmarks.llm_judge import JUDGE_MODEL, make_judge_fn  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import calibration_summary  # noqa: E402
from jevrag.primitives.answer_abstain import (  # noqa: E402
    DEFAULT_ABSTAIN_THRESHOLD,
    check_answer,
)
from jevrag.primitives.chunk_boundary import (  # noqa: E402
    decide_document,
    label_for,
)
from jevrag.primitives.context_selection import (  # noqa: E402
    decide_query,
)
from jevrag.primitives.sufficiency import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    DEFAULT_THRESHOLD,
    run_sufficiency_with_trace,
)

DOC_DIR = jevrag_kaggle_root() / "docbench-input" / "0"
PDF_NAME = "P19-1598.pdf"
QA_NAME = "0_qa.jsonl"
DOC_ID = "doc0"
PROMPT_TEMPLATE_FILE = rag_gate_root() / "data" / "prompt_template.txt"
GENERATOR_MODEL = "openai/gpt-4o-mini"  # proven low-volume OpenRouter path
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OUT_PREFIX = "outputs/doc20"

ALL_TYPES = (TEXT_ONLY, "multimodal-t", "meta-data")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_sufficiency(questions, retrieve_fn, answer_fn, judge_fn):
    rows = []
    for i, q in enumerate(questions, 1):
        rec, trace = run_sufficiency_with_trace(
            question_id=q["id"], question=q["question"], gold=q["answer"],
            qtype=q["type"], decision=JEV, retrieve_fn=retrieve_fn,
            answer_fn=answer_fn, threshold=DEFAULT_THRESHOLD,
            max_rounds=DEFAULT_MAX_ROUNDS, per_round_k=K)
        j = judge_fn(q["question"], q["answer"], rec["prediction"])
        rows.append({**rec, "judge_verdict": j["verdict"],
                     "judge_reason": j["reason"],
                     "f1": f1_score(rec["prediction"], q["answer"])})
        print(f"  [suff {i}/{len(questions)}] rounds={rec['rounds_used']} "
              f"judge={j['verdict']} f1={rows[-1]['f1']:.3f} "
              f"q={q['question'][:60]!r}", flush=True)
    return rows


def run_chunk_boundary(pages):
    from scripts.eval_chunk_boundary import SENT_SPLIT, build_candidates

    text = "\n\n".join(p["text"] for p in pages)
    cands = build_candidates(text, DOC_ID, max_neg=50)
    print(f"  chunk-boundary candidates={len(cands)} "
          f"(pos={sum(c['is_split'] for c in cands)})", flush=True)
    records, _ = decide_document(cands, JEV)
    by_idx = {r["boundary_index"]: r for r in records}
    assert len(by_idx) == len(cands)

    print("  loading cosine baseline (MiniLM)...", flush=True)
    from sentence_transformers import SentenceTransformer
    from scripts.eval_chunk_boundary import cosine_scores
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cos = cosine_scores(cands, model)

    import numpy as np
    jev_conf = np.array([by_idx[c["boundary_index"]]["confidence"]
                         for c in cands])
    correct = np.array([c["is_split"] for c in cands], dtype=float)
    rows = []
    for c, jc, cs in zip(cands, jev_conf, cos):
        rows.append({"doc_id": c["doc_id"],
                     "boundary_index": c["boundary_index"],
                     "is_split": c["is_split"], "jev_confidence": float(jc),
                     "cosine_score": float(cs), "jev_label": label_for(jc),
                     "before": c["before"][:200], "after": c["after"][:200],
                     **{k: v for k, v in by_idx[c["boundary_index"]].items()
                        if k in ("latency_ms", "input_tokens", "policy")},})
    cal_j = calibration_summary(jev_conf, correct)
    cal_c = calibration_summary(np.array(cos), correct)
    print(f"  jev: AURC {cal_j['aurc']:.4f} ECE {cal_j['ece']:.4f} "
          f"Brier-skill {cal_j['brier_skill']:.4f}", flush=True)
    print(f"  cos: AURC {cal_c['aurc']:.4f} ECE {cal_c['ece']:.4f} "
          f"Brier-skill {cal_c['brier_skill']:.4f}", flush=True)
    return rows, {"jev": cal_j, "cos": cal_c}


def run_context_selection(question, chunks, index):
    from jevrag.benchmarks.docbench import bm25_retrieve

    res = bm25_retrieve(question["question"], index, chunks, top_k=20)
    cands = res["chunks"]
    gold_ev = (question.get("evidence") or "").strip().strip('"')
    gold_words = {w for w in gold_ev.lower().split() if len(w) > 4}
    labels = []
    for c in cands:
        text = c["text"].lower()
        overlap = sum(1 for w in gold_words if w in text)
        labels.append(1 if gold_words and overlap >= max(3, len(gold_words) // 3)
                      else 0)
    print(f"  context-selection: {len(cands)} candidates, "
          f"{sum(labels)} evidence-positive", flush=True)
    decide_cands = [{"query": question["question"], "query_id": question["id"],
                     "passage": c["text"], "passage_id": c["title"],
                     "passage_title": c["title"]} for c in cands]
    records, _ = decide_query(decide_cands, JEV)
    rows = []
    for r, c, lab in zip(records, cands, labels):
        rows.append({**r, "correct": lab, "bm25_score": c.get("bm25_score", 0.0),
                     "text": c["text"][:300]})
        print(f"    {c['title']}: conf={r['confidence']:.2f} "
              f"sel={r['selected']} gold={lab}", flush=True)
    import numpy as np
    conf = np.array([r["confidence"] for r in rows])
    corr = np.array(labels, dtype=float)
    cal = calibration_summary(conf, corr)
    print(f"  AURC {cal['aurc']:.4f} (oracle {cal['oracle_aurc']:.4f}, "
          f"random {cal['random_aurc']:.4f}) ECE {cal['ece']:.4f} "
          f"Brier-skill {cal['brier_skill']:.4f}", flush=True)
    return rows, cal


def run_abstain(questions, retrieve_fn, answer_fn, judge_fn):
    rows = []
    for i, q in enumerate(questions, 1):
        rec_s, _ = run_sufficiency_with_trace(
            question_id=q["id"], question=q["question"], gold=q["answer"],
            qtype=q["type"], decision=JEV, retrieve_fn=retrieve_fn,
            answer_fn=answer_fn, threshold=DEFAULT_THRESHOLD,
            max_rounds=DEFAULT_MAX_ROUNDS, per_round_k=K)
        evidence = list(retrieve_fn(q["question"],
                                    rec_s["rounds_used"] * K))
        arec, _ = check_answer(
            question_id=q["id"], question=q["question"], gold=q["answer"],
            evidence=evidence, prediction=rec_s["prediction"], decision=JEV,
            threshold=DEFAULT_ABSTAIN_THRESHOLD)
        j = judge_fn(q["question"], q["answer"], rec_s["prediction"])
        rows.append({**arec, "judge_verdict": j["verdict"],
                     "judge_reason": j["reason"],
                     "f1": f1_score(rec_s["prediction"], q["answer"]),
                     "rounds_used": rec_s["rounds_used"]})
        print(f"  [abst {i}/{len(questions)}] conf={arec['confidence']:.2f} "
              f"action={arec['action']} judge={j['verdict']}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["sufficiency", "chunk",
                                                  "selection", "abstain"],
                    help="subset of primitives to run")
    args = ap.parse_args()
    print(f"license: {LICENSE_NOTE}", flush=True)

    global JEV, K
    from openai import OpenAI
    from scripts.produce_records import make_answer_fn

    chunks, index = build_doc_index(DOC_DIR / PDF_NAME)
    print(f"doc: {PDF_NAME}: {len(chunks)} chunks indexed", flush=True)
    K = default_k_for_doc(len(chunks))
    retrieve_fn = make_retrieve_fn(index, chunks)
    questions = load_docbench_questions(DOC_ID, DOC_DIR / QA_NAME,
                                        types=ALL_TYPES)
    text_only = [q for q in questions if q["type"] == TEXT_ONLY]
    print(f"questions: {len(questions)} "
          f"({len(text_only)} text-only), per_round_k={K}", flush=True)

    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    answer_fn = make_answer_fn(or_client, GENERATOR_MODEL, template)
    judge_fn = make_judge_fn(or_client, JUDGE_MODEL)
    JEV = JevDecision(api_key=load_jev_key())

    t0 = time.perf_counter()
    if "sufficiency" in args.only:
        rows = run_sufficiency(questions, retrieve_fn, answer_fn, judge_fn)
        write_jsonl(REPO_ROOT / f"{OUT_PREFIX}_sufficiency.jsonl", rows)
    if "chunk" in args.only:
        from jevrag.benchmarks.docbench import extract_pdf_pages
        pages = extract_pdf_pages(DOC_DIR / PDF_NAME)
        rows, cals = run_chunk_boundary(pages)
        write_jsonl(REPO_ROOT / f"{OUT_PREFIX}_chunk.jsonl", rows)
    if "selection" in args.only:
        rows, cal = run_context_selection(text_only[0], chunks, index)
        write_jsonl(REPO_ROOT / f"{OUT_PREFIX}_selection.jsonl", rows)
    if "abstain" in args.only:
        rows = run_abstain(questions, retrieve_fn, answer_fn, judge_fn)
        write_jsonl(REPO_ROOT / f"{OUT_PREFIX}_abstain.jsonl", rows)
    print(f"wall {(time.perf_counter() - t0):.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
