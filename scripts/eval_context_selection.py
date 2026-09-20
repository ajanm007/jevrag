"""V2 eval: context-selection (Jev, per-passage) vs BM25-score baseline.

Ground truth and why
---------------------
HotpotQA, the frozen RAG-Gate val split (300 questions; 242 bridge /
58 comparison). ``correct`` per candidate = **the passage's title is one of the
question's ``supporting_facts.titles``** — HotpotQA's own "gold document"
label (2 per question by construction, 8 distractors).

Why this dataset: HotpotQA is the safe default when a parallel wrap
approach's choice isn't settled (already indexed, already scored, zero
new data-acquisition risk) — flagged so it can be reconciled against any
other arm's choice, since the two must share data and the ``correct``
definition or the comparison means nothing.

Honest limitations of this label, stated up front:
- Supporting facts are sentence-level; a passage is labeled "gold" at document
  granularity, so a gold document can still contain little useful text.
- Distractors are often topically related, so this label *under*-labels
  genuine relevance (false negatives in the label). Both arms are scored
  against the same labels, so the comparison is fair; the absolute numbers are
  approximate (same caveat chunk-boundary reported for paragraph breaks).
- Retrieval is BM25 top-k, so the candidate set is a rank-biased sample.

Baseline: BM25 rank/score alone — no relevance gate at all. Raw Okapi
BM25 is unbounded, so per query the score is divided by that query's top score
(a per-query, rank-preserving map into [0, 1]); the harness refuses anything
outside [0, 1] by design — chunk-boundary's cosine baseline hit exactly that
guard once, elsewhere in this project.

Usage:
    py -3.13 scripts/eval_context_selection.py --n 30 --top-k 6
    py -3.13 scripts/eval_context_selection.py --n 2 --top-k 3 \\
        --out outputs/context_selection_smoke.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(r"D:\Research\RAG-Gate")))  # src.bm25_retriever

from scripts._load_run_keys import load_jev_key  # noqa: E402

from jevrag.benchmarks.hotpotqa import load_questions  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import (  # noqa: E402 — reuse, unchanged
    accuracy_at_coverage,
    aurc,
    calibration_summary,
    coverage_at_accuracy,
    oracle_aurc,
)
from jevrag.eval.cost import format_cost_table, summarize_cost  # noqa: E402
from jevrag.primitives.context_selection import (  # noqa: E402
    COST_SCOPE_CALL_SHARED,
    COST_SCOPE_PER_PASSAGE,
    DEFAULT_INCLUDE_THRESHOLD,
    decide_query,
    decide_query_batched,
)

ASSETS = Path(r"D:\Research\RAG-Gate\hotpot qa")  # prebuilt index + corpus
DEFAULT_SEED = 20260920
THRESHOLD_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def gold_titles(question: dict) -> set[str]:
    """The question's supporting-fact titles, unescaped.

    Corpus titles and dataset titles are escaped inconsistently (corpus stores
    ``&quot;...&quot;`` where the dataset has real quotes), so both sides go
    through ``html.unescape`` before matching. Matching is exact otherwise —
    no fuzzy matching, which would inflate the label.
    """
    return {html.unescape(str(t)) for t in
            question["supporting_facts"]["titles"]}


def load_question_subset(ids_file: Path) -> list[dict]:
    """The val questions with these exact ids, in file order (matched rerun).

    Packet 12: the matched comparison needs Muse's exact 20 question ids, not
    a fresh seeded sample. Fails loudly if an id is missing from the frozen
    val split — a silent subset swap would wreck the comparison.
    """
    wanted = [line.strip() for line in
              ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(q["id"]): q for q in load_questions(split="val")}
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise ValueError(f"ids not in the frozen val split: {missing}")
    if len(set(wanted)) != len(wanted):
        raise ValueError("duplicate ids in --ids-file")
    return [by_id[qid] for qid in wanted]


def build_context_candidates(question: dict) -> list[dict]:
    """The question's own distractor-context paragraphs as candidates.

    This is the candidate construction a matched rerun needs: it is
    exactly what ``scripts/eval_rag_jev.py`` fed the wrapped adapter —
    ``context.titles`` + ``context.sentences`` in dataset order (2 gold + 8
    distractors by HotpotQA construction, minus whatever a given question
    actually carries). Same candidates, same per-question counts, same order
    as the rag-jev run — the denominator finally matches.
    """
    qid = str(question["id"])
    titles = question["context"]["titles"]
    sents = question["context"]["sentences"]
    return [
        {
            "query": str(question["question"]),
            "query_id": qid,
            "passage_id": f"{qid}:{i}",
            "passage_title": str(title),
            "passage": f"{title}\n{' '.join(sents[i])}",
        }
        for i, title in enumerate(titles)
    ]



def is_gold(title: str, golds: set[str]) -> int:
    return int(html.unescape(str(title)) in golds)


def build_candidates(question: dict, chunks: list[dict]) -> list[dict]:
    """Retrieval chunks -> primitive candidate dicts (per-passage state)."""
    qid = str(question["id"])
    return [
        {
            "query": str(question["question"]),
            "query_id": qid,
            "passage_id": f"{qid}:{rank}",
            "passage_title": str(chunk.get("title", "")),
            "passage": str(chunk.get("text", "")),
        }
        for rank, chunk in enumerate(chunks)
    ]


def bm25_confidences(chunks: list[dict]) -> list[float]:
    """Per-query rank-preserving map of raw BM25 into [0, 1].

    score / max(score) for that query: the rank-1 passage gets 1.0 and the
    rest are scaled relative to it. A rank-normalized retrieval score, *not* a
    probability — its AURC is the meaningful comparison; its ECE/Brier are
    reported for completeness and read accordingly.
    """
    raw = [float(c.get("bm25_score", 0.0)) for c in chunks]
    top = max(raw) if raw else 0.0
    if top <= 0.0:
        return [0.0 for _ in raw]
    return [min(1.0, max(0.0, s / top)) for s in raw]


def make_context_bm25_scorer():
    """BM25 scores for the question's own context paragraphs (matched rerun).

    In ``--from-context`` mode the candidates are NOT retrieval output, so the
    no-gate baseline is: score those same paragraphs by BM25 for the query —
    looked up in the prebuilt index by (unescaped) title — and max-normalize
    per query exactly as :func:`bm25_confidences` does. Reuses RAG-Gate's
    tokenizer and index; nothing reimplemented.
    """
    from src.bm25_retriever import load_bm25_index, tokenize_for_bm25
    from src.data_loader import load_corpus

    bm25 = load_bm25_index(ASSETS / "bm25_index.pkl")
    documents = load_corpus(ASSETS / "corpus.json")
    title_to_idx: dict[str, int] = {}
    for i, d in enumerate(documents):
        title_to_idx.setdefault(html.unescape(str(d["title"])), i)

    def score(question: str, titles: list[str]) -> tuple[list[float], list[float]]:
        scores = bm25.get_scores(tokenize_for_bm25(str(question)))
        raw = []
        for t in titles:
            idx = title_to_idx.get(html.unescape(str(t)))
            raw.append(float(scores[idx]) if idx is not None else 0.0)
        top = max(raw) if raw else 0.0
        norm = ([min(1.0, max(0.0, s / top)) for s in raw]
                if top > 0.0 else [0.0 for _ in raw])
        return raw, norm

    return score



def load_done_query_ids(path: Path) -> set[str]:
    """Checkpoint read: which queries already have rows on disk."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(str(json.loads(line)["query_id"]))
    return done


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_questions(
    questions: list[dict],
    retrieve_fn,
    jev: JevDecision,
    *,
    top_k: int,
    threshold: float,
    out_path: Path,
    arm: str = "per_passage",
    from_context: bool = False,
    context_bm25=None,
) -> None:
    """Decision phase: per-passage or set-level (one call per query), checkpointed.

    ``from_context`` (for a matched rerun): candidates are the question's
    own distractor-context paragraphs — the exact candidate construction
    ``scripts/eval_rag_jev.py`` used — so rows match the rag-jev run's
    denominators and candidate sets. ``top_k`` is unused in this mode.
    """
    done = load_done_query_ids(out_path)
    todo = [q for q in questions if str(q["id"]) not in done]
    calls = 1 if arm == "batched" else top_k
    mode = "from_context" if from_context else "bm25_top_k"
    print(f"{len(questions)} questions ({mode}); {len(done)} already on disk; "
          f"{len(todo)} to run (arm={arm}: {calls} Jev call(s) per question "
          f"-> {len(todo) * calls} calls)", flush=True)
    t0 = time.perf_counter()
    for i, q in enumerate(todo, 1):
        if from_context:
            cands = build_context_candidates(q)
            raw_bm25, bm25_norm = context_bm25(
                q["question"], [c["passage_title"] for c in cands])
        else:
            chunks = retrieve_fn(str(q["question"]), top_k)
            cands = build_candidates(q, chunks)
            raw_bm25 = [float(c.get("bm25_score", 0.0)) for c in chunks]
            bm25_norm = bm25_confidences(chunks)
        if arm == "batched":
            records, _ = decide_query_batched(cands, jev, threshold=threshold)
        else:
            records, _ = decide_query(cands, jev, threshold=threshold)
        golds = gold_titles(q)
        rows = []
        for rank, (cand, rec, bconf) in enumerate(zip(cands, records, bm25_norm)):
            rows.append({
                "query_id": cand["query_id"],
                "query_type": str(q.get("type", "unknown")),
                "passage_id": cand["passage_id"],
                "passage_title": cand["passage_title"],
                "retrieval_rank": rank,
                "relevant": is_gold(cand["passage_title"], golds),
                "bm25_score_raw": raw_bm25[rank],
                "bm25_confidence": float(bconf),
                "jev_confidence": float(rec["confidence"]),
                "jev_selected": bool(rec["selected"]),
                "latency_ms": float(rec["latency_ms"]),
                "input_tokens": int(rec["input_tokens"]),
                "cost_scope": rec["cost_scope"],
                "arm": arm,
                "mode": mode,
                "policy": rec["policy"],
            })
        append_rows(out_path, rows)
        if i % 5 == 0 or i == len(todo):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            print(f"  [{i}/{len(todo)}] {rate:.3f} q/s", flush=True)



def threshold_sweep(confidence: list[float],
                    correct: list[int]) -> list[dict]:
    """Precision/recall of the include action across operating points.

    ``correct`` here means "this passage is a gold document". Recall is over
    the gold passages that made it into the candidate set.
    """
    total_gold = sum(correct)
    out = []
    n = len(confidence)
    for t in THRESHOLD_SWEEP:
        selected = [c >= t for c in confidence]
        n_sel = sum(selected)
        tp = sum(1 for s, c in zip(selected, correct) if s and c)
        out.append({
            "threshold": t,
            "n_included": n_sel,
            "include_rate": n_sel / n if n else float("nan"),
            "precision": (tp / n_sel) if n_sel else float("nan"),
            "recall": (tp / total_gold) if total_gold else float("nan"),
        })
    return out


def rank_wise_gold_rate(rows: list[dict]) -> dict[int, dict]:
    """Gold rate by retrieval rank — how much of the signal is just rank."""
    by_rank: dict[int, list[int]] = {}
    for r in rows:
        by_rank.setdefault(int(r["retrieval_rank"]), []).append(int(r["relevant"]))
    return {k: {"n": len(v), "gold_rate": sum(v) / len(v)}
            for k, v in sorted(by_rank.items())}


def analyze(rows: list[dict], *, threshold: float) -> dict:
    jev_conf = [float(r["jev_confidence"]) for r in rows]
    bm25_conf = [float(r["bm25_confidence"]) for r in rows]
    correct = [int(r["relevant"]) for r in rows]
    n_queries = len({r["query_id"] for r in rows})

    summary: dict = {
        "n_rows": len(rows),
        "n_queries": n_queries,
        "candidates_per_query": (len(rows) / n_queries) if n_queries else None,
        "base_rate_gold": sum(correct) / len(correct) if correct else None,
        "rank_wise_gold_rate": rank_wise_gold_rate(rows),
        "threshold_used": threshold,
        "arms": {},
    }
    for name, conf in (("jev", jev_conf), ("bm25", bm25_conf)):
        cal = calibration_summary(conf, correct)
        cal["oracle_aurc"] = oracle_aurc(correct)
        cal["random_aurc"] = 1.0 - cal["base_rate_correct"]
        cal["accuracy_at_coverage_0.8"] = accuracy_at_coverage(conf, correct, 0.8)
        cal["coverage_at_accuracy_0.8"] = coverage_at_accuracy(conf, correct, 0.8)
        summary["arms"][name] = cal

    summary["aurc_delta_bm25_minus_jev"] = (
        summary["arms"]["bm25"]["aurc"] - summary["arms"]["jev"]["aurc"])
    summary["threshold_sweep_jev"] = threshold_sweep(jev_conf, correct)
    summary["include_all_precision"] = summary["base_rate_gold"]
    summary["cost_scope"] = (COST_SCOPE_CALL_SHARED
                             if any(r.get("cost_scope") == COST_SCOPE_CALL_SHARED
                                    for r in rows) else COST_SCOPE_PER_PASSAGE)
    summary["cost"] = summarize_cost(_cost_rows(rows))
    return summary


def _cost_rows(rows: list[dict]) -> list[dict]:
    """Rows shaped for summarize_cost.

    Per-passage rows go through as-is. Call-shared rows (batched arm) carry the
    whole call's tokens on every row, so they are collapsed to one row per
    query — the call — before any summing. Without this, the batched arm's
    token total would be inflated by ~kx (k = candidates per query).
    """
    if not any(r.get("cost_scope") == COST_SCOPE_CALL_SHARED for r in rows):
        return [{"latency_ms": r["latency_ms"],
                 "input_tokens": r["input_tokens"]} for r in rows]
    per_query: dict[str, dict] = {}
    for r in rows:
        qid = str(r["query_id"])
        cur = per_query.get(qid)
        if cur is None or int(r["input_tokens"]) > int(cur["input_tokens"]):
            per_query[qid] = r
    return [{"latency_ms": r["latency_ms"], "input_tokens": r["input_tokens"]}
            for r in per_query.values()]


def report(summary: dict) -> None:
    a = summary["arms"]
    print(f"\n=== context-selection: {summary['n_queries']} queries, "
          f"{summary['n_rows']} candidate passages "
          f"({summary['candidates_per_query']:.1f}/query), "
          f"gold base rate {summary['base_rate_gold']:.3f} ===")
    print("\ngold rate by retrieval rank (no-gate reference):")
    for rank, st in summary["rank_wise_gold_rate"].items():
        print(f"  rank {rank}: {st['gold_rate']:.3f}  (n={st['n']})")

    for name in ("jev", "bm25"):
        c = a[name]
        print(f"\n--- {name} ---")
        print(f"  AURC {c['aurc']:.4f}  (oracle {c['oracle_aurc']:.4f}, "
              f"random {c['random_aurc']:.4f})")
        print(f"  ECE {c['ece']:.4f}  MCE {c['mce']:.4f}  Brier {c['brier']:.4f} "
              f"(skill {c['brier_skill']:+.4f})")
        print(f"  acc@cov0.8 {c['accuracy_at_coverage_0.8']:.4f}  "
              f"cov@acc0.8 {c['coverage_at_accuracy_0.8']:.4f}")

    print(f"\nAURC delta (bm25 - jev): "
          f"{summary['aurc_delta_bm25_minus_jev']:+.4f}  (+ favors jev)")
    print(f"include-all precision at top-k: {summary['include_all_precision']:.3f}")

    print(f"\ninclude-action sweep (Jev, operating threshold inside the sweep):")
    print(f"  {'thr':>5}{'incl':>7}{'rate':>8}{'prec':>8}{'recall':>8}")
    for row in summary["threshold_sweep_jev"]:
        print(f"  {row['threshold']:>5.2f}{row['n_included']:>7}"
              f"{row['include_rate']:>8.3f}{row['precision']:>8.3f}"
              f"{row['recall']:>8.3f}")

    print(f"\ncost (Jev arm, cost_scope={summary['cost_scope']}):")
    print(format_cost_table(summary["cost"]))


def main() -> int:
    import collections

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="questions sampled from the frozen val split")
    ap.add_argument("--top-k", type=int, default=6,
                    help="BM25 candidates per question (passages judged)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--threshold", type=float, default=DEFAULT_INCLUDE_THRESHOLD)
    ap.add_argument("--arm", choices=["per_passage", "batched"],
                    default="per_passage",
                    help="per_passage: one Jev call per candidate (the "
                         "comparable baseline shape); batched: one call per "
                         "query via the existing multi-question ask()")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="exact question ids, one per line, for a matched "
                         "rerun against another run's exact sample; "
                         "overrides --n/--seed sampling")
    ap.add_argument("--from-context", action="store_true",
                    help="candidates = the question's own distractor-context "
                         "paragraphs (what eval_rag_jev.py fed the rag-jev "
                         "adapter), not BM25 retrieval output; the BM25 "
                         "baseline becomes a same-paragraph score lookup")
    ap.add_argument("--force", action="store_true",
                    help="delete the checkpoint file and re-run everything")
    ap.add_argument("--analyze-only", action="store_true",
                    help="no live calls; summarize rows already on disk")
    args = ap.parse_args()

    if args.out is None:
        args.out = REPO_ROOT / ("outputs/context_selection_hotpotqa_val.jsonl"
                                if args.arm == "per_passage"
                                else "outputs/context_selection_batched.jsonl")
    if args.summary is None:
        args.summary = REPO_ROOT / (
            "reports/context_selection_jev_summary.json" if args.arm == "per_passage"
            else "reports/context_selection_batched_summary.json")

    if args.ids_file is not None:
        sampled = load_question_subset(args.ids_file)
        print(f"matched set: {len(sampled)} question ids from {args.ids_file} "
              "(no sampling)")
    else:
        val = sorted(load_questions(split="val"), key=lambda q: str(q["id"]))
        n = min(args.n, len(val))
        rng = random.Random(args.seed)
        sampled = rng.sample(val, n)
        print(f"frozen val split: {len(val)} questions; seeded sample n={n} "
              f"(seed={args.seed})")
    types = collections.Counter(q["type"] for q in sampled)
    print(f"types={dict(types)}", flush=True)
    print("query ids: " + " ".join(str(q["id"]) for q in sampled), flush=True)

    if args.force and args.out.exists():
        args.out.unlink()
        print(f"--force: removed {args.out}")

    if not args.analyze_only:
        from src.bm25_retriever import bm25_retrieve, load_bm25_index
        from src.data_loader import load_corpus

        bm25 = load_bm25_index(ASSETS / "bm25_index.pkl")
        documents = load_corpus(ASSETS / "corpus.json")
        print(f"retriever ready: BM25 over {len(documents)} documents "
              "(prebuilt RAG-Gate index, unchanged)", flush=True)

        def retrieve_fn(question: str, k: int) -> list[dict]:
            return bm25_retrieve(question, bm25, documents, top_k=k)["chunks"]

        jev = JevDecision(api_key=load_jev_key())
        run_questions(sampled, retrieve_fn, jev, top_k=args.top_k,
                      threshold=args.threshold, out_path=args.out, arm=args.arm,
                      from_context=args.from_context,
                      context_bm25=(make_context_bm25_scorer()
                                    if args.from_context else None))
        jev.close()
    elif not args.out.exists():
        print(f"nothing on disk at {args.out} and --analyze-only was set")
        return 1

    rows = read_rows(args.out)
    summary = analyze(rows, threshold=args.threshold)
    summary.update({
        "dataset": "hotpotqa",
        "split": "val",
        "seed": args.seed,
        "top_k": args.top_k,
        "correct_definition": (
            "passage title in question.supporting_facts.titles (HotpotQA gold "
            "document label; html.unescape applied to both sides)"),
        "baseline_confidence_definition": (
            "per-query raw BM25 score / that query's max BM25 score — "
            "rank-preserving map into [0,1], not a probability"),
        "rows_file": str(args.out),
        "arm": args.arm,
        "mode": "from_context" if args.from_context else "bm25_top_k",
        "question_ids_file": (str(args.ids_file) if args.ids_file else None),
        "policy": "context_selection_jev",
    })
    report(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    print(f"\nwrote {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



