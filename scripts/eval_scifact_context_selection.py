"""Packet 14: our from-scratch context_selection.py on rag-jev's home benchmark.

Same setup Muse's packet-13 reproduction used (scripts/eval_scifact_ragjev.py,
imported loader, untouched): 300 BEIR/SciFact test queries, BM25 top-20
candidates, rank by Jev relevance, NDCG@10 — plus the usual calibration
(AURC/ECE/Brier) via the untouched harness. The only difference is the
decision path: our per-passage `decide_query` instead of the rag-jev adapter.

⚠️  LICENSE: BEIR SciFact is CC BY-NC 2.0 (non-commercial). Research/
evaluation use only — same disclosure as the loader module.

Usage:
    py -3.13 scripts/eval_scifact_context_selection.py [--limit 5]
Results append per query to outputs/scifact_context_selection.jsonl (resume
skips done query ids — a 6000-call run must survive interruption).
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

from jevrag.benchmarks.scifact import (  # noqa: E402 — Muse's loader, as-is
    LICENSE_NOTE,
    N_TEST_QUERIES,
    bm25_top_k,
    build_bm25_index,
    load_corpus,
    load_qrels,
    load_queries,
    ndcg_at_k,
    rank_by_relevance,
    test_query_ids,
)
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import calibration_summary  # noqa: E402
from jevrag.primitives.context_selection import (  # noqa: E402
    DEFAULT_INCLUDE_THRESHOLD,
    decide_query,
)

DATA_DIR = Path(r"D:\JevRAG-kaggle\scifact\scifact")


def load_done_ids(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["query_id"])
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=20,
                    help="candidates per query (20 = rag-jev's published setup)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_INCLUDE_THRESHOLD)
    ap.add_argument("--out", default="outputs/scifact_context_selection.jsonl")
    args = ap.parse_args()

    data = Path(args.data)
    print(f"license: {LICENSE_NOTE}", flush=True)
    corpus = load_corpus(data / "corpus.jsonl")
    queries = load_queries(data / "queries.jsonl")
    qrels = load_qrels(data / "qrels" / "test.tsv")
    qids = test_query_ids(qrels)
    assert len(qids) == N_TEST_QUERIES, \
        f"expected 300 test queries, got {len(qids)}"
    print(f"corpus={len(corpus)} queries(test)={len(qids)}", flush=True)
    doc_ids, index = build_bm25_index(corpus)
    print("bm25 index ready", flush=True)

    if args.limit:
        qids = qids[:args.limit]
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    todo = [q for q in qids if q not in done]
    print(f"todo={len(todo)} done={len(done)} "
          f"(arm=per_passage, top_k={args.top_k} -> "
          f"{len(todo) * args.top_k} Jev calls)", flush=True)

    if todo:
        jev = JevDecision(api_key=load_jev_key())
    t0 = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, qid in enumerate(todo, 1):
            cands = bm25_top_k(queries[qid], doc_ids, index, k=args.top_k)
            primitive_cands = [
                {
                    "query": queries[qid],
                    "query_id": qid,
                    "passage_id": d,
                    "passage_title": corpus[d]["title"],
                    "passage": f"{corpus[d]['title']}\n{corpus[d]['text']}",
                }
                for d, _ in cands
            ]
            records, _ = decide_query(primitive_cands, jev,
                                      threshold=args.threshold)
            rel = {r["passage_id"]: r["confidence"] for r in records}
            jev_rank = rank_by_relevance([d for d, _ in cands], rel)
            bm25_rank = [d for d, _ in cands]
            row = {
                "query_id": qid,
                "ndcg10_jev": ndcg_at_k(jev_rank, qrels[qid]),
                "ndcg10_bm25": ndcg_at_k(bm25_rank, qrels[qid]),
                "judgments": [
                    {"doc_id": r["passage_id"],
                     "relevance": r["confidence"],
                     "correct": int(r["passage_id"] in qrels[qid]),
                     "selected": r["selected"],
                     "latency_ms": r["latency_ms"],
                     "input_tokens": r["input_tokens"]}
                    for r in records],
                "policy": records[0]["policy"] if records else None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 5 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] "
                      f"ndcg10_jev={row['ndcg10_jev']:.3f} "
                      f"ndcg10_bm25={row['ndcg10_bm25']:.3f} "
                      f"{(time.perf_counter() - t0) / max(i, 1):.1f}s/q",
                      flush=True)
    if todo:
        jev.close()

    rows = [json.loads(l) for l in
            open(out_path, encoding="utf-8") if l.strip()]
    import numpy as np

    jev_ndcg = float(np.mean([r["ndcg10_jev"] for r in rows]))
    bm25_ndcg = float(np.mean([r["ndcg10_bm25"] for r in rows]))
    conf = [j["relevance"] for r in rows for j in r["judgments"]]
    corr = [j["correct"] for r in rows for j in r["judgments"]]
    cal = calibration_summary(conf, corr)
    tok = sum(j["input_tokens"] for r in rows for j in r["judgments"])
    n_selected = sum(1 for r in rows for j in r["judgments"] if j["selected"])
    print(f"\nNDCG@10 over {len(rows)} queries: "
          f"ours {jev_ndcg:.4f} (rag-jev repro 0.7390, published 0.7513), "
          f"bm25 {bm25_ndcg:.4f} (published 0.6647)")
    print(f"calibration over {len(conf)} judgments: "
          f"AURC {cal['aurc']:.4f} (oracle {cal['oracle_aurc']:.4f}, "
          f"random {cal['random_aurc']:.4f}), ECE {cal['ece']:.4f}, "
          f"Brier {cal['brier']:.4f} (skill {cal['brier_skill']:+.4f})")
    print(f"selected {n_selected}/{len(conf)} at threshold {args.threshold}; "
          f"total Jev input tokens {tok} (est ${tok / 1e6 * 0.042:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
