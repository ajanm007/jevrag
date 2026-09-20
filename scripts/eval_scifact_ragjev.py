"""Reproduce `rag-jev`'s SciFact NDCG@10 + calibrate its relevance scores.

Setup (their published benchmark): 300 BEIR/SciFact test queries, BM25
top-20 candidates per query, rank by Jev relevance, NDCG@10. Published:
75.13 (Jev) / 72.11 (Ettin) / 66.47 (BM25). This script reproduces the Jev
arm and — nearly free, same candidates — our own BM25 arm: if our BM25
lands near 66.47, the candidate construction matches theirs and the Jev
comparison is apples-to-apples; if not, the gap is documented, not hidden.

Usage: py -3.13 scripts/eval_scifact_ragjev.py [--limit 5] [--resume]
Results append per-query to outputs/scifact_ragjev.jsonl (resume skips
done query ids — a 6000-call run must survive interruption).

License: BEIR SciFact is CC BY-NC 2.0, research/evaluation use only.
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
from jevrag._rag_gate import jevrag_kaggle_root  # noqa: E402
from jevrag.benchmarks.scifact import (  # noqa: E402
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
from jevrag.eval.calibration import calibration_summary  # noqa: E402

DATA_DIR = jevrag_kaggle_root() / "scifact" / "scifact"


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
    ap.add_argument("--min-relevance", type=float,
                    default=DEFAULT_MIN_RELEVANCE)
    ap.add_argument("--out", default="outputs/scifact_ragjev.jsonl")
    args = ap.parse_args()

    data = Path(args.data)
    print(f"license: {LICENSE_NOTE}")
    corpus = load_corpus(data / "corpus.jsonl")
    queries = load_queries(data / "queries.jsonl")
    qrels = load_qrels(data / "qrels" / "test.tsv")
    qids = test_query_ids(qrels)
    assert len(qids) == N_TEST_QUERIES, f"expected 300 test queries, got {len(qids)}"
    print(f"corpus={len(corpus)} queries(test)={len(qids)}")
    doc_ids, index = build_bm25_index(corpus)
    print("bm25 index ready")

    if args.limit:
        qids = qids[:args.limit]
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    todo = [q for q in qids if q not in done]
    print(f"todo={len(todo)} done={len(done)}")
    if not todo:
        print("nothing to do")
        return 0

    api_key = load_jev_key()
    t0 = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, qid in enumerate(todo, 1):
            cands = bm25_top_k(queries[qid], doc_ids, index, k=20)
            passages = [{"id": d, "text": f"{corpus[d]['title']}\n{corpus[d]['text']}"}
                        for d, _ in cands]
            out = select_passages(queries[qid], passages, api_key=api_key,
                                  min_relevance=args.min_relevance)
            rel = {r["doc_id"]: r["confidence"] for r in out["records"]}
            jev_rank = rank_by_relevance([d for d, _ in cands], rel)
            bm25_rank = [d for d, _ in cands]
            row = {
                "query_id": qid,
                "status": out["status"],
                "ndcg10_jev": ndcg_at_k(jev_rank, qrels[qid]),
                "ndcg10_bm25": ndcg_at_k(bm25_rank, qrels[qid]),
                "judgments": [
                    {"doc_id": r["doc_id"], "relevance": r["confidence"],
                     "correct": int(r["doc_id"] in qrels[qid]),
                     "selected": r["selected"], "reason": r["reason"],
                     "input_tokens": r["input_tokens"]} for r in out["records"]],
                "usage": out["usage"],
                "elapsed_ms": out["elapsed_ms"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] status={out['status']} "
                      f"ndcg10_jev={row['ndcg10_jev']:.3f} "
                      f"ndcg10_bm25={row['ndcg10_bm25']:.3f} "
                      f"{(time.perf_counter() - t0) / max(i, 1):.1f}s/q",
                      flush=True)

    rows = [json.loads(l) for l in
            open(out_path, encoding="utf-8") if l.strip()]
    import numpy as np
    jev = np.mean([r["ndcg10_jev"] for r in rows])
    bm25 = np.mean([r["ndcg10_bm25"] for r in rows])
    print(f"\nNDCG@10 over {len(rows)} queries: jev {jev:.4f} (published 0.7513), "
          f"bm25 {bm25:.4f} (published 0.6647)")
    conf, corr = [], []
    for r in rows:
        for j in r["judgments"]:
            conf.append(np.nan if j["relevance"] is None else j["relevance"])
            corr.append(j["correct"])
    cal = calibration_summary(np.array(conf), np.array(corr))
    print(f"calibration over {len(conf)} judgments: "
          f"AURC {cal['aurc']:.4f} (oracle {cal['oracle_aurc']:.4f}, "
          f"random {cal['random_aurc']:.4f}), ECE {cal['ece']:.4f}, "
          f"Brier {cal['brier']:.4f} (skill {cal['brier_skill']:.4f})")
    tok = sum(r["usage"]["input_tokens"] for r in rows)
    print(f"total Jev input tokens so far: {tok} "
          f"(est ${tok / 1e6 * 0.042:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
