"""Diagnose chunk-boundary ingestion nondeterminism (packet 29, step 1).

Rebuilds ingestion candidates from scratch on every run (fresh PDF
extraction + paragraph split, so candidate-construction variation would
show) and judges them with live Jev via decide_document, logging the exact
candidate hash plus the raw per-candidate confidence/label each time.
Comparing rows separates the three candidate causes: differing candidate
hashes = construction bug; identical hashes + varying confidences = backend
output variation; identical everything + differing chunks = wrapper bug.

Usage:
    JEVRAG_KAGGLE_PATH=/mnt/d/JevRAG-kaggle python3 scripts/diag_cb_stability.py
        [--runs 7] [--out outputs/cb_stability.jsonl]

Live spend: runs * n_candidates Jev calls (here 7*9=63, ~$0.003).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key  # noqa: E402

from jevrag._rag_gate import jevrag_kaggle_root  # noqa: E402
from jevrag.benchmarks.docbench import extract_pdf_pages  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.pipeline import split_paragraphs  # noqa: E402
from jevrag.primitives.chunk_boundary import (  # noqa: E402
    DEFAULT_SPLIT_THRESHOLD,
    decide_document,
    label_for,
)

DOC_DIR = jevrag_kaggle_root() / "docbench-input" / "0"
PDF_NAME = "P19-1598.pdf"
DOC_ID = "doc0"


def build_candidates() -> list[dict]:
    pages = extract_pdf_pages(DOC_DIR / PDF_NAME)
    paras = split_paragraphs(pages)
    return [
        {
            "doc_id": DOC_ID,
            "boundary_index": i,
            "before": paras[i]["text"],
            "after": paras[i + 1]["text"],
        }
        for i in range(len(paras) - 1)
    ]


def cand_hash(cands: list[dict]) -> str:
    blob = json.dumps(cands, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=1,
                    help="votes per candidate (decide_document repeats=)")
    ap.add_argument("--out", default="outputs/cb_stability.jsonl")
    args = ap.parse_args()

    jev = JevDecision(api_key=load_jev_key())
    out_path = REPO_ROOT / args.out
    if out_path.exists():
        out_path.unlink()
        print(f"note: truncating {out_path}")

    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for run in range(1, args.runs + 1):
            cands = build_candidates()  # fresh extraction every run
            records, trace = decide_document(
                cands, jev, threshold=DEFAULT_SPLIT_THRESHOLD,
                repeats=args.repeats)
            by_idx = {r["boundary_index"]: r for r in records}
            t_by_idx = {t["boundary_index"]: t for t in trace}
            row = {
                "run": run,
                "repeats": args.repeats,
                "n_candidates": len(cands),
                "candidates_sha256": cand_hash(cands),
                "confidences": [round(by_idx[i]["confidence"], 4)
                                for i in range(len(cands))],
                "labels": [label_for(by_idx[i]["confidence"])
                           for i in range(len(cands))],
                "n_splits": sum(1 for r in records
                                if r["result"] == "split"),
                "jev_input_tokens": sum(int(r["input_tokens"])
                                        for r in records),
            }
            if args.repeats > 1:
                row["per_call"] = [
                    [round(v, 4) for v in t_by_idx[i]["confidences"]]
                    for i in range(len(cands))]
            f.write(json.dumps(row) + "\n")
            print(f"[run {run}/{args.runs}] hash={row['candidates_sha256']} "
                  f"splits={row['n_splits']} "
                  f"conf={row['confidences']}", flush=True)
    print(f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
