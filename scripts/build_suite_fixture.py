"""Build the A6 smoke fixture (packet V03-A6 §6): HotpotQA in contract format.

20 val questions (first unique-text, file order — the same pick rule as the
A4/A5 arms runner, so the smoke is continuous with those runs), ``shared``
mode with the REAL pool file (the full 9,769-doc hotpotqa_corpus.json),
10 calibrate + 10 report by file order (stated, not stratified — n=20 is a
wiring proof, not a sample). supporting_ids resolve HotpotQA supporting
titles against pool titles by exact match; a question whose titles do not
all resolve is SKIPPED loudly (never shipped with empty supporting_ids —
that is a contract build error).

Writes (v03_a6_ prefix):
  outputs/v03_a6_hotpotqa_records.jsonl  (header + 20 rows)
  outputs/v03_a6_hotpotqa_pool.jsonl     (9,769 pool rows)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevrag._rag_gate import rag_gate_root  # noqa: E402
from jevrag.benchmarks.hotpotqa import load_questions  # noqa: E402

N_TOTAL = 20
N_CALIBRATE = 10
OUT_RECORDS = REPO_ROOT / "outputs" / "v03_a6_hotpotqa_records.jsonl"
OUT_POOL = REPO_ROOT / "outputs" / "v03_a6_hotpotqa_pool.jsonl"


def main() -> int:
    questions = load_questions("val")
    corpus = json.loads(
        (rag_gate_root() / "data" / "hotpotqa_corpus.json"
         ).read_text(encoding="utf-8"))["documents"]
    pool = [{"id": f"hp{i}", "title": str(d["title"]),
             "text": str(d["text"])} for i, d in enumerate(corpus)]
    by_title: dict[str, list[str]] = {}
    for p in pool:
        by_title.setdefault(p["title"], []).append(p["id"])
    print(f"pool: {len(pool)} paragraphs", flush=True)

    picked, seen, skipped = [], set(), []
    for q in questions:
        t = str(q["question"])
        if t in seen:
            continue
        need = [str(x) for x in q["supporting_facts"]["titles"]]
        ids = [i for title in need for i in by_title.get(title, [])]
        if len(ids) < len(need):
            skipped.append((str(q["id"]),
                            [x for x in need if x not in by_title]))
            continue
        seen.add(t)
        picked.append((q, ids))
        if len(picked) == N_TOTAL:
            break
    print(f"picked {len(picked)}, skipped {len(skipped)} "
          f"(unresolved titles): {skipped[:3]}", flush=True)
    if len(picked) < N_TOTAL:
        raise ValueError("not enough resolvable questions")

    header = {"__header__": {
        "dataset": "hotpotqa", "pool_mode": "shared",
        "pool_file": str(OUT_POOL),
        "source": {"repo": "RAG-Gate checkout (read in place, never copied)",
                   "file": "data/hotpotqa_1000.json + data/hotpotqa_corpus.json",
                   "license": "private research checkout"},
        "seed": 20261001, "split_seed": 20261002,
        "n": N_TOTAL, "n_calibrate": N_CALIBRATE,
        "n_report": N_TOTAL - N_CALIBRATE,
        "sample_rule": ("first 20 unique-text val questions in file order "
                        "with all supporting titles resolving; "
                        "first 10 calibrate, next 10 report (NOT stratified, "
                        "wiring proof only)"),
        "metric_definitions": {}}}
    with open(OUT_RECORDS, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for i, (q, ids) in enumerate(picked):
            f.write(json.dumps({
                "id": str(q["id"]), "question": str(q["question"]),
                "answer": str(q.get("answer", "")), "aliases": [],
                "type": str(q.get("type", "unknown")),
                "hops": None,
                "split": "calibrate" if i < N_CALIBRATE else "report",
                "supporting_ids": ids,
                "paragraphs": [],
            }) + "\n")
    with open(OUT_POOL, "w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    print(f"wrote {OUT_RECORDS} + {OUT_POOL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
