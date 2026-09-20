"""jevrag.benchmarks.scifact — BEIR/SciFact loader for the `rag-jev` repro.

`rag-jev`'s published number: NDCG@10 75.13 on 300 BEIR/SciFact queries
(Jev) vs 72.11 (Ettin) vs 66.47 (BM25), from identical BM25 top-20
candidates. This module loads exactly that setup: BEIR-format corpus,
queries, and test qrels — NOT the original AllenAI SUPPORT/REFUTE
claim-verification task.

⚠️  LICENSE: BEIR SciFact is CC BY-NC 2.0 (non-commercial). Research/
evaluation use here; disclose wherever results surface.

Data lives outside the repo (`D:\\JevRAG-kaggle\\scifact\\`, BEIR zip
layout: corpus.jsonl, queries.jsonl, qrels/test.tsv). The 300 test queries
are the query ids appearing in qrels/test.tsv (queries.jsonl also carries
train queries — do not evaluate on those).

BM25 note: RAG-Gate's `bm25_retriever.py` imports HotpotQA-pathed
`config.settings` at module load, so it doesn't port to a new corpus
cleanly. Same resolution as `docbench.py`: reuse the tokenizer shape +
`rank_bm25` math directly here (minimum needed, no new retrieval
infrastructure, no forced reuse that doesn't fit).
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

#: Test query count in BEIR/SciFact — the reproduction target, not a sample.
N_TEST_QUERIES = 300

LICENSE_NOTE = (
    "BEIR SciFact is CC BY-NC 2.0 (non-commercial). "
    "Used here for research/evaluation purposes only."
)


def tokenize_for_bm25(text: str) -> list[str]:
    """Same shape as RAG-Gate's tokenizer (cited, not reimplemented)."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def load_corpus(path: str | Path) -> dict[str, dict[str, str]]:
    """corpus.jsonl → {doc_id: {"title": ..., "text": ...}}."""
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                corpus[str(row["_id"])] = {"title": row.get("title", ""),
                                           "text": row.get("text", "")}
    return corpus


def load_queries(path: str | Path) -> dict[str, str]:
    """queries.jsonl → {query_id: claim text} (all splits; filter by qrels)."""
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                queries[str(row["_id"])] = row.get("text", "")
    return queries


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """test.tsv → {query_id: {doc_id: score}} (binary 1s in practice)."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = \
                int(row["score"])
    return qrels


def test_query_ids(qrels: dict[str, dict[str, int]]) -> list[str]:
    """The 300 test query ids, sorted numerically for a stable run order."""
    return sorted(qrels.keys(), key=lambda q: int(q))


def build_bm25_index(corpus: dict[str, dict[str, str]]):
    """rank_bm25 over `title + text` per abstract (BEIR convention)."""
    from rank_bm25 import BM25Okapi

    doc_ids = sorted(corpus.keys())
    tokenized = [tokenize_for_bm25(corpus[d]["title"] + " " + corpus[d]["text"])
                 for d in doc_ids]
    return doc_ids, BM25Okapi(tokenized)


def bm25_top_k(query: str, doc_ids: list[str], index, k: int = 20
               ) -> list[tuple[str, float]]:
    """Top-k (doc_id, raw BM25 score), descending."""
    scores = index.get_scores(tokenize_for_bm25(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(doc_ids[i], float(scores[i])) for i in order]


def ndcg_at_k(ranked_doc_ids: list[str], rels: dict[str, int], k: int = 10
              ) -> float:
    """Binary/graded NDCG@k. Standard 2^rel − 1 gains, log2(i+1) discounts."""
    gains = [(2 ** rels.get(d, 0) - 1) for d in ranked_doc_ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2 ** r - 1 for r in rels.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def rank_by_relevance(candidate_ids: list[str],
                      relevance: dict[str, float | None]) -> list[str]:
    """Rank candidates by Jev relevance desc; None (bypassed) sinks last."""
    return sorted(candidate_ids,
                  key=lambda d: (relevance.get(d) is None,
                                 -(relevance.get(d) or 0.0)))
