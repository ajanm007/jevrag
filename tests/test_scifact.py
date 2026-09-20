"""Tests for jevrag.benchmarks.scifact — all offline, synthetic files only."""

import pytest

from jevrag.benchmarks import scifact as sc


def test_ndcg_perfect_and_worst():
    rels = {"a": 1}
    assert sc.ndcg_at_k(["a", "b", "c"], rels) == pytest.approx(1.0)
    assert sc.ndcg_at_k(["b", "c", "a"], rels, k=2) == pytest.approx(0.0)
    assert sc.ndcg_at_k(["b", "c"], {}) == pytest.approx(0.0)  # no relevants


def test_ndcg_graded_ranking():
    rels = {"a": 2, "b": 1}
    perfect = sc.ndcg_at_k(["a", "b"], rels)
    swapped = sc.ndcg_at_k(["b", "a"], rels)
    assert perfect == pytest.approx(1.0)
    assert 0.0 < swapped < 1.0


def test_rank_by_relevance_bypass_sinks_last():
    order = sc.rank_by_relevance(["x", "y", "z"],
                                 {"x": 0.2, "y": None, "z": 0.9})
    assert order == ["z", "x", "y"]


def test_loader_roundtrip_and_test_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"_id": "d1", "title": "T1", "text": "abstract one"}\n'
        '{"_id": "d2", "title": "T2", "text": "abstract two"}\n',
        encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"_id": "7", "text": "claim seven"}\n',
                       encoding="utf-8")
    qrels = tmp_path / "q.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n7\td2\t1\n",
                     encoding="utf-8")
    c = sc.load_corpus(corpus)
    assert c["d1"]["title"] == "T1"
    assert sc.load_queries(queries) == {"7": "claim seven"}
    qr = sc.load_qrels(qrels)
    assert qr == {"7": {"d2": 1}}
    assert sc.test_query_ids(qr) == ["7"]


def test_bm25_top_k_shape():
    corpus = {"d1": {"title": "T1", "text": "quantum physics entanglement"},
              "d2": {"title": "T2", "text": "cooking pasta recipes"},
              "d3": {"title": "T3", "text": "quantum computing qubits"}}
    ids, index = sc.build_bm25_index(corpus)
    top = sc.bm25_top_k("quantum physics", ids, index, k=2)
    assert [d for d, _ in top] == ["d1", "d3"]
    assert all(s >= 0 for _, s in top)
