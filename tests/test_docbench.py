"""Tests for jevrag.benchmarks.docbench + llm_judge — all offline.

The one test touching the real sample PDF skips when the file isn't present
(it lives outside the repo); everything else is synthetic. No network, no keys.
"""

import json

import pytest

from jevrag.benchmarks import docbench as db
from jevrag.benchmarks import llm_judge as lj

SAMPLE_DIR = r"D:\JevRAG-kaggle\docbench-sample"


def test_chunk_sizes_overlap_and_pages():
    pages = [{"page": 1, "text": "a" * 1000}, {"page": 2, "text": "b" * 1000}]
    chunks = db.chunk_pages(pages, chunk_chars=500, overlap_chars=100)
    assert len(chunks) > 2
    assert all(len(c["text"]) <= 500 for c in chunks)
    assert chunks[0]["page"] == 1 and chunks[0]["title"].startswith("p1")
    assert any(c["page"] == 2 for c in chunks)
    # Overlap: consecutive chunks share text.
    assert chunks[0]["text"][-100:] == chunks[1]["text"][:100]


def test_chunk_rejects_bad_params():
    with pytest.raises(ValueError):
        db.chunk_pages([{"page": 1, "text": "x"}], chunk_chars=100,
                       overlap_chars=100)


def test_bm25_retrieve_finds_relevant_chunk():
    docs = [
        {"title": "p1", "text": "linked watekst two factual knowledge entities"},
        {"title": "p2", "text": "cooking recipes pasta tomato basil oven"},
        {"title": "p3", "text": "perplexity language model evaluation metric"},
    ]
    index = db.build_bm25_index(docs)
    out = db.bm25_retrieve("factual knowledge entities tokens", index, docs,
                           top_k=1)
    assert out["chunks"][0]["title"] == "p1"
    assert out["score_top1"] > 0
    assert len(out["scores_topk"]) == 1


def test_default_k_for_doc_scales_with_size():
    assert db.default_k_for_doc(41) == 10  # folder-0 calibration point
    assert db.default_k_for_doc(8) == 5  # floor holds for tiny docs
    assert db.default_k_for_doc(200) == 20  # ceiling: Jev state budget
    assert db.default_k_for_doc(993) == 20
    with pytest.raises(ValueError):
        db.default_k_for_doc(0)


def test_make_retrieve_fn_cumulative():
    docs = [{"title": f"p{i}", "text": f"document number {i} content words"}
            for i in range(8)]
    retrieve_fn = db.make_retrieve_fn(db.build_bm25_index(docs), docs)
    assert len(retrieve_fn("document content", 3)) == 3
    assert len(retrieve_fn("document content", 6)) == 6


def test_load_filters_types_and_synthesizes_ids(tmp_path):
    qa = tmp_path / "0_qa.jsonl"
    qa.write_text("\n".join([
        json.dumps({"question": "Q1", "answer": "A1", "type": "text-only",
                    "evidence": "E1"}),
        json.dumps({"question": "Q2", "answer": "A2", "type": "meta-data",
                    "evidence": ""}),
        "",
    ]), encoding="utf-8")
    qs = db.load_docbench_questions("doc0", qa)
    assert len(qs) == 1
    assert qs[0] == {"id": "doc0:0", "question": "Q1", "answer": "A1",
                     "type": "text-only", "evidence": "E1"}
    both = db.load_docbench_questions("doc0", qa,
                                      types=("text-only", "meta-data"))
    assert [q["id"] for q in both] == ["doc0:0", "doc0:1"]


def test_judge_parse_verdict():
    v, r = lj.parse_verdict("VERDICT: CORRECT\nREASON: same fact, reworded")
    assert (v, r) == (True, "same fact, reworded")
    v, _ = lj.parse_verdict("verdict: incorrect\nreason: contradicts gold")
    assert v is False
    v, _ = lj.parse_verdict("looks good to me")
    assert v is None  # never guessed


def test_empty_prediction_short_circuits_no_api_call():
    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw)
            raise AssertionError("must not be called for empty predictions")

    client = type("Cl", (), {"chat": type("Ch", (), {
        "completions": FakeCompletions()})()})()
    judge = lj.make_judge_fn(client, "test-model")
    for blank in ("", "   ", "\n", " \t\n "):
        out = judge("Q?", "gold answer here", blank)
        assert out["verdict"] is False
        assert out["raw"] is None
        assert out["model"] == "test-model"
        assert out["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert calls == []


def test_make_judge_fn_with_fake_client():
    class FakeMsg:
        content = "VERDICT: CORRECT\nREASON: key fact present"

    class FakeResp:
        choices = [type("C", (), {"message": FakeMsg()})()]
        usage = type("U", (), {"prompt_tokens": 120,
                               "completion_tokens": 20})()

    class FakeCompletions:
        def create(self, **kw):
            assert kw["temperature"] == 0
            return FakeResp()

    client = type("Cl", (), {"chat": type("Ch", (), {
        "completions": FakeCompletions()})()})()
    judge = lj.make_judge_fn(client, "test-model")
    out = judge("Q?", "gold answer here", "predicted answer here")
    assert out["verdict"] is True
    assert out["model"] == "test-model"
    assert out["usage"] == {"input_tokens": 120, "output_tokens": 20}
    assert "VERDICT" in out["raw"]


def test_empty_pdf_raises(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "empty.pdf"
    w = PdfWriter()
    w.add_blank_page(width=100, height=100)
    with open(path, "wb") as f:
        w.write(f)
    with pytest.raises(ValueError, match="no extractable text"):
        db.extract_pdf_pages(path)


@pytest.mark.skipif(
    __import__("pathlib").Path(SAMPLE_DIR).joinpath("P19-1598.pdf").exists()
    is False, reason="sample PDF outside repo, not present")
def test_real_sample_extraction_and_loader():
    from pathlib import Path

    pages = db.extract_pdf_pages(Path(SAMPLE_DIR) / "P19-1598.pdf")
    assert len(pages) >= 5  # an ACL paper, multi-page
    chunks = db.chunk_pages(pages)
    assert len(chunks) > len(pages)
    qs = db.load_docbench_questions(
        "doc0", Path(SAMPLE_DIR) / "0_qa.jsonl")
    assert len(qs) == 1 and qs[0]["type"] == "text-only"
