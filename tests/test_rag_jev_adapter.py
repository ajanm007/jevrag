"""Tests for jevrag.adapters.rag_jev_selector — offline via a fake provider.

The adapter hardcodes the real `Jev` provider (correct for production), so
tests monkeypatch `rag_jev.Jev` with a fake scoring the same shapes. No
network, no key.
"""

import pytest

from jevrag.adapters import rag_jev_selector as adapter


class FakeJev:
    def __init__(self, api_key=None, model="jev-latest"):
        self.model = model

    async def score(self, query, document, guidance):
        from rag_jev.models import Judgment

        hot = "gold" in document.text
        return Judgment(relevance=0.9 if hot else 0.1, model="fake",
                        input_tokens=100, output_tokens=0)

    async def aclose(self):
        pass


class FailingJev(FakeJev):
    async def score(self, query, document, guidance):
        from rag_jev.provider import ProviderError

        raise ProviderError("deadline_exceeded")


PASSAGES = [
    {"id": "d0", "text": "gold passage about the question topic"},
    {"id": "d1", "text": "unrelated filler passage nothing relevant"},
]


def test_mapping_and_filter_operating_point(monkeypatch):
    monkeypatch.setattr("rag_jev.Jev", FakeJev)
    out = adapter.select_passages("some question?", PASSAGES, api_key="x",
                                  min_relevance=0.5)
    assert out["status"] == "applied"
    recs = {r["doc_id"]: r for r in out["records"]}
    assert recs["d0"]["confidence"] == pytest.approx(0.9)
    assert recs["d0"]["selected"] is True
    assert recs["d0"]["reason"] == "retained"
    assert recs["d1"]["confidence"] == pytest.approx(0.1)
    assert recs["d1"]["selected"] is False
    assert recs["d1"]["reason"] == "below_threshold"
    assert all(r["policy"] == adapter.POLICY_RAG_JEV for r in out["records"])
    assert out["usage"]["input_tokens"] == 200  # summed per-doc Jev cost


def test_bypassed_surfaces_nulls_not_fabrication(monkeypatch):
    monkeypatch.setattr("rag_jev.Jev", FailingJev)
    out = adapter.select_passages("some question?", PASSAGES, api_key="x",
                                  timeout_ms=10000)
    assert out["status"] == "bypassed"
    assert [r["confidence"] for r in out["records"]] == [None, None]
    assert [r["selected"] for r in out["records"]] == [None, None]
    assert all(r["reason"] == "bypassed" for r in out["records"])


def test_default_min_relevance_is_documented_midpoint():
    assert adapter.DEFAULT_MIN_RELEVANCE == 0.5
