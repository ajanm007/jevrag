"""jevrag.benchmarks.docbench — DocBench PDF benchmark wiring (second dataset).

DocBench (github.com/Anni-Zou/DocBench, arXiv 2407.10701): 229 real PDFs
across five domains with 1,102 QA pairs. Structurally different from HotpotQA:
long-form sentence gold answers (not short spans), PDF sources (no prebuilt
index), skewed question types.

⚠️  LICENSE DISCLOSURE: DocBench's licensing is unclarified — no LICENSE
file exists in its repository and an open issue asking the authors is
unanswered as of 2026-09-20. Used here for research/evaluation purposes only.
Repeat this disclosure wherever DocBench results surface.

Scope: ONE document end-to-end first (folder 0: an ACL paper + 7 questions,
1 text-only). Do not scale to the other 228 until this chain is confirmed.
Only ``text-only`` questions are in scope for scored eval — ``multimodal-t``
needs table/figure parsing and ``meta-data`` needs page-lookup tooling, both
real projects of their own, not filters to widen silently.

The scored-eval contract this module feeds: chunks shaped as HotpotQA-style
``{title, text}`` dicts, a cumulative top-n ``retrieve_fn``, and question
dicts with ``id``/``question``/``answer``/``type`` so the shared sufficiency
loop runs unchanged. EM is NOT meaningful here (long-form golds) — score
with ``llm_judge`` + F1, never EM alone.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

#: Only in-scope type for the scored eval this packet targets.
TEXT_ONLY = "text-only"

LICENSE_NOTE = (
    "DocBench's licensing is unclarified — no LICENSE file exists in its "
    "repository and an open issue asking the authors is unanswered as of "
    "2026-09-20. Used here for research/evaluation purposes only."
)


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract a PDF to per-page text with ``pypdf``.

    Returns ``[{"page": 1-based number, "text": str}]`` for non-empty pages
    only. Page numbers are kept because DocBench's own meta-data questions
    reference them — cheap to preserve now, expensive to reconstruct later.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i, "text": text})
    if not pages:
        raise ValueError(f"no extractable text in {pdf_path}")
    return pages


def chunk_pages(pages: list[dict[str, Any]], chunk_chars: int = 1200,
                overlap_chars: int = 200) -> list[dict[str, Any]]:
    """Fixed-size chunks with overlap over page-concatenated text.

    Returns HotpotQA-shaped ``{title, text, page}`` dicts, where ``page`` is
    the 1-based page the chunk starts on and ``title`` names the span
    (``p{start}`` or ``p{start}-{end}``). Chunk boundaries are positional, not
    semantic — deliberately dumb at this scale; V1.1's chunk-boundary work is
    what would make them smart.
    """
    if chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("need 0 <= overlap < chunk_chars")
    full, page_of_offset = [], []
    for p in pages:
        for ch in p["text"]:
            full.append(ch)
            page_of_offset.append(p["page"])
        for ch in "\n\n":
            full.append(ch)
            page_of_offset.append(p["page"])
    text = "".join(full)
    page_starts = [i for i, pg in enumerate(page_of_offset)
                   if i == 0 or pg != page_of_offset[i - 1]]

    def page_at(offset: int) -> int:
        return page_of_offset[min(offset, len(page_of_offset) - 1)]

    def page_end(offset: int) -> int:
        idx = bisect_right(page_starts, min(offset, len(text) - 1)) - 1
        return page_of_offset[page_starts[max(idx, 0)]]

    chunks, start, step = [], 0, chunk_chars - overlap_chars
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        body = text[start:end].strip()
        if body:
            p0, p1 = page_at(start), page_end(end - 1)
            chunks.append({
                "title": f"p{p0}" if p0 == p1 else f"p{p0}-{p1}",
                "text": body,
                "page": p0,
            })
        if end == len(text):
            break
        start += step
    return chunks


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop 1-char tokens.

    Same logic as RAG-Gate's ``bm25_retriever.tokenize_for_bm25`` (not
    reimplemented from scratch — same function, cited). BM25 math itself is
    ``rank_bm25``'s, untouched.
    """
    import re

    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def build_bm25_index(documents: list[dict[str, Any]]):
    """BM25Okapi over chunk ``text`` fields. Fresh per document — no prebuilt
    DocBench index exists; this is the real new retrieval work of the packet."""
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize_for_bm25(d["text"]) for d in documents]
    return BM25Okapi(tokenized)


def bm25_retrieve(query: str, bm25_index, documents: list[dict[str, Any]],
                  top_k: int = 5) -> dict[str, Any]:
    """Top-k chunks by BM25 score — same return shape as RAG-Gate's retriever
    (``chunks`` with ``bm25_score`` added, ``score_top1``, ``scores_topk``)."""
    scores = bm25_index.get_scores(tokenize_for_bm25(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    chunks, topk = [], []
    for idx in top:
        doc = dict(documents[idx])
        doc["bm25_score"] = float(scores[idx])
        chunks.append(doc)
        topk.append(float(scores[idx]))
    return {"chunks": chunks, "score_top1": topk[0] if topk else 0.0,
            "scores_topk": topk}


def default_k_for_doc(n_chunks: int, min_k: int = 5, divisor: int = 4,
                      max_k: int = 20) -> int:
    """Retrieval depth scaled to document size, with a ceiling.

    ``min(max_k, max(min_k, n_chunks // divisor))``. The floor/divisor are a
    heuristic calibrated on one 41-chunk doc (gold span at rank ~6-15, k=10
    fixed it) — re-validate per domain. The ceiling is budget-derived, not
    tuned: round 3 retrieves 3k chunks into the Jev state (~1200 chars each),
    so k=20 caps the worst case at ~72k chars ≈ 18k tokens, inside Jev's 32k
    state budget. Hit live on first contact with 463- and 993-chunk PDFs
    (uncapped formula gave k=115/248 — Jev calls that size are oversized and
    fail). Recall-vs-budget on long docs is now the open scaling question
    (more rounds? denser chunks? two-stage retrieve?) — for morning, not
    tonight.
    """
    if n_chunks < 1:
        raise ValueError("need at least one chunk")
    return min(max_k, max(min_k, n_chunks // divisor))


def make_retrieve_fn(bm25_index, documents: list[dict[str, Any]]):
    """Cumulative top-n ``retrieve_fn`` closing over one document's index —
    the shape the shared sufficiency loop consumes."""

    def retrieve_fn(question: str, n_docs: int) -> list[dict[str, Any]]:
        return bm25_retrieve(question, bm25_index, documents, top_k=n_docs)["chunks"]

    return retrieve_fn


def build_doc_index(pdf_path: str | Path, chunk_chars: int = 1200,
                    overlap_chars: int = 200
                    ) -> tuple[list[dict[str, Any]], Any]:
    """One call: extract → chunk → index. Returns (chunks, bm25_index)."""
    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_chars=chunk_chars,
                         overlap_chars=overlap_chars)
    return chunks, build_bm25_index(chunks)


def load_docbench_questions(doc_id: str, qa_path: str | Path,
                            types: tuple[str, ...] = (TEXT_ONLY,)
                            ) -> list[dict[str, Any]]:
    """Load one document's QA file, filtered to the in-scope types.

    Returns HotpotQA-shaped dicts (``id`` synthesized as ``{doc_id}:{line}``
    since DocBench ships none, plus ``question``/``answer``/``type``/
    ``evidence``) so the shared loop and harness consume them unchanged.
    """
    questions = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if q.get("type") not in types:
                continue
            questions.append({
                "id": f"{doc_id}:{lineno}",
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "type": q.get("type", "unknown"),
                "evidence": q.get("evidence", ""),
            })
    return questions
