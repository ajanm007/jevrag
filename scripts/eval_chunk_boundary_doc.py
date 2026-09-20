"""Packet 22 retest: chunk-boundary vs REAL section-header boundaries.

Packet 20's ground truth was invalid (page breaks mistaken for paragraph
breaks — both arms ≈ noise). This script corrects it on the same document
(P19-1598.pdf): positives are manually-verified section boundaries, not
whatever `\\n\\\\n` happens to survive pypdf.

Ground-truth method (packet option 1+3 combined, disclosed):
- A numbered-header regex (`^\\d+(\\.\\d+)*\\s+[A-Z]`) PROPOSES candidates
  on the extracted line structure.
- Every positive below was VERIFIED by a human reading of the PDF, and the
  regex's misses/false alarms measured against that reading: 2 false
  positives ('21 April 1989' table date, '3 This is not a failure...'
  footnote) removed with stated reasons; unnumbered subsections,
  Acknowledgements, References, Abstract added by hand (the regex cannot
  see unnumbered headers — its real recall gap, reported not hidden).
- Matching is on whitespace-collapsed lowercase text (robust to pypdf
  spacing artifacts like 'WikiT ext-2' / 'T raining'), verified unique per
  header.

Design decisions, stated: positives are section-level only (numbered
sections/subsections + Abstract + Acknowledgements + References = 15
breaks); unnumbered subsections excluded by design (disclosed); header
lines dropped from section bodies (cleaner windows); References body
excluded from negative sampling (ref-to-ref pairs are topic-disjoint
without being section breaks — pure label noise); front matter kept as
section 0 so the Abstract break has a (noisy, affiliation-blob) before-
window, exactly the kind of window a real pipeline faces.

Negatives: seeded (20260920) sample of within-section adjacent sentence
pairs, cap 50 — same discipline as the Wikipedia eval. Cosine baseline
identical to the earlier page-break run (MiniLM, (1-cos)/2 fixed map).

Usage: py -3.13 scripts/eval_chunk_boundary_doc.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key  # noqa: E402

from jevrag.benchmarks.docbench import LICENSE_NOTE, extract_pdf_pages  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import (  # noqa: E402 — reuse, unchanged
    accuracy_at_coverage,
    aurc,
    calibration_summary,
    coverage_at_accuracy,
    oracle_aurc,
)
from jevrag.primitives.chunk_boundary import (  # noqa: E402
    decide_document,
    label_for,
)
from scripts.eval_chunk_boundary import (  # noqa: E402 — windows/baseline
    SENT_SPLIT,
    cosine_scores,
)

DOC_PDF = Path(r"D:\JevRAG-kaggle\docbench-input\0\P19-1598.pdf")
DOC_ID = "doc0-p19-1598"
OUT_PATH = REPO_ROOT / "outputs" / "doc22_chunk_sections.jsonl"
SEED = 20260920
MAX_NEG = 50
WINDOW = 2

#: Section headers in document order, clean strings. Matching normalizes
#: (lowercase, whitespace-collapsed) on both sides. 'FRONT-MATTER' is the
#: title/author block (kept so the Abstract break has a before-window).
SECTIONS = [
    "FRONT-MATTER",
    "Abstract",
    "1 Introduction",
    "2 Knowledge Graph Language Model",
    "2.1 Problem Setup and Notation",
    "2.2 Generative KG Language Model",
    "2.3 Parameterizing the Distributions",
    "3 Linked WikiText-2",
    "4 Training and Inference for KGLM",
    "5 Experiments",
    "5.1 Evaluation Setup",
    "5.2 Results",
    "6 Related Work",
    "7 Conclusions and Future Work",
    "Acknowledgements",
    "References",
]

#: Sections whose bodies are excluded from negative sampling (see docstring).
NO_NEG_SECTIONS = {"References"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_flat(text: str) -> str:
    """Lowercase with ALL whitespace removed — robust to pypdf intra-word
    spacing artifacts ('WikiT ext-2', 'T raining'). Strict full-line
    equality keeps this from over-matching body text."""
    return re.sub(r"\s+", "", text).strip().lower()


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(paragraph.strip()) if s.strip()]


def segment_pages(pages: list[dict]) -> list[tuple[str, str]]:
    """Split page-line text into (section-name, body) at verified headers.

    Header lines (matched normalized, in order, each exactly once) start
    their section; everything before the first header is FRONT-MATTER.
    Bodies have header lines removed and internal newlines collapsed.
    """
    lines = [ln for p in pages for ln in p["text"].split("\n")]
    norm_headers = [normalize_flat(h) for h in SECTIONS if h != "FRONT-MATTER"]
    found: dict[str, int] = {}
    for i, ln in enumerate(lines):
        n = normalize_flat(ln)
        if not n:
            continue
        for h in norm_headers:
            if h not in found and (n == h or (n.startswith(h) and
                                             not n[len(h)].isalnum())):
                # 'startswith(h + " ")' covers run-in headers ('Differences
                # from WikiText-2 Although our...'); the header portion
                # still marks the break. Verified unique below.
                found[h] = i
                break
    missing = [h for h in norm_headers if h not in found]
    if missing:
        raise ValueError(f"headers not found in extraction: {missing}")
    order = [norm_headers.index(normalize(h)) for h in found]
    if sorted(order) != order:
        raise ValueError("headers found out of document order")
    # Uniqueness: each header line index used once.
    if len(set(found.values())) != len(found):
        raise ValueError("two headers matched the same line")
    bounds = sorted(found.values())
    sections: list[tuple[str, str]] = []
    # front matter: lines before the first header.
    sections.append(("FRONT-MATTER",
                     " ".join(l for l in lines[:bounds[0]] if l.strip())))
    idx_of = {v: k for k, v in found.items()}
    for bi, b in enumerate(bounds):
        end = bounds[bi + 1] if bi + 1 < len(bounds) else len(lines)
        name = next(h for h in SECTIONS if normalize_flat(h) == idx_of[b])
        body_lines = [l for l in lines[b + 1:end] if l.strip()]
        sections.append((name, " ".join(body_lines)))
    return sections


def build_candidates(sections: list[tuple[str, str]]) -> list[dict]:
    """Section breaks (label 1) + seeded within-section pairs (label 0)."""
    sents = [(name, split_sentences(body)) for name, body in sections]
    positives, negatives, idx = [], [], 0
    for si in range(len(sents) - 1):
        before = sents[si][1][-WINDOW:]
        after = sents[si + 1][1][:WINDOW]
        if before and after:
            positives.append({"doc_id": DOC_ID, "boundary_index": idx,
                              "before": " ".join(before),
                              "after": " ".join(after), "is_split": 1,
                              "break": f"{sents[si][0]} → {sents[si+1][0]}"})
            idx += 1
        name, ss = sents[si]
        if name in NO_NEG_SECTIONS:
            continue
        for j in range(len(ss) - 1):
            b = ss[max(0, j - WINDOW + 1):j + 1]
            a = ss[j + 1:j + 1 + WINDOW]
            if b and a:
                negatives.append({"doc_id": DOC_ID, "boundary_index": idx,
                                  "before": " ".join(b),
                                  "after": " ".join(a), "is_split": 0,
                                  "break": f"within {name}"})
                idx += 1
    rng = random.Random(SEED)
    rng.shuffle(negatives)
    cands = positives + negatives[:MAX_NEG]
    cands.sort(key=lambda c: c["boundary_index"])
    return cands


def main() -> int:
    print(f"license: {LICENSE_NOTE}", flush=True)
    pages = extract_pdf_pages(DOC_PDF)
    sections = segment_pages(pages)
    print("sections: " + " | ".join(n for n, _ in sections), flush=True)
    cands = build_candidates(sections)
    n_pos = sum(c["is_split"] for c in cands)
    print(f"candidates={len(cands)} (pos={n_pos})", flush=True)

    jev = JevDecision(api_key=load_jev_key())
    t0 = time.perf_counter()
    records, _ = decide_document(cands, jev)
    wall = time.perf_counter() - t0
    by_idx = {r["boundary_index"]: r for r in records}
    assert len(by_idx) == len(cands), "every candidate asked exactly once"

    print("loading cosine baseline (MiniLM)...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cos = cosine_scores(
        [{"before": c["before"], "after": c["after"]} for c in cands], model)

    import numpy as np
    jev_conf = np.array([by_idx[c["boundary_index"]]["confidence"]
                         for c in cands])
    correct = np.array([c["is_split"] for c in cands], dtype=float)
    cos_arr = np.array(cos)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c, jc, cs in zip(cands, jev_conf, cos):
            f.write(json.dumps({
                "doc_id": c["doc_id"], "boundary_index": c["boundary_index"],
                "is_split": c["is_split"], "break": c.get("break", ""),
                "jev_confidence": float(jc), "cosine_score": float(cs),
                "jev_label": label_for(jc),
                **{k: v for k, v in by_idx[c["boundary_index"]].items()
                   if k in ("latency_ms", "input_tokens", "policy")},
            }, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH} in {wall:.0f}s", flush=True)

    for name, scores in (("jev", jev_conf), ("cosine", cos_arr)):
        cal = calibration_summary(scores, correct)
        print(f"--- {name} ---", flush=True)
        print(f"  AURC {cal['aurc']:.4f}  (oracle {oracle_aurc(correct):.4f}, "
              f"random {1.0 - correct.mean():.4f})", flush=True)
        print(f"  ECE {cal['ece']:.4f}  Brier {cal['brier']:.4f} "
              f"(skill {cal['brier_skill']:.4f})", flush=True)
        print(f"  acc@cov0.8 {accuracy_at_coverage(scores, correct, 0.8):.4f}  "
              f"cov@acc0.8 {coverage_at_accuracy(scores, correct, 0.8):.4f}",
              flush=True)
    print(f"AURC delta (cosine - jev): "
          f"{aurc(cos_arr, correct) - aurc(jev_conf, correct):+.4f} "
          "(+ favors jev)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
