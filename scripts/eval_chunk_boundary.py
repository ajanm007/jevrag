"""V1.1 eval: chunk-boundary (Jev one-shot) vs cosine-similarity baseline.

One Wikipedia article, sentence-pair candidates, paragraph breaks as weak
ground truth (author formatting, not semantic truth — stated, not hidden):
every paragraph break is a positive; a seeded sample of within-paragraph
adjacent pairs are negatives.

Both arms see the same 2+2-sentence window. Scores: Jev P(split) vs
1 - cosine(mean(window) embeddings). Both go through the UNCHANGED
calibration harness (AURC, ECE, risk-coverage) — that reuse without edits
is half the finding. Per-round Jev latency/tokens recorded per candidate.

Usage:
    py -3.13 scripts/eval_chunk_boundary.py [--article Photosynthesis] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key  # noqa: E402

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

UA = {"User-Agent": "JevRAG-eval/0.1 (research evaluation)"}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")


def fetch_article(title: str, cache: Path) -> str:
    """Wikipedia plain-text extract, cached on disk (don't refetch)."""
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    import requests

    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": "1",
                "titles": title, "format": "json"},
        headers=UA, timeout=60)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    text = page.get("extract", "")
    if len(text) < 1000:
        raise ValueError(f"article fetch failed for {title!r}")
    cache.write_text(text, encoding="utf-8")
    return text


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(paragraph.strip()) if s.strip()]


def build_candidates(text: str, doc_id: str, window: int = 2, seed: int = 20260920,
                     max_neg: int | None = None) -> list[dict]:
    """All paragraph breaks (label 1) + seeded sample of within-paragraph
    adjacent pairs (label 0). Windows of `window` sentences either side."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    sents_per_para = [split_sentences(p) for p in paragraphs]
    positives, negatives = [], []
    idx = 0
    for pi in range(len(sents_per_para) - 1):
        before = sents_per_para[pi][-window:]
        after = sents_per_para[pi + 1][:window]
        if before and after:
            positives.append({"doc_id": doc_id, "boundary_index": idx,
                              "before": " ".join(before),
                              "after": " ".join(after), "is_split": 1})
            idx += 1
        sents = sents_per_para[pi]
        for si in range(len(sents) - 1):
            b, a = sents[max(0, si - window + 1):si + 1], sents[si + 1:si + 1 + window]
            if b and a:
                negatives.append({"doc_id": doc_id, "boundary_index": idx,
                                  "before": " ".join(b), "after": " ".join(a),
                                  "is_split": 0})
                idx += 1
    rng = random.Random(seed)
    rng.shuffle(negatives)
    if max_neg is not None:
        negatives = negatives[:max_neg]
    cands = positives + negatives
    cands.sort(key=lambda c: c["boundary_index"])
    return cands


def cosine_scores(candidates: list[dict], model) -> list[float]:
    """Split score = (1 - cosine)/2 over mean window embeddings.

    Same window both arms see. The /2 maps cosine's [-1, 1] range onto [0, 1]
    as a fixed, data-independent map — required because the shared harness
    rightly refuses non-probability scores (it caught 1-cos exceeding 1 on
    the first run rather than computing silently). Higher = more split-like,
    same direction as Jev's P(split), so the harness consumes both
    identically. Rank metrics (AURC) are invariant to the choice of monotonic
    map; ECE/Brier are computed on this fixed mapping, stated here.
    """
    import numpy as np

    befores = [" ".join(split_sentences(c["before"][-500:])) or c["before"]
               for c in candidates]
    afters = [c["after"] for c in candidates]
    emb_b = np.asarray(model.encode(befores, normalize_embeddings=True))
    emb_a = np.asarray(model.encode(afters, normalize_embeddings=True))
    cos = (emb_b * emb_a).sum(axis=1)
    return [float((1.0 - c) / 2.0) for c in cos]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", default="Photosynthesis")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total candidates (smoke runs)")
    ap.add_argument("--neg", type=int, default=60,
                    help="within-paragraph negatives sampled (seeded)")
    ap.add_argument("--out", default="outputs/chunk_boundary_wikipedia.jsonl")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    cache = REPO_ROOT / "outputs" / f"wiki_{args.article}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)

    text = fetch_article(args.article, cache)
    cands = build_candidates(text, args.article, max_neg=args.neg)
    if args.limit:
        pos = [c for c in cands if c["is_split"]] [:args.limit // 2]
        neg = [c for c in cands if not c["is_split"]][:args.limit // 2]
        cands = sorted(pos + neg, key=lambda c: c["boundary_index"])
    print(f"article={args.article} candidates={len(cands)} "
          f"(pos={sum(c['is_split'] for c in cands)})")

    jev = JevDecision(api_key=load_jev_key())
    records, _ = decide_document(cands, jev)
    by_idx = {r["boundary_index"]: r for r in records}
    assert len(by_idx) == len(cands), "every candidate asked exactly once"

    print("loading embedder for the cosine baseline...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cos = cosine_scores(cands, model)

    import numpy as np
    jev_conf = np.array([by_idx[c["boundary_index"]]["confidence"] for c in cands])
    correct = np.array([c["is_split"] for c in cands], dtype=float)
    cos_arr = np.array(cos)

    with open(out_path, "w", encoding="utf-8") as f:
        for c, jc, cs in zip(cands, jev_conf, cos):
            f.write(json.dumps({
                "doc_id": c["doc_id"], "boundary_index": c["boundary_index"],
                "is_split": c["is_split"],
                "jev_confidence": float(jc), "cosine_score": float(cs),
                "jev_label": label_for(jc),
                **{k: v for k, v in by_idx[c["boundary_index"]].items()
                   if k in ("latency_ms", "input_tokens", "policy")},
            }, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")

    for name, scores in (("jev", jev_conf), ("cosine", cos_arr)):
        cal = calibration_summary(scores, correct)
        print(f"\n--- {name} ---")
        print(f"  AURC {cal['aurc']:.4f}  (oracle {oracle_aurc(correct):.4f}, "
              f"random {1.0 - correct.mean():.4f})")
        print(f"  ECE {cal['ece']:.4f}  Brier {cal['brier']:.4f} "
              f"(skill {cal['brier_skill']:.4f})")
        print(f"  acc@cov0.8 {accuracy_at_coverage(scores, correct, 0.8):.4f}  "
              f"cov@acc0.8 {coverage_at_accuracy(scores, correct, 0.8):.4f}")
    print(f"\nAURC delta (cosine - jev): {aurc(cos_arr, correct) - aurc(jev_conf, correct):+.4f} "
          "(+ favors jev)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
