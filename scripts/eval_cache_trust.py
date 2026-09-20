"""Eval: cache-trust gate on constructed HotpotQA cache-hit scenarios (first cut).

No (cache hit, was-it-safe) dataset exists, and HotpotQA has no caching
layer — so the scenario is constructed but the LABEL stays real: EM of the
cached answer against the current query's gold.

Per case: a cache entry (query + gold answer from one val question) and a
current query that is either a paraphrase of the same question (safe to
serve; 4o-mini-generated, disclosed) or a genuinely different val question
(unsafe; the cached answer is wrong). Similarity scores are real cosines
from openai/text-embedding-3-small via OpenRouter; ages/scoping are
assigned by design across fresh/stale and scoped/unscoped (disclosed);
lengths are the real char lengths. Fixed scenario choices, not tuned
findings: similarity threshold 0.70, TTL 3600s.

Usage: py -3.13 scripts/eval_cache_trust.py [--n-safe 12] [--n-unsafe 12]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(r"D:\Research\RAG-Gate")))

import requests  # noqa: E402

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402

from jevrag.benchmarks.hotpotqa import exact_match, load_questions  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import calibration_summary  # noqa: E402
from jevrag.primitives.cache_trust import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TTL_SECONDS,
    CacheTrustState,
    check_cache,
)

EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
PARAPHRASE_MODEL = "openai/gpt-4o-mini"
PARAPHRASE_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Fresh questions for this eval — past the val[:30] packets 15/16 used.
VAL_OFFSET = 30

#: Assigned entry ages by design (seconds): half fresh, half near TTL expiry.
FRESH_AGES = [90.0, 150.0, 240.0, 360.0, 480.0, 600.0]
STALE_AGES = [2100.0, 2400.0, 2700.0, 3000.0, 3300.0, 3500.0]

PARAPHRASE_CACHE = REPO_ROOT / "outputs" / "cache_trust_paraphrases.json"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embed_texts(or_key: str, texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {or_key}",
                 "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=(10, 120),
    )
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def paraphrase(or_key: str, question: str) -> str:
    body = {
        "model": PARAPHRASE_MODEL,
        "messages": [{"role": "user", "content": (
            "Rephrase this question preserving its exact meaning. Reply with "
            f"only the rephrased question, one sentence:\n\n{question}")}],
        "temperature": 0.7,
        "max_tokens": 64,
    }
    resp = requests.post(
        PARAPHRASE_URL,
        headers={"Authorization": f"Bearer {or_key}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=(10, 60),
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").strip()


def load_paraphrases(or_key: str, questions: list[dict]) -> dict[str, str]:
    cached: dict[str, str] = {}
    if PARAPHRASE_CACHE.exists():
        cached = json.loads(PARAPHRASE_CACHE.read_text(encoding="utf-8"))
    missing = [q for q in questions if str(q["id"]) not in cached]
    for q in missing:
        cached[str(q["id"])] = paraphrase(or_key, q["question"])
        print(f"  paraphrased {q['id']}: {cached[str(q['id'])]!r}", flush=True)
    PARAPHRASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PARAPHRASE_CACHE.write_text(json.dumps(cached, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    return cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-safe", type=int, default=12)
    ap.add_argument("--n-unsafe", type=int, default=12)
    ap.add_argument("--threshold", type=float,
                    default=DEFAULT_SIMILARITY_THRESHOLD)
    ap.add_argument("--out", default="outputs/cache_trust_val.jsonl")
    args = ap.parse_args()

    need = args.n_safe + args.n_unsafe
    questions = load_questions(split="val")[VAL_OFFSET:VAL_OFFSET + need]
    assert len(questions) == need, "val slice too short"
    safe_qs, unsafe_qs = questions[:args.n_safe], questions[args.n_safe:]

    or_key = load_openrouter_key()
    jev = JevDecision(api_key=load_jev_key())

    # Current queries: paraphrases (safe) + different questions (unsafe).
    # Cache entries: safe cases reuse their own question's gold; unsafe cases
    # pair current question j with cached entry from safe question j % n_safe.
    paraphrases = load_paraphrases(or_key, safe_qs)
    current_texts = ([paraphrases[str(q["id"])] for q in safe_qs]
                     + [q["question"] for q in unsafe_qs])
    cached_texts = ([q["question"] for q in safe_qs]
                    + [safe_qs[i % len(safe_qs)]["question"]
                       for i in range(len(unsafe_qs))])
    vecs = embed_texts(or_key, current_texts + cached_texts)
    n_cur = len(current_texts)
    sims = [cosine(vecs[i], vecs[n_cur + i]) for i in range(n_cur)]

    cases = []
    for i, q in enumerate(safe_qs):
        cases.append({"kind": "safe", "qid": str(q["id"]),
                      "current": paraphrases[str(q["id"])],
                      "gold": q["answer"], "cached_answer": q["answer"],
                      "sim": sims[i]})
    for j, q in enumerate(unsafe_qs):
        src = safe_qs[j % len(safe_qs)]
        cases.append({"kind": "unsafe", "qid": str(q["id"]),
                      "current": q["question"], "gold": q["answer"],
                      "cached_answer": src["answer"],
                      "sim": sims[args.n_safe + j],
                      "cached_from": str(src["id"])})

    ages = (FRESH_AGES + STALE_AGES) * ((len(cases) // 12) + 1)
    rows, t0 = [], time.perf_counter()
    for i, c in enumerate(cases):
        state = CacheTrustState.make(
            similarity_score=c["sim"], similarity_threshold=args.threshold,
            cache_age_seconds=ages[i], cache_ttl_seconds=DEFAULT_TTL_SECONDS,
            entry_scoped_to_conversation=(i % 2 == 0),
            query_length_chars=len(c["current"]),
            answer_length_chars=len(c["cached_answer"]))
        rec, _ = check_cache(query_id=c["qid"], query=c["current"],
                             cached_answer=c["cached_answer"],
                             state=state, decision=jev)
        em = 1 if exact_match(c["cached_answer"], c["gold"]) else 0
        rows.append({**rec, "kind": c["kind"], "correct": em,
                     "similarity": c["sim"], "margin": state.score_margin,
                     "age_s": ages[i],
                     "scoped": state.entry_scoped_to_conversation})
        print(f"[{i + 1}/{len(cases)}] kind={c['kind']} em={em} "
              f"sim={c['sim']:.3f} margin={state.score_margin:+.3f} "
              f"serve_conf={rec['confidence']} action={rec['action']} "
              f"fallback={rec['fallback_used']}", flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import numpy as np
    conf = np.array([np.nan if r["confidence"] is None else r["confidence"]
                     for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=float)
    cal = calibration_summary(conf, corr, n_bins=5)
    served = [r for r in rows if r["action"] == "serve"]
    vetoed = [r for r in rows if r["action"] == "regenerate"]
    print(f"\nn={len(rows)} served={len(served)} vetoed={len(vetoed)} "
          f"fallbacks={sum(r['fallback_used'] for r in rows)}")
    print(f"served EM: {sum(r['correct'] for r in served) / max(len(served), 1):.4f} "
          f"(n={len(served)}) vs overall EM {corr.mean():.4f}")
    print(f"serve-confidence: AURC {cal['aurc']:.4f} (oracle "
          f"{cal['oracle_aurc']:.4f}, random {cal['random_aurc']:.4f}), "
          f"ECE {cal['ece']:.4f}, Brier {cal['brier']:.4f} "
          f"(skill {cal['brier_skill']:.4f}), nan={cal['n_nan_confidence']}")
    print(f"similarity: safe mean "
          f"{np.mean([r['similarity'] for r in rows if r['kind'] == 'safe']):.3f} "
          f"vs unsafe mean "
          f"{np.mean([r['similarity'] for r in rows if r['kind'] == 'unsafe']):.3f} "
          f"(threshold {args.threshold})")
    print(f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
