"""Eval-hardening: cache-trust gate on genuinely near-threshold hits.

The first-pass scenario separated bimodally (paraphrase ~0.93 vs unrelated
~0.13), so its near-perfect numbers measured clean-signal reading, not hard
discrimination. This eval builds the traffic that actually matters:

- IN-BAND EM=1: diluted / instruction-appended variants of real questions
  (similarity empirically 0.73-0.92 in the probe — thin margin, intent and
  gold preserved so the EM label stays legitimately 1).
- IN-BAND EM=0: mined cross-question pairs landing in [0.50, 0.80] plus two
  natural 0.608 pairs found while probing this data (real EM=0).
- SUB-THRESHOLD EM=0: same-template pairs in [0.40, 0.55) (real EM=0).
- STALE-GRID: high-sim paraphrases with stale ages (TTL proximity
  0.83-0.99) + unscoped — EM=1 by label, expected vetoes; measures the
  coverage price of staleness caution across a range, not once.
- CONTROLS: fresh+scoped paraphrases (expect serve).

Labels are EM of the cached answer against the current query's real gold,
same discipline as the first-pass eval. Excluded deliberately:
truncation/entity-only variants — the full question's answer intent does
not survive them, so the EM label would mislead rather than stay real.

Fixed scenario choices (not tuned): threshold 0.70, TTL 3600s, assigned
ages/scoping by design, real embedding cosines, 4o-mini paraphrases
(disclosed, cached). Fresh val slice, disjoint from packets 15/16/17.

Usage: py -3.13 scripts/eval_cache_trust_hard.py [--out ...]
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

VAL_OFFSET = 54
N_BASE = 8       # base questions for in-band variants
N_STALE = 10     # stale-grid paraphrase cases
N_CONTROL = 6    # fresh paraphrase controls
BAND_LO, BAND_HI = 0.50, 0.80   # mined EM=0 band
SUB_LO, SUB_HI = 0.40, 0.55     # sub-threshold template band

PARAPHRASE_CACHE = REPO_ROOT / "outputs" / "cache_trust_hard_paraphrases.json"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_texts(or_key: str, texts: list[str]) -> list[list[float]]:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), 100):
        resp = requests.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {or_key}",
                     "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": texts[i:i + 100]},
            timeout=(10, 120))
        resp.raise_for_status()
        vecs.extend(d["embedding"] for d in resp.json()["data"])
    return vecs


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
        json=body, timeout=(10, 60))
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").strip()


def load_paraphrases(or_key: str, questions: list[dict]) -> dict[str, str]:
    cached: dict[str, str] = {}
    if PARAPHRASE_CACHE.exists():
        cached = json.loads(PARAPHRASE_CACHE.read_text(encoding="utf-8"))
    for q in questions:
        if str(q["id"]) not in cached:
            cached[str(q["id"])] = paraphrase(or_key, q["question"])
            print(f"  paraphrased {q['id']}: {cached[str(q['id'])]!r}",
                  flush=True)
    PARAPHRASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PARAPHRASE_CACHE.write_text(json.dumps(cached, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    return cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float,
                    default=DEFAULT_SIMILARITY_THRESHOLD)
    ap.add_argument("--out", default="outputs/cache_trust_hard_val.jsonl")
    args = ap.parse_args()

    or_key = load_openrouter_key()
    jev = JevDecision(api_key=load_jev_key())
    val = load_questions(split="val")

    base = val[VAL_OFFSET:VAL_OFFSET + N_BASE]
    stale_qs = val[VAL_OFFSET + N_BASE:VAL_OFFSET + N_BASE + N_STALE]
    control_qs = val[VAL_OFFSET + N_BASE + N_STALE:
                     VAL_OFFSET + N_BASE + N_STALE + N_CONTROL]
    pool = val[VAL_OFFSET + N_BASE + N_STALE + N_CONTROL:]
    used_ids = {q["id"] for q in base + stale_qs + control_qs}

    paraphrases = load_paraphrases(or_key, stale_qs + control_qs)

    # --- texts to embed: base fulls + variants + stale/control paraphrases
    # --- + pool (for mining EM=0 pairs)
    texts, kinds = [], []
    for q in base:
        texts.append(q["question"])
        kinds.append(("base", q["id"]))
    variants: list[tuple[str, str, str]] = []  # (text, kind, base_id)
    for q in base:
        variants.append((q["question"] + " Also, what is the capital of France?",
                         "diluted", q["id"]))
        variants.append((q["question"] + " Explain in detail with examples "
                         "and citations.", "instruct", q["id"]))
    for text, kind, bid in variants:
        texts.append(text)
        kinds.append((kind, bid))
    for q in stale_qs + control_qs:
        texts.append(paraphrases[str(q["id"])])
        kinds.append(("paraphrase", q["id"]))
    pool_texts = [q["question"] for q in pool if q["id"] not in used_ids]
    pool_qs = [q for q in pool if q["id"] not in used_ids]
    texts.extend(pool_texts)

    vecs = embed_texts(or_key, texts)
    n_scoped = len(kinds)
    pool_vecs = vecs[n_scoped:]
    by_key = {}
    for (kind, bid), v in zip(kinds, vecs[:n_scoped]):
        by_key.setdefault(bid, {})[kind] = v
    # base fulls are the first N_BASE texts in order
    for q, v in zip(base, vecs[:N_BASE]):
        by_key[q["id"]]["full"] = v

    cases: list[dict] = []

    # A. in-band EM=1 variants (intent+gold preserved).
    for i, (text, kind, bid) in enumerate(variants):
        q = next(q for q in base if q["id"] == bid)
        sim = cosine(by_key[bid][kind], by_key[bid]["full"])
        staleish = (i % 4 == 3)  # every 4th: stale+unscoped interplay
        cases.append({"group": f"inband-{kind}", "qid": bid,
                      "current": text, "gold": q["answer"],
                      "cached_answer": q["answer"], "sim": sim,
                      "age_s": 3200.0 if staleish else 240.0,
                      "scoped": False if staleish else True})

    mined = []
    for text, kind, bid in variants:
        v = by_key[bid][kind]
        for pq, pv in zip(pool_qs, pool_vecs):
            s = cosine(v, pv)
            if BAND_LO <= s < BAND_HI:
                mined.append((s, text, kind, bid, pq))
    mined.sort(key=lambda t: -t[0])
    seen: set[str] = set()
    for s, text, kind, bid, pq in mined:
        if pq["id"] in seen or len([c for c in cases
                                    if c["group"] == "mined"]) >= 8:
            break
        seen.add(pq["id"])
        cases.append({"group": "mined", "qid": pq["id"], "current": text,
                      "gold": pq["answer"],
                      "cached_answer": next(q for q in base
                                            if q["id"] == bid)["answer"],
                      "sim": s, "age_s": 300.0, "scoped": True,
                      "note": f"variant-of-{bid[:8]} vs {pq['id'][:8]}"})

    # C. sub-threshold template pairs [SUB_LO, SUB_HI): pool-vs-pool.
    subs = []
    for i in range(len(pool_qs)):
        for j in range(i + 1, len(pool_qs)):
            s = cosine(pool_vecs[i], pool_vecs[j])
            if SUB_LO <= s < SUB_HI:
                subs.append((s, pool_qs[i], pool_qs[j]))
    subs.sort(key=lambda t: -t[0])
    seen_c: set[str] = set()
    for s, qa, qb in subs:
        if qa["id"] in seen_c or qb["id"] in seen_c or len(
                [c for c in cases if c["group"] == "subthreshold"]) >= 8:
            continue
        seen_c.update((qa["id"], qb["id"]))
        cases.append({"group": "subthreshold", "qid": qb["id"],
                      "current": qb["question"], "gold": qb["answer"],
                      "cached_answer": qa["answer"], "sim": s,
                      "age_s": 300.0, "scoped": True})

    # D. stale grid: paraphrase, EM=1, TTL proximity 0.83-0.99, unscoped.
    stale_ages = [3000.0, 3100.0, 3200.0, 3300.0, 3400.0,
                  3450.0, 3500.0, 3530.0, 3560.0, 3590.0]
    for q, age in zip(stale_qs, stale_ages):
        cases.append({"group": "stale-grid", "qid": q["id"],
                      "current": paraphrases[str(q["id"])], "gold": q["answer"],
                      "cached_answer": q["answer"], "sim": None,
                      "age_s": age, "scoped": False})

    # E. controls: fresh+scoped paraphrases.
    for q in control_qs:
        cases.append({"group": "control", "qid": q["id"],
                      "current": paraphrases[str(q["id"])], "gold": q["answer"],
                      "cached_answer": q["answer"], "sim": None,
                      "age_s": 180.0, "scoped": True})

    # Fill sim=None (stale-grid + controls): embed their full texts now.
    need_full = [q for q in stale_qs + control_qs]
    full_vecs = embed_texts(or_key, [q["question"] for q in need_full])
    full_by_id = {q["id"]: v for q, v in zip(need_full, full_vecs)}
    for c in cases:
        if c["sim"] is None:
            c["sim"] = cosine(by_key[c["qid"]]["paraphrase"],
                              full_by_id[c["qid"]])

    # B2. natural in-band pairs (packet-18 probe: 30,135 val pairs mined,
    # exactly 2 landed in [0.55, 0.85)). All four directed cases are EM=0 —
    # asserted live, not assumed. Fresh+scoped isolates similarity.
    # Disclosure: 5adcfbf4 also appears as a subthreshold current query and
    # 5ab1d7ac as a stale-grid paraphrase — independent gate calls with
    # scenario-specific labels, no carryover.
    vec_by_id = {q["id"]: v for q, v in zip(pool_qs, pool_vecs)}
    vec_by_id.update(full_by_id)
    by_val_id = {q["id"]: q for q in val}
    for cached_id, current_id in [
        ("5a825bb455429940e5e1a874", "5adcfbf45542990d50227d86"),
        ("5adcfbf45542990d50227d86", "5a825bb455429940e5e1a874"),
        ("5ab1d7ac554299449642c7e6", "5ab6369655429953192ad2a1"),
        ("5ab6369655429953192ad2a1", "5ab1d7ac554299449642c7e6"),
    ]:
        cq, qq = by_val_id[cached_id], by_val_id[current_id]
        assert not exact_match(cq["answer"], qq["answer"]), \
            f"natural pair {cached_id[:8]}/{current_id[:8]} is not EM=0"
        s = cosine(vec_by_id[cached_id], vec_by_id[current_id])
        assert BAND_LO <= s < 0.90, f"natural pair sim {s:.3f} out of band"
        cases.append({"group": "natural-band", "qid": qq["id"],
                      "current": qq["question"], "gold": qq["answer"],
                      "cached_answer": cq["answer"], "sim": s,
                      "age_s": 240.0, "scoped": True})

    print(f"groups: " + ", ".join(
        f"{g}={sum(1 for c in cases if c['group'] == g)}"
        for g in sorted({c['group'] for c in cases})), flush=True)

    rows, t0 = [], time.perf_counter()
    for i, c in enumerate(cases, 1):
        state = CacheTrustState.make(
            similarity_score=c["sim"], similarity_threshold=args.threshold,
            cache_age_seconds=c["age_s"],
            cache_ttl_seconds=DEFAULT_TTL_SECONDS,
            entry_scoped_to_conversation=c["scoped"],
            query_length_chars=len(c["current"]),
            answer_length_chars=len(c["cached_answer"]))
        rec, _ = check_cache(query_id=c["qid"], query=c["current"],
                             cached_answer=c["cached_answer"],
                             state=state, decision=jev)
        em = 1 if exact_match(c["cached_answer"], c["gold"]) else 0
        rows.append({**rec, "group": c["group"], "correct": em,
                     "similarity": c["sim"], "margin": state.score_margin,
                     "age_s": c["age_s"], "scoped": c["scoped"],
                     **({"note": c["note"]} if "note" in c else {})})
        print(f"[{i}/{len(cases)}] {c['group']} em={em} "
              f"sim={c['sim']:.3f} margin={state.score_margin:+.3f} "
              f"serve_conf={rec['confidence']} "
              f"stale={rec['staleness_risk']} action={rec['action']} "
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
    near = [r for r in rows if abs(r["similarity"] - args.threshold) <= 0.15]
    nconf = np.array([r["confidence"] for r in near])
    ncorr = np.array([r["correct"] for r in near], dtype=float)
    ncal = calibration_summary(nconf, ncorr, n_bins=5)
    flips = [r for r in rows if r["confidence"] is not None
             and r["confidence"] >= 0.5 and r["action"] == "regenerate"]
    print(f"\nn={len(rows)} served={len(served)} "
          f"fallbacks={sum(r['fallback_used'] for r in rows)}")
    print(f"served EM: {sum(r['correct'] for r in served) / max(len(served), 1):.4f} "
          f"(n={len(served)}) vs overall EM {corr.mean():.4f}")
    print(f"ALL: serve-conf AURC {cal['aurc']:.4f} (oracle "
          f"{cal['oracle_aurc']:.4f}, random {cal['random_aurc']:.4f}), "
          f"ECE {cal['ece']:.4f}, Brier {cal['brier']:.4f} "
          f"(skill {cal['brier_skill']:.4f}), nan={cal['n_nan_confidence']}")
    print(f"NEAR-THRESHOLD (|sim-0.70|<=0.15, n={len(near)}): AURC "
          f"{ncal['aurc']:.4f} (oracle {ncal['oracle_aurc']:.4f}, random "
          f"{ncal['random_aurc']:.4f}), ECE {ncal['ece']:.4f}, Brier skill "
          f"{ncal['brier_skill']:.4f}")
    print(f"staleness flips (serve_conf>=0.5 yet vetoed): {len(flips)}/{len(rows)}; "
          f"of which actually-correct (false vetoes): "
          f"{sum(r['correct'] for r in flips)}")
    print(f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
