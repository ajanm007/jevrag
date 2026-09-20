"""Small live eval of the packet-19 independent cache-safety design.

Generates a seeded synthetic scenario of cache hits — latent safety truth
first, noisy structural signals conditioned on it (so there is irreducible
error and no deterministic rule Jev can just invert), then asks the real
JevDecision backend per hit. Reuses the unchanged calibration harness.

    py -3.13 scripts/eval_cache_safety_check.py [--n 40] [--seed 20260920]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key  # noqa: E402

from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import calibration_summary  # noqa: E402
from jevrag.primitives.cache_safety_check import (  # noqa: E402
    DEFAULT_RISK_CEILING,
    DEFAULT_THRESHOLD,
    CacheHitState,
    decide_hit,
    decide_hit_single,
    make_safety_record,
    verdict_for,
)

SEED = 20260920


def generate_scenario(n: int, seed: int = SEED) -> list[dict]:
    """Seeded cache hits with latent ground truth.

    Generation order matters: the safety label is drawn FIRST, then the
    signals are sampled conditioned on it with real overlap — weak-but-safe
    hits and strong-but-stale ones exist, because they exist in production.
    ``prior_successful_serves`` is 0 for a share of safe hits (new entries),
    so "untested" is not a giveaway for unsafe.
    """
    rng = random.Random(seed)
    hits = []
    for i in range(n):
        safe = rng.random() < 0.55  # ~55% base rate: most hits are fine
        bar = 0.85
        if safe:
            sim = min(0.99, rng.gauss(0.93, 0.035))
            if rng.random() < 0.18:  # safe but near-threshold
                sim = rng.uniform(bar + 0.005, bar + 0.04)
            age_frac = min(1.0, abs(rng.gauss(0.25, 0.2)))
            tested = rng.random() < 0.75
        else:
            cause = rng.random()
            if cause < 0.45:  # weak match
                sim = rng.uniform(bar - 0.06, bar + 0.02)
                age_frac = min(1.0, abs(rng.gauss(0.3, 0.2)))
            elif cause < 0.75:  # stale entry, decent match
                sim = min(0.99, rng.gauss(0.90, 0.03))
                age_frac = rng.uniform(0.85, 1.6)
            else:  # strong, fresh — but the cached answer is wrong anyway
                sim = min(0.99, rng.gauss(0.93, 0.03))
                age_frac = min(1.0, abs(rng.gauss(0.2, 0.15)))
            tested = rng.random() < 0.45
        prior = rng.randint(1, 40) if tested else 0
        cost = rng.choice([500, 1500, 3000, 8000, 20000])
        hits.append({
            "hit_id": f"h{i:03d}",
            "state": CacheHitState(
                match_similarity=round(max(0.0, sim), 4),
                serve_threshold=bar,
                entry_age_hours=round(age_frac * 24.0, 2),
                ttl_hours=24.0,
                regeneration_cost_tokens=cost,
                prior_successful_serves=prior,
                hit_id=f"h{i:03d}"),
            "actually_safe": safe,
        })
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="outputs/cache_safety_check.jsonl")
    args = ap.parse_args()

    hits = generate_scenario(args.n, args.seed)
    base = sum(h["actually_safe"] for h in hits) / len(hits)
    print(f"scenario: n={len(hits)} seed={args.seed} base_rate={base:.3f}",
          flush=True)

    jev = JevDecision(api_key=load_jev_key())
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        t0 = time.perf_counter()
        for i, h in enumerate(hits, 1):
            res = decide_hit(h["state"], jev)
            conf = res.confidence
            verdict = verdict_for(conf["safe_to_serve"],
                                  conf["mismatch_risk"],
                                  threshold=DEFAULT_THRESHOLD,
                                  risk_ceiling=DEFAULT_RISK_CEILING)
            # Single-question ablation, same backend, second call.
            single = decide_hit_single(h["state"], jev)
            rec = make_safety_record(
                hit_id=h["hit_id"], safe_confidence=conf["safe_to_serve"],
                risk_confidence=conf["mismatch_risk"], verdict=verdict,
                latency_ms=float(res.metadata.get("latency_ms", 0.0)),
                input_tokens=int(res.metadata.get("input_tokens", 0)))
            rec["actually_safe"] = bool(h["actually_safe"])
            rec["single_confidence"] = float(single.confidence)
            rec["single_verdict"] = ("serve" if float(single.confidence)
                                     >= DEFAULT_THRESHOLD else "regenerate")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(hits):
                print(f"  [{i}/{len(hits)}] done", flush=True)
    jev.close()
    wall = time.perf_counter() - t0
    return summarize(out_path, wall, len(hits))


def summarize(out_path: Path, wall: float, n: int) -> int:
    rows = [json.loads(l) for l in
            out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    correct = [int(r["actually_safe"]) for r in rows]
    cal_two = calibration_summary([r["safe_confidence"] for r in rows], correct)
    cal_one = calibration_summary([r["single_confidence"] for r in rows],
                                  correct)

    def acc(key: str) -> float:
        return sum(1 for r in rows
                   if (r[key] == "serve") == bool(r["actually_safe"])) / len(rows)

    print(f"\n--- two-question arm (n={len(rows)}, wall {wall:.0f}s) ---")
    print(f"  verdict accuracy {acc('verdict'):.3f}")
    print(f"  safe_conf ECE {cal_two['ece']:.4f} Brier skill "
          f"{cal_two['brier_skill']:+.4f} AURC {cal_two['aurc']:.4f}")
    print("--- single-question ablation ---")
    print(f"  verdict accuracy {acc('single_verdict'):.3f}")
    print(f"  conf ECE {cal_one['ece']:.4f} Brier skill "
          f"{cal_one['brier_skill']:+.4f} AURC {cal_one['aurc']:.4f}")
    tok = sum(r["input_tokens"] for r in rows)
    print(f"tokens {tok * 2} (both arms) est ${tok * 2 / 1e6 * 0.042:.4f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
