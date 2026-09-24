"""Eval: RAGAS faithfulness as the third abstention-gate signal (CLINE_24).

SECURITY (disclosed 2026-09-24): the pinned ragas==0.4.3 has an unpatched
SSRF in its multi-modal faithfulness collections module
(_try_process_local_file/_try_process_url following attacker-controlled
URLs/paths from `retrieved_contexts`). No patched ragas version exists yet;
the vendor did not respond to disclosure. This script passes retrieved
evidence text as `retrieved_contexts` (below) — safe here because that
evidence comes from a fixed local benchmark dataset (HotpotQA), but do NOT
repurpose this script to score live/externally-sourced retrieval content
until ragas ships a fix. See the `[ragas]` extra's comment in pyproject.toml.

Three abstention-gate signals, ONE shared ground truth, ONE shared metric:

    jev      answer-abstain's own confidence (already in the records)
    logprob  mean token logprob of the generation (already in the records)
    ragas    RAGAS v0.4.3 ``Faithfulness`` on (question, prediction,
             re-derived BM25 evidence) — the new leg, live judge calls

Data provenance (verified, not assumed): the ORIGINAL n=30 set the earlier
Jev-vs-logprob head-to-head used — ``outputs/answer_abstain_4omini_val.jsonl``
(gpt-4o-mini generator, since qwen3-8b returns zero logprobs through
OpenRouter; see MODULE_TRACKER 2026-09-20). Recovered and confirmed by
reproducing that comparison's exact AURCs (0.1544 Jev / 0.1624 logprob) from
this file before scoring anything new.

Evidence: the records do not store evidence text, so it is re-derived through
the same deterministic BM25 path ``eval_answer_abstain.py`` used
(``make_retrieve_fn()(question, rounds_used * per_round_k)``), and the
re-derivation is *verified* row-by-row against the stored ``in_evidence``
flag — if that matches 30/30, the re-derived evidence is the evidence the
original run judged against.

RAGAS interface (verified against v0.4.3 source, not memory):
``from ragas.metrics.collections import Faithfulness`` (the current
collections API; ``ragas.metrics.Faithfulness`` is deprecated and warns),
``llm_factory(model, client=AsyncOpenAI(...))``, two async LLM calls per row
(statement extraction + one NLI verdicts call), no embeddings, score in [0,1]
or NaN when no statements could be extracted.

The judge-model confound, disclosed rather than absorbed: RAGAS's own
documented default/example judge is gpt-4o-mini — the same model that
generated these predictions. A default-configured run would therefore be a
self-judge. We run TWO judges and report both:

    openai/gpt-4o-mini       same model as the generator (RAGAS default)
    google/gemini-2.5-flash  different vendor — no self-judge

Cost gate: 2 calls/row/judge (~60-120 small structured calls total); OpenRouter
key spend is snapshotted before/after so real spend is measured, not guessed.

Usage:
  py -3.13 scripts/eval_ragas_faithfulness.py --dry-run          # no LLM calls
  py -3.13 scripts/eval_ragas_faithfulness.py --judge openai/gpt-4o-mini --limit 2
  py -3.13 scripts/eval_ragas_faithfulness.py --compare-only     # AURC + bootstrap
  py -3.13 scripts/eval_ragas_faithfulness.py                    # both judges, full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevrag._rag_gate import rag_gate_root  # noqa: E402

sys.path.insert(0, str(rag_gate_root()))

from scripts._load_run_keys import load_openrouter_key  # noqa: E402
from scripts.produce_records import (  # noqa: E402 — same deterministic evidence path
    make_retrieve_fn,
)

from jevrag.benchmarks.hotpotqa import exact_match, normalize_answer  # noqa: E402
from jevrag.eval.calibration import aurc  # noqa: E402 — the tested implementation

#: Same records the Jev-vs-logprob head-to-head used (recovered original).
RECORDS = REPO_ROOT / "outputs" / "answer_abstain_4omini_val.jsonl"
#: Evidence re-derivation: eval_answer_abstain.py's own convention.
PER_ROUND_K = 5
#: Judge models. The first is RAGAS's default AND the generator's model.
DEFAULT_JUDGES = ("openai/gpt-4o-mini", "google/gemini-2.5-flash")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
#: Significance test: same protocol as every prior paired comparison
#: (paired bootstrap, 10,000 resamples, percentile CI, two-sided p).
N_BOOT = 10_000
BOOT_SEED = 0

OUT_TPL = "outputs/ragas_faithfulness_{judge}.jsonl"


def prediction_in_evidence(prediction: str, evidence: list[dict]) -> bool:
    """Same normalization rule eval_answer_abstain.py recorded with."""
    pred = normalize_answer(prediction or "")
    if not pred:
        return False
    hay = normalize_answer(" ".join(str(d.get("text", "")) for d in evidence))
    return pred in hay


def load_rows() -> list[dict]:
    with open(RECORDS, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def derive_evidence(rows: list[dict], retrieve_fn) -> tuple[list[list[dict]], int]:
    """Re-derive each row's evidence and verify against its stored in_evidence.

    Returns (evidence_per_row, n_matches). A mismatch means the re-derived
    BM25 ranking differs from what the original run judged — which would make
    the RAGAS leg score a different evidence set than Jev saw, so it is a
    hard stop, not a warning to scroll past.
    """
    evidence = []
    matches = 0
    for r in rows:
        ev = list(retrieve_fn(r["question"], int(r["rounds_used"]) * PER_ROUND_K))
        evidence.append(ev)
        if prediction_in_evidence(r["prediction"], ev) == bool(r["in_evidence"]):
            matches += 1
    return evidence, matches


def openrouter_spend() -> float | None:
    """Best-effort OpenRouter key-spend snapshot in USD, for measuring real
    cost before/after — not a billing integration."""
    try:
        import requests

        r = requests.get(
            f"{OPENROUTER_BASE_URL}/auth/key",
            headers={"Authorization": f"Bearer {load_openrouter_key()}"},
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["data"]["usage"])
    except Exception:
        return None


def judge_out_path(judge: str) -> Path:
    return REPO_ROOT / "outputs" / f"ragas_faithfulness_{judge.replace('/', '_')}.jsonl"


def score_judge(judge: str, rows: list[dict], evidence: list[list[dict]],
                limit: int | None, rescore: bool, concurrency: int) -> list[dict] | None:
    """RAGAS faithfulness for every row with one judge model.

    Live scoring is spend, so completed rows are appended to the output file
    as they finish and are never re-bought: a rerun (or a crash mid-run, e.g.
    an OpenRouter 402) resumes with only the missing rows. Concurrency
    defaults to 1 — the account's in-flight credit budget rejects parallel
    requests when the balance is small (verified: concurrency 4 hit 402
    ``in_flight_budget_exhausted``). Lazily imports ragas with an actionable
    error when absent.
    """
    out_path = judge_out_path(judge)
    n = limit or len(rows)
    want = rows[:n]
    if rescore:
        out_path.unlink(missing_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["question_id"]] = rec
    todo = [i for i, r in enumerate(want) if r["question_id"] not in done]
    if not todo:
        print(f"[{judge}] all {n} rows already scored in {out_path.name} — no spend")
        return [done[r["question_id"]] for r in want]
    if done:
        print(f"[{judge}] resuming: {len(done)} rows cached, {len(todo)} to score")

    try:
        import ragas
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "ragas is required for live scoring: pip install 'jevrag[ragas]' "
            f"(or pip install ragas==0.4.3). Original error: {e}"
        )

    print(f"[{judge}] ragas {ragas.__version__} Faithfulness, {len(todo)} row(s), "
          f"concurrency={concurrency} (2 LLM calls/row: statements + NLI)")
    client = AsyncOpenAI(api_key=load_openrouter_key(),
                         base_url=OPENROUTER_BASE_URL)
    metric = Faithfulness(llm=llm_factory(judge, client=client))
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(i: int) -> dict:
        async with sem:
            res = await metric.ascore(
                user_input=rows[i]["question"],
                response=rows[i]["prediction"],
                retrieved_contexts=[str(d.get("text", "")) for d in evidence[i]],
            )
        rec = {"question_id": rows[i]["question_id"], "judge": judge,
               "faithfulness": float(res.value)}
        # Append immediately: crash-safe resume, completed rows are never re-bought.
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    async def _all():
        return await asyncio.gather(*(_one(i) for i in todo))

    results = asyncio.run(_all())
    all_rows = [done[r["question_id"]] for r in want if r["question_id"] in done] + results
    n_nan = sum(1 for r in all_rows if r["faithfulness"] != r["faithfulness"])
    print(f"[{judge}] scored {len(results)} new row(s) -> "
          f"{len(all_rows)} total ({n_nan} NaN) in "
          f"{out_path.relative_to(REPO_ROOT)}")
    return all_rows


def paired_bootstrap(a, b, correct, n_boot: int, seed: int) -> dict:
    """Paired bootstrap on Δ AURC = AURC(a) − AURC(b), same protocol as every
    prior paired comparison in this project: 10,000 paired row resamples,
    percentile 95% CI, two-sided p. `aurc` is the vendored, tested one — the
    only thing resampled is the rows."""
    import numpy as np

    from jevrag.eval.calibration import aurc as _aurc

    a, b, correct = map(np.asarray, (a, b, correct))
    rng = np.random.default_rng(seed)
    n = len(correct)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = _aurc(a[idx], correct[idx]) - _aurc(b[idx], correct[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return {
        "observed_delta": float(_aurc(a, correct) - _aurc(b, correct)),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": float(min(p, 1.0)),
        "n_boot": int(n_boot),
        "n_rows": int(n),
    }


def compare(rows: list[dict], judges: list[str], n_boot: int, seed: int) -> dict:
    """AURC for every available signal + the paired tests. Same arrays for all."""
    import numpy as np

    from jevrag._vendor import selective as _sel
    from jevrag.eval.calibration import aurc as _aurc

    corr = np.array([r["correct"] for r in rows], dtype=float)
    signals = {
        "jev (answer-abstain)": np.array([r["confidence"] for r in rows], dtype=float),
        "logprob (rag-gate)": np.array([r["mean_logprob"] for r in rows], dtype=float),
    }
    for judge in judges:
        path = judge_out_path(judge)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            faith = {}
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    faith[rec["question_id"]] = rec["faithfulness"]
        signals[f"ragas faithfulness [{judge}]"] = np.array(
            [faith[r["question_id"]] for r in rows], dtype=float)

    print("\nAURC on identical ground truth (EM correctness, n=30) — "
          "lower is better; oracle and random as anchors")
    table = {}
    for name, s in signals.items():
        nan = int(np.isnan(s).sum())
        row = {"aurc": float(_aurc(s, corr)), "n_nan": nan}
        table[name] = row
        print(f"  {name:<45s} AURC {row['aurc']:.4f}"
              + (f"  ({nan} NaN score(s))" if nan else ""))
    print(f"  {'oracle / random':<45s} AURC {_sel.oracle_aurc(corr):.4f} / "
          f"{float(1.0 - corr.mean()):.4f}")

    names = list(signals)
    pairs = [(names[0], names[1])] if len(names) >= 2 else []
    for other in names[2:]:
        pairs.append((names[0], other))
        pairs.append((other, names[1]))
    print(f"\npaired bootstrap (Δ = left − right; negative favors left), "
          f"{n_boot} resamples, seed {seed}:")
    tests = {}
    for left, right in pairs:
        mask = ~(np.isnan(signals[left]) | np.isnan(signals[right]))
        if not mask.all():
            print(f"  note: {left} vs {right}: dropped "
                  f"{int((~mask).sum())} row(s) with a NaN score (pairwise)")
        res = paired_bootstrap(signals[left][mask], signals[right][mask],
                               corr[mask], n_boot, seed)
        key = f"{left} vs {right}"
        tests[key] = res
        verdict = "significant" if res["p_two_sided"] < 0.05 else "NOT significant"
        print(f"  {key}: Δ={res['observed_delta']:+.4f} "
              f"CI [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}] "
              f"p={res['p_two_sided']:.4f} — {verdict}")
    return {"aurc": table, "tests": tests,
            "oracle_aurc": float(_sel.oracle_aurc(corr)),
            "random_aurc": float(1.0 - corr.mean()),
            "base_em": float(corr.mean()), "n_rows": int(len(rows))}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="RAGAS faithfulness as a third "
                                             "abstention-gate signal (CLINE_24)")
    ap.add_argument("--records", default=str(RECORDS),
                    help="n=30 JSONL the Jev-vs-logprob head-to-head used")
    ap.add_argument("--judge", action="append", default=None,
                    help=f"judge model id via OpenRouter (repeatable); default: "
                         f"{' and '.join(DEFAULT_JUDGES)}")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first k rows (smoke/cost gate)")
    ap.add_argument("--rescore", action="store_true",
                    help="discard existing per-judge outputs and spend again")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel judge requests (default 1: the account's "
                         "in-flight credit budget rejects parallel requests "
                         "when the balance is small)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify data + evidence re-derivation and print the "
                         "plan/cost estimate; no LLM calls")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip scoring; AURC + bootstrap from existing outputs")
    ap.add_argument("--allow-evidence-mismatch", action="store_true",
                    help="proceed even if re-derived evidence disagrees with the "
                         "stored in_evidence flags (diagnostic use only)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--out", default="outputs/ragas_compare_report.json")
    args = ap.parse_args()

    rows = load_rows() if args.records == str(RECORDS) else [
        json.loads(line) for line in open(args.records, encoding="utf-8")
        if line.strip()
    ]
    n = len(rows)
    print(f"records: {Path(args.records).name}  n={n}  base EM "
          f"{sum(r['correct'] for r in rows) / n:.4f}  "
          f"generator {rows[0].get('generator', '?')}")

    # Ground truth provenance: stored correct must equal EM(prediction, gold).
    em_ok = sum(int(exact_match(r["prediction"], r["gold"])) == r["correct"]
                for r in rows)
    print(f"ground-truth check: stored correct == EM(prediction, gold) on "
          f"{em_ok}/{n}")

    retrieve_fn = make_retrieve_fn()
    evidence, ev_matches = derive_evidence(rows, retrieve_fn)
    print(f"evidence re-derivation: in_evidence flag matches on {ev_matches}/{n} "
          f"rows (rounds_used x {PER_ROUND_K} BM25 chunks)")
    if ev_matches < n and not args.allow_evidence_mismatch:
        print("ABORT: re-derived evidence disagrees with the original run's own "
              "flag — the RAGAS leg would judge different evidence than Jev saw. "
              "Investigate before spending anything.", file=sys.stderr)
        return 2

    judges = args.judge or list(DEFAULT_JUDGES)
    if args.dry_run:
        calls = 2 * (args.limit or n) * len(judges)
        print(f"\nDRY RUN — no LLM calls made. Would score {len(judges)} judge(s) "
              f"x {args.limit or n} rows = {calls} structured calls "
              f"(statement extraction + NLI per row).")
        for j in judges:
            print(f"  judge {j}: "
                  + ("cached, no spend" if judge_out_path(j).exists()
                     else "live calls via OpenRouter"))
        return 0

    spend_before = openrouter_spend()
    scored = list(judges)
    if not args.compare_only:
        t0 = time.perf_counter()
        for judge in judges:
            score_judge(judge, rows, evidence, args.limit, args.rescore,
                        args.concurrency)
        print(f"scoring wall {(time.perf_counter() - t0):.0f}s")
    spend_after = openrouter_spend()
    if spend_before is not None and spend_after is not None:
        print(f"OpenRouter spend delta this run: "
              f"${spend_after - spend_before:.6f} (key total ${spend_after:.6f})")
    else:
        print("OpenRouter spend snapshot unavailable — cost not measured")

    if args.limit:
        print(f"--limit {args.limit}: comparison runs on the full file only "
              "once all rows are scored; skipping analysis this run")
        return 0

    report = compare(rows, scored, args.n_boot, args.seed)
    report["records"] = Path(args.records).name
    report["judges"] = judges
    report["evidence_matches"] = f"{ev_matches}/{n}"
    report["spend_delta_usd"] = (None if spend_before is None or spend_after is None
                                 else round(spend_after - spend_before, 6))
    out = REPO_ROOT / args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



