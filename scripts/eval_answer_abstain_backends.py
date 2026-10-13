"""answer-abstain at n=300: Jev vs logprob on identical items (packet V03-B1).

The n=30 head-to-head was declared "essentially a tie" (AURC delta -0.0080,
p=0.887) with a CI more than 10x wider than the effect. n=30 cannot separate
the two signals. This runs the same experiment at n=300 on the val split.

Method (unchanged from the n=30 script, so the two runs are comparable):
  1. Sufficiency loop per question (Jev gate) with the 4o-mini generator,
     capturing per-token logprobs off the same completion.
  2. The stopping round's evidence is re-derived deterministically via BM25.
  3. The SAME prediction + evidence is scored twice: (a) JevDecision through
     the existing answer_abstain primitive, (b) LogprobDecision (zero cost).
     Generator identity is held constant within the comparison, which is what
     makes it valid.

Two label proxies are kept separate and never merged, exactly as the n=30
script does: EM correctness (the harness label) and the failure-mode split
(wrong prediction whose content is absent from the evidence).

ORIENTATION CONVENTION FOR THE FAILURE-MODE PROXY -- read before comparing
anyone else's numbers to this output. This proxy is scored with the
"ungrounded" event as the POSITIVE class, i.e. the label array is

    positive = 1.0 if (wrong AND content absent from evidence) else 0.0

which is the INVERSE of the EM-correctness label used for proxy 1 (where
correct = 1.0). A signal scores well here only if it assigns HIGH confidence to
the ungrounded cases -- the opposite polarity from proxy 1, where high
confidence should mark a CORRECT answer. Scoring this proxy with
`correct` semantics instead will not reproduce these numbers and will look
plausible while being wrong. Concretely at n=300 (base rate 0.1833):

    oracle AURC 0.3335 | random AURC 0.8167 | Jev 0.9053 | logprob 0.8625

Both backends land ABOVE the random line, i.e. both are worse than an
uninformative signal at ranking ungrounded-ness. Reported as such; the anchors
are printed alongside the AURCs so the claim is checkable at a glance rather
than asserted in prose.

No tuning: both backends run at their existing defaults.

Usage:
  py -3.13 scripts/eval_answer_abstain_backends.py --n 300
  py -3.13 scripts/eval_answer_abstain_backends.py --compare-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevrag._rag_gate import rag_gate_root  # noqa: E402

sys.path.insert(0, str(rag_gate_root()))

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402
from scripts.produce_records import (  # noqa: E402 -- reuse, not duplication
    GENERATOR_PER_DOC_CHARS,
    GENERATOR_TOTAL_BUDGET_CHARS,
    OPENROUTER_BASE_URL,
    PROMPT_TEMPLATE_FILE,
    build_generator_context,
    make_retrieve_fn,
)
from scripts.eval_answer_abstain import prediction_in_evidence  # noqa: E402

from jevrag.benchmarks.hotpotqa import exact_match, load_questions  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.primitives.answer_abstain import (  # noqa: E402
    DEFAULT_ABSTAIN_THRESHOLD,
    GroundingState,
    action_for,
    check_answer,
    grounding_question,
)
from jevrag.primitives.sufficiency import (  # noqa: E402
    DEFAULT_MAX_ROUNDS,
    DEFAULT_THRESHOLD,
    run_sufficiency_with_trace,
)

#: 4o-mini via OpenRouter -- the same generator the n=30 head-to-head used, so
#: the two runs share generator identity. The generator is held constant across
#: both backends (they score the *same* generations); it is not a variable here.
#: OpenRouter model id is the bare id; GENERATOR_ID is the full provenance tag
#: the records carry. (The n=30 records stored the bare id in `generator`; this
#: run stores the prefixed tag it actually called through, which is more precise.
#: Stated here so the field isn't read as a same-value join key across runs.)
GENERATOR_MODEL = "openai/gpt-4o-mini"
GENERATOR_ID = "openrouter:openai/gpt-4o-mini"

#: Same significance protocol as every prior paired comparison in this project.
N_BOOT = 10_000
BOOT_SEED = 0


def paired_bootstrap(a, b, correct, n_boot: int, seed: int) -> dict:
    """Re-exported from `scripts.eval_ragas_faithfulness` -- the project's
    established protocol (10,000 paired row resamples, percentile 95% CI,
    two-sided p; AURC is the vendored function, only rows are resampled).
    Reused rather than reimplemented so p-values stay comparable with every
    comparison already on record.
    """
    from scripts.eval_ragas_faithfulness import paired_bootstrap as _pb

    return _pb(a, b, correct, n_boot, seed)


def make_logprob_answer_fn(client, model: str, template: str,
                           per_doc_chars: int = GENERATOR_PER_DOC_CHARS,
                           total_budget_chars: int = GENERATOR_TOTAL_BUDGET_CHARS):
    """Same prompt and context builder as `produce_records.make_answer_fn`,
    plus `logprobs=True`.

    Returns `(answer, usage)` so the sufficiency loop's token accounting is
    untouched, and stashes the mean logprob in a closure side-channel
    (`answer_fn.LAST_LOGPROB`). The side channel exists because the loop's
    `answer_fn` contract is `(prediction, usage)` and the primitives are not
    editable under this packet -- stated rather than worked around by editing
    one.
    """
    last: dict = {}

    def answer_fn(question: str, evidence: list[dict]):
        context = build_generator_context(
            evidence, per_doc_chars=per_doc_chars,
            total_budget_chars=total_budget_chars,
        )
        prompt = template.replace("{context}", context).replace("{question}", question)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=32,
            logprobs=True,
            top_logprobs=1,
        )
        choice = resp.choices[0]
        answer = (choice.message.content or "").strip()
        tokens: list[float] = []
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        for entry in (content or []):
            value = getattr(entry, "logprob", None)
            if value is not None:
                tokens.append(float(value))
        last["mean_logprob"] = (
            float(sum(tokens) / len(tokens)) if tokens else float("nan"))
        last["n_logprob_tokens"] = len(tokens)
        usage = getattr(resp, "usage", None)
        return answer, {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }

    answer_fn.LAST_LOGPROB = last  # type: ignore[attr-defined]
    return answer_fn


def run_one(q, jev, logprob_backend, retrieve_fn, answer_fn,
            per_round_k: int, threshold: float) -> tuple[dict, dict]:
    """One question: sufficiency loop, then the abstain check for both backends.

    Both backends see the same prediction and the same evidence by
    construction; only the confidence signal differs.
    """
    qid, question, gold = str(q["id"]), q["question"], q["answer"]
    rec, _trace = run_sufficiency_with_trace(
        question_id=qid, question=question, gold=gold,
        qtype=q.get("type", "unknown"), decision=jev,
        retrieve_fn=retrieve_fn, answer_fn=answer_fn,
        threshold=DEFAULT_THRESHOLD, max_rounds=DEFAULT_MAX_ROUNDS,
        per_round_k=per_round_k)
    evidence = list(retrieve_fn(question, rec["rounds_used"] * per_round_k))
    arec, _ = check_answer(
        question_id=qid, question=question, gold=gold, evidence=evidence,
        prediction=rec["prediction"], decision=jev, threshold=threshold)

    state = GroundingState(question=question, evidence=evidence,
                           prediction=rec["prediction"],
                           question_id=qid).to_jev_state()
    last = answer_fn.LAST_LOGPROB
    state["mean_logprob"] = last["mean_logprob"]
    state["n_logprob_tokens"] = last["n_logprob_tokens"]
    state["generator"] = GENERATOR_ID
    lp_conf = float(logprob_backend.ask(state, [grounding_question()]).confidence)

    em = exact_match(rec["prediction"], gold)
    row = {
        "question_id": qid, "question": question, "gold": gold,
        "prediction": rec["prediction"],
        "type": q.get("type", "unknown"),
        "correct": int(em),
        "rounds_used": rec["rounds_used"],
        "in_evidence": bool(prediction_in_evidence(rec["prediction"], evidence)),
        "generator": GENERATOR_ID,
        "jev_confidence": float(arec["confidence"]),
        "jev_action": arec["action"],
        "jev_latency_ms": arec["latency_ms"],
        "jev_input_tokens": arec["input_tokens"],
        "logprob_confidence": lp_conf,
        "logprob_action": action_for(lp_conf, threshold),
        "mean_logprob": last["mean_logprob"],
        "n_logprob_tokens": last["n_logprob_tokens"],
    }
    trace = {"question_id": qid, "rounds_used": rec["rounds_used"],
             "suff_input_tokens": rec["input_tokens"],
             "jev_abstain_input_tokens": arec["input_tokens"],
             "suff_latency_ms": rec["latency_ms"]}
    return row, trace


def compare(rows: list[dict], n_boot: int, seed: int) -> dict:
    """Both backends on identical rows; the two label proxies stay separate."""
    import numpy as np

    from jevrag._vendor import selective as _sel
    from jevrag.eval.calibration import aurc as _aurc
    from jevrag.eval.calibration import calibration_summary

    corr = np.array([r["correct"] for r in rows], dtype=float)
    jev = np.array([r["jev_confidence"] for r in rows], dtype=float)
    lp = np.array([r["logprob_confidence"] for r in rows], dtype=float)

    out: dict = {"n": len(rows), "base_em": float(corr.mean()),
                 "oracle_aurc": float(_sel.oracle_aurc(corr)),
                 "random_aurc": float(1.0 - corr.mean())}

    # --- Proxy 1: EM correctness -- identical labels for both backends ---
    block = {}
    for name, sig in (("jev", jev), ("logprob", lp)):
        keep = ~np.isnan(sig)
        cal = calibration_summary(sig[keep], corr[keep], n_bins=10)
        block[name] = {
            "aurc": float(_aurc(sig[keep], corr[keep])),
            "n_dropped_nan": int((~keep).sum()),
            "ece": cal["ece"], "brier": cal["brier"],
            "brier_skill": cal["brier_skill"],
            "n_distinct_confidences": int(np.unique(sig[keep]).size),
            "confidence_min": float(sig[keep].min()),
            "confidence_max": float(sig[keep].max()),
        }
    out["em_correctness"] = block

    mask = ~(np.isnan(jev) | np.isnan(lp))
    out["bootstrap_em_correctness"] = paired_bootstrap(
        jev[mask], lp[mask], corr[mask], n_boot, seed)

    # --- Proxy 2: the failure-mode split, kept separate, never merged ---
    # Positive = wrong prediction whose normalized content is absent from the
    # stopping round's evidence (the cleaner ungrounded cases).
    positive = np.array(
        [1.0 if ((not r["correct"]) and (not r["in_evidence"])) else 0.0
         for r in rows], dtype=float)
    usable = ~(np.isnan(jev) | np.isnan(lp))
    fm = {"definition": "positive = wrong prediction AND normalized content "
                        "absent from the stopping round's evidence",
          "n_positive": int(positive.sum()),
          "n_usable": int(usable.sum())}
    n_pos = int(positive[usable].sum())
    n_fm = int(usable.sum())
    # Anchors travel WITH the AURCs, so "both worse than random" is checkable at a
    # glance instead of being a prose claim. base_rate = positive.mean(); the
    # random-signal AURC is 1 - base_rate and the oracle is the perfect ranker.
    fm["base_rate"] = float(positive[usable].mean()) if n_fm else None
    fm["random_aurc"] = (1.0 - fm["base_rate"]) if n_fm else None
    if 0 < n_pos < n_fm:
        fm["oracle_aurc"] = float(_sel.oracle_aurc(positive[usable]))
        fm["aurc"] = {
            name: float(_aurc(sig[usable], positive[usable]))
            for name, sig in (("jev", jev), ("logprob", lp))
        }
        fm["both_worse_than_random"] = bool(
            all(v > fm["random_aurc"] for v in fm["aurc"].values()))
        fm["bootstrap"] = paired_bootstrap(
            jev[usable], lp[usable], positive[usable], n_boot, seed)
    else:
        fm["aurc"] = None
    out["failure_mode_split"] = fm

    # --- Selective risk at matched coverage (EM labels) ---
    risk_at = {}
    for cov in (0.5, 0.6, 0.7, 0.8, 0.9):
        entry = {}
        for name, sig in (("jev", jev), ("logprob", lp)):
            keep = ~np.isnan(sig)
            s, c = sig[keep], corr[keep]
            t = _sel.threshold_for_coverage(s, cov)
            answered = s >= t
            acc = float(c[answered].mean()) if answered.any() else None
            entry[name] = {
                "target_coverage": cov,
                "coverage": float(answered.mean()),
                "selective_accuracy": acc,
                "selective_risk": None if acc is None else 1.0 - acc,
                "n_answered": int(answered.sum()), "threshold": float(t),
            }
        risk_at[f"{cov:.2f}"] = entry
    out["selective_risk_at_matched_coverage"] = risk_at

    # --- Fixed-threshold operating points (existing defaults, untuned) ---
    ops = {}
    for name, act_key in (("jev", "jev_action"), ("logprob", "logprob_action")):
        passed = [r for r in rows if r[act_key] == "pass"]
        n_abstain = len(rows) - len(passed)
        wrong = [r for r in rows if not r["correct"]]
        abstained_wrong = [r for r in wrong if r[act_key] == "abstain"]
        ops[name] = {
            "abstain_rate": n_abstain / len(rows),
            "n_pass": len(passed), "n_abstain": n_abstain,
            "passed_em": (sum(r["correct"] for r in passed) / len(passed))
                         if passed else None,
            "overall_em": float(corr.mean()),
            "n_correct": int(corr.sum()), "n_wrong": len(wrong),
            "wrong_recall_abstained": (len(abstained_wrong) / len(wrong))
                                      if wrong else None,


            "abstain_precision": (len(abstained_wrong) / n_abstain)
                                 if n_abstain else None,
        }
    out["operating_point"] = ops
    return out


def print_report(summary: dict, n_boot: int, seed: int) -> None:
    print(f"\nn={summary['n']}  base EM {summary['base_em']:.4f}  "
          f"oracle AURC {summary['oracle_aurc']:.4f}  "
          f"random AURC {summary['random_aurc']:.4f}")

    print("\nproxy 1 -- EM correctness (identical rows, identical labels):")
    print(f"  {'backend':<10}{'AURC':>9}{'ECE':>9}{'Brier':>9}{'skill':>9}"
          f"{'distinct':>10}{'conf range':>24}")
    for name in ("jev", "logprob"):
        b = summary["em_correctness"][name]
        rng = f"[{b['confidence_min']:.4f}, {b['confidence_max']:.4f}]"
        print(f"  {name:<10}{b['aurc']:>9.4f}{b['ece']:>9.4f}{b['brier']:>9.4f}"
              f"{b['brier_skill']:>9.4f}{b['n_distinct_confidences']:>10}{rng:>24}")
    bs = summary["bootstrap_em_correctness"]
    verdict = "SIGNIFICANT" if bs["p_two_sided"] < 0.05 else "NOT significant"
    print(f"  paired bootstrap ({n_boot} resamples, seed {seed}), "
          f"delta = jev - logprob (negative favors jev): "
          f"{bs['observed_delta']:+.4f} "
          f"CI [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}] "
          f"p={bs['p_two_sided']:.4f} -- {verdict}")

    fm = summary["failure_mode_split"]
    print("\nproxy 2 -- failure-mode split (kept separate, NOT merged):")
    print(f"  {fm['definition']}")
    print(f"  n_positive={fm['n_positive']}  n_usable={fm['n_usable']}")
    if fm["aurc"]:
        # Anchors on one line, so "both worse than random" is checkable here
        # rather than asserted in prose. Lower AURC = better; higher than the
        # random line means the signal is worse than uninformative.
        print(f"  AURC (lower=better) | oracle {fm['oracle_aurc']:.4f} | "
              f"random {fm['random_aurc']:.4f} | jev {fm['aurc']['jev']:.4f} | "
              f"logprob {fm['aurc']['logprob']:.4f}")
        print(f"  both worse than random: {fm['both_worse_than_random']}"
              "  (positive class = UNGROUNDED, so high confidence should mark "
              "the ungrounded case -- inverse polarity to proxy 1)")
        b = fm["bootstrap"]
        v = "SIGNIFICANT" if b["p_two_sided"] < 0.05 else "NOT significant"
        print(f"  paired bootstrap: delta={b['observed_delta']:+.4f} "
              f"CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] "
              f"p={b['p_two_sided']:.4f} -- {v}")
    else:
        print("  degenerate for AURC (one class or no usable rows) -- not computed")

    print("\nselective risk at matched coverage (EM labels):")
    print(f"  {'target':<8}{'backend':<10}{'coverage':>10}{'risk':>9}{'n_ans':>8}")
    for target, entry in summary["selective_risk_at_matched_coverage"].items():
        for name in ("jev", "logprob"):
            r = entry[name]
            risk = "--" if r["selective_risk"] is None else f"{r['selective_risk']:.4f}"
            print(f"  {target:<8}{name:<10}{r['coverage']:>10.4f}{risk:>9}"
                  f"{r['n_answered']:>8}")

    print("\nfixed-threshold operating points (existing defaults, untuned):")
    print(f"  {'backend':<10}{'abstain':>9}{'n_pass':>8}{'passed EM':>12}"
          f"{'overall EM':>12}{'wrong abstained':>18}")
    for name, o in summary["operating_point"].items():
        pe = "--" if o["passed_em"] is None else f"{o['passed_em']:.4f}"
        wr = ("--" if o["wrong_recall_abstained"] is None
              else f"{o['wrong_recall_abstained']:.4f}")
        print(f"  {name:<10}{o['abstain_rate']:>9.4f}{o['n_pass']:>8}{pe:>12}"
              f"{o['overall_em']:>12.4f}{wr:>18}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="answer-abstain at n=300, Jev vs logprob, same items "
                    "(packet V03-B1)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="val")
    ap.add_argument("--threshold", type=float, default=DEFAULT_ABSTAIN_THRESHOLD)
    ap.add_argument("--per-round-k", type=int, default=5)
    ap.add_argument("--out", default="outputs/v03_b1_answer_abstain_n300.jsonl")
    ap.add_argument("--trace", default="outputs/v03_b1_trace_n300.jsonl")
    ap.add_argument("--summary", default="outputs/v03_b1_summary_n300.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--compare-only", action="store_true",
                    help="recompute metrics from the JSONL; no live calls")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    summary_path = REPO_ROOT / args.summary

    if args.compare_only:
        rows = [json.loads(line) for line in
                open(out_path, encoding="utf-8") if line.strip()]
        summary = compare(rows, args.n_boot, args.seed)
        print_report(summary, args.n_boot, args.seed)
        summary_path.write_text(
            json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
        return 0

    from openai import OpenAI

    from jevrag.backends import LogprobDecision

    questions = load_questions(split=args.split)[: args.n]
    # No sampling: the first n questions in file order off the frozen split.
    # The seed below governs the bootstrap resampling only.
    print(f"split={args.split} n={len(questions)} (frozen split from "
          f"data_loader.py; first {args.n} in file order, no sampling; "
          f"bootstrap seed {args.seed})")

    jev = JevDecision(api_key=load_jev_key())
    lp = LogprobDecision()
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=load_openrouter_key())
    template = Path(str(PROMPT_TEMPLATE_FILE)).read_text(encoding="utf-8")
    retrieve_fn = make_retrieve_fn()
    answer_fn = make_logprob_answer_fn(client, GENERATOR_MODEL, template)

    done: set[str] = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["question_id"])
    trace_path = REPO_ROOT / args.trace
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jev_tokens = 0
    t0 = time.perf_counter()
    for i, q in enumerate(questions, 1):
        qid = str(q["id"])
        if qid in done:
            continue
        row, trace = run_one(q, jev, lp, retrieve_fn, answer_fn,
                             args.per_round_k, args.threshold)
        jev_tokens += int(row["jev_input_tokens"])
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(questions)}] {qid} em={row['correct']} "
              f"jev={row['jev_confidence']:.2f} lp={row['logprob_confidence']:.4f} "
              f"rounds={row['rounds_used']} ntok={row['n_logprob_tokens']}",
              flush=True)
        if i % 10 == 0 or i == len(questions):
            print(f"  [{i}/{len(questions)}] "
                  f"{(time.perf_counter() - t0) / i:.2f} s/q", flush=True)

    rows = [json.loads(line) for line in
            open(out_path, encoding="utf-8") if line.strip()]
    summary = compare(rows, args.n_boot, args.seed)
    summary["jev_abstain_input_tokens_this_run"] = jev_tokens
    print_report(summary, args.n_boot, args.seed)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}, {trace_path}, {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
