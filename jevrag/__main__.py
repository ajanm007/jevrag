"""jevrag - the CLI. THE BAR (PRD §7):

    jevrag eval sufficiency --dataset hotpotqa

Returns a real risk-coverage curve, a real ECE number, an accuracy-vs-
fixed-iteration-baseline comparison, and a cost/latency table. On the real
dataset. No placeholders, no TODOs where the numbers go.

The measurement path consumes decision-path records (INTERFACE.md shape, JSONL).
It never calls Jev or a retriever itself - that seam is deliberate, so this
command runs identically against a live Jev run and against a records file
produced hours earlier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .benchmarks.hotpotqa import evaluate_results, label_records, split_lookup
from .eval.calibration import (
    accuracy_at_coverage,
    aurc,
    calibration_summary,
    coverage_at_accuracy,
    reliability_bins,
    risk_coverage_curve,
    threshold_for_coverage,
)
from .eval.cost import format_cost_table, summarize_cost
from .eval.crc import crc_calibrate, crc_readout

REQUIRED_KEYS = {
    "question_id", "question", "gold", "prediction",
    "confidence", "rounds_used", "latency_ms", "input_tokens", "type",
    "generator",
}

#: Generator tags whose accuracy/calibration numbers must never be presented
#: as real. Omission (not just a banner): a banner above a full table still
#: screenshots. Missing (None) counts too — unproven provenance is no
#: provenance. Cost/latency are always real measurements and are still shown.
NON_REPORTABLE_GENERATORS = {"extractive_fallback", "unknown", None}


def _generators_of(records: list[dict]) -> set:
    return {r.get("generator") for r in records}


def _check_generator_parity(records: list[dict],
                            baseline_records: list[dict] | None
                            ) -> tuple[str | None, str | None]:
    """Enforce the hard guard: parity between arms.

    Returns (gated_generator, baseline_generator). Raises ValueError — the
    caller turns it into a loud non-zero exit — when an arm mixes generators
    internally, or when the two arms disagree. A mismatched delta EM measures
    the generator, not the gate, and that delta is V1's headline claim.
    """
    gated = _generators_of(records)
    if len(gated) != 1:
        raise ValueError(
            f"gated arm mixes generators {sorted(gated, key=str)} — refusing "
            "to report over a mixed run"
        )
    gated_gen = next(iter(gated))
    if not baseline_records:
        return gated_gen, None
    base = _generators_of(baseline_records)
    if len(base) != 1:
        raise ValueError(
            f"baseline arm mixes generators {sorted(base, key=str)} — refusing "
            "to report over a mixed run"
        )
    baseline_gen = next(iter(base))
    if gated_gen != baseline_gen:
        raise ValueError(
            f"generator parity violated: gated={gated_gen!r} vs "
            f"baseline={baseline_gen!r} — delta EM would measure the "
            "generator, not the gate. Refusing the comparison."
        )
    return gated_gen, baseline_gen


def load_records(path: Path) -> list[dict]:
    """Load JSONL records, validating the INTERFACE.md shape loudly.

    Extra keys are allowed (e.g. ``policy``); missing keys are a bug in the
    decision path and we say so, per-record, instead of filling defaults.
    """
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = REQUIRED_KEYS - set(rec)
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: record missing INTERFACE.md keys {sorted(missing)}"
                )
            records.append(rec)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def join_splits(records: list[dict], splits: dict[str, str]) -> None:
    """Attach frozen val/test labels. Unknown question ids are rejected loudly."""
    for rec in records:
        qid = rec["question_id"]
        if qid not in splits:
            raise ValueError(
                f"record question_id {qid!r} is not in the dataset - "
                "cannot assign its frozen val/test split"
            )
        rec["split"] = splits[qid]


def _conf_correct(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    labeled = label_records(records)
    conf = np.array(
        [np.nan if r["confidence"] is None else float(r["confidence"]) for r in labeled]
    )
    corr = np.array([r["correct"] for r in labeled], dtype=float)
    return conf, corr


def _print_accuracy(em_f1: dict) -> None:
    print("\n--- Accuracy (EM / F1, SQuAD normalization) ---")
    ov = em_f1["overall"]
    print(f"  overall: EM {ov['em']:.4f}  F1 {ov['f1']:.4f}  (n={ov['count']})")
    for qtype, s in em_f1["by_type"].items():
        print(f"  {qtype:<10} EM {s['em']:.4f}  F1 {s['f1']:.4f}  (n={s['count']})")


def _print_calibration(cal: dict, bins: list[dict]) -> None:
    print("\n--- Calibration (is the confidence real?) ---")
    print(f"  ECE   : {cal['ece']:.4f}   (0 = perfect; n_bins={cal['n_bins']})")
    print(f"  MCE   : {cal['mce']:.4f}   (worst bin)")
    print(f"  Brier : {cal['brier']:.4f}   (skill vs base rate: {cal['brier_skill']:.4f})")
    print(f"  base rate correct: {cal['base_rate_correct']:.4f}   "
          f"NaN confidences: {cal['n_nan_confidence']}")
    print("  reliability table:")
    for b in bins:
        if b["n"] == 0:
            print(f"    [{b['lo']:.1f},{b['hi']:.1f})   n=0")
        else:
            print(f"    [{b['lo']:.1f},{b['hi']:.1f})   n={b['n']:<5} "
                  f"conf={b['mean_confidence']:.3f} acc={b['empirical_accuracy']:.3f} "
                  f"gap={b['gap']:.3f}")


def _print_risk_coverage(cal: dict, curve: dict, conf, corr) -> None:
    print("\n--- Risk / coverage (selective prediction) ---")
    print(f"  AURC        : {cal['aurc']:.4f}   (lower is better)")
    print(f"  oracle AURC : {cal['oracle_aurc']:.4f}   (perfect-signal lower bound)")
    print(f"  random AURC : {cal['random_aurc']:.4f}   (uninformative signal)")
    for cov_target in (0.8, 0.9, 0.95):
        print(f"  accuracy @ coverage {cov_target:.2f}: "
              f"{accuracy_at_coverage(conf, corr, cov_target):.4f}")
    for acc_target in (0.8, 0.9):
        print(f"  coverage @ accuracy {acc_target:.2f}: "
              f"{coverage_at_accuracy(conf, corr, acc_target):.4f}")
    print(f"  full curve: {len(curve['coverage'])} threshold points "
          f"(coverage {curve['coverage'][0]:.3f} -> {curve['coverage'][-1]:.3f})")


def _tune_operating_point(tune_records: list[dict], conf, corr,
                          target_coverage: float = 0.9,
                          applied_split: str = "test") -> dict:
    """Pick a threshold on the tune (val) split, report it applied to `conf`/`corr`.

    The threshold is tuned on POOLED val, deliberately: `threshold_for_coverage`
    never sees correctness labels — it picks a confidence quantile — and the
    15-point EM gap between val provenances lives in correctness
    (generator-side), not in the confidence distribution (Jev + retrieval are
    hosting-independent). Pooling n=300 stabilizes the quantile versus n=180 or
    n=120 alone. What IS reported per provenance subset is val-side selective
    accuracy at that threshold — the honesty requirement sits on the reported
    numbers, not the quantile computation. Mean confidence per subset is printed
    alongside so the hosting-independence premise stays a checked claim, not an
    assumption.
    Both sides are printed: what the threshold achieved on val (where it was
    picked — optimistic by construction) and what it delivers on the evaluated
    split. The gap between them IS the threshold-transferability finding
    (PRD §6). The accuracy/calibration sections above reflect the production
    gate the records were generated with (0.7 default) — this section is a
    tuned alternative shown alongside, never a silent replacement (§8).
    """
    t_conf, t_corr = _conf_correct(tune_records)
    threshold = threshold_for_coverage(t_conf, target_coverage=target_coverage)

    def _at(c, v):
        ans = np.asarray(c) >= threshold
        return (float(np.mean(ans)),
                float(np.mean(np.asarray(v)[ans])) if ans.any() else float("nan"))

    val_cov, val_acc = _at(t_conf, t_corr)
    print("\n--- Operating point (threshold tuned on VAL, applied here) ---")
    print(f"  production gate in these records: 0.7 (untuned default; "
          "main-report numbers above)")
    print(f"  tuned threshold {threshold:.4f} (target coverage {target_coverage:.2f} "
          f"on pooled val, n={len(t_conf)})")
    print(f"  val  (tuned here): coverage {val_cov:.4f}  selective accuracy {val_acc:.4f}")
    val_subsets = {}
    tune_subsets = _partition_by_generator(tune_records)
    if len(tune_subsets) > 1:
        for gen, sub in tune_subsets:
            sc, sv = _conf_correct(sub)
            c, a = _at(sc, sv)
            mean_c = float(np.nanmean(sc)) if len(sc) else float("nan")
            print(f"    per subset {gen!r} (n={len(sub)}): coverage {c:.4f}  "
                  f"selective accuracy {a:.4f}  mean_conf {mean_c:.4f}")
            val_subsets[str(gen)] = {"n": len(sub), "coverage": c,
                                     "selective_accuracy": a,
                                     "mean_confidence": mean_c}
    answered = np.asarray(conf) >= threshold
    app_cov = float(np.mean(answered))
    app_acc = float(np.mean(np.asarray(corr)[answered])) if answered.any() else float("nan")
    if applied_split == "test":
        print(f"  test (applied): coverage {app_cov:.4f}  selective accuracy {app_acc:.4f}")
    else:
        print(f"  {applied_split} (applied — same split as tune, optimistic, not "
              "transfer; transfer pending test data): coverage "
              f"{app_cov:.4f}  selective accuracy {app_acc:.4f}")
    return {
        "tuned_on": "val",
        "pooled": True,
        "tune_generators": sorted([str(g) for g, _ in tune_subsets]),
        "target_coverage": target_coverage,
        "threshold": threshold,
        "val": {"n": len(t_conf), "coverage": val_cov, "selective_accuracy": val_acc},
        "val_subsets": val_subsets,
        "applied_split": applied_split,
        "applied": {"coverage": app_cov, "selective_accuracy": app_acc},
    }


def _paired_rounds_summary(records: list[dict],
                           baseline_records: list[dict]) -> dict:
    """Paired gated-vs-baseline comparison by question_id (Finding 1).

    Equal EM is the *precondition* for the sufficiency finding, not the
    finding: the axis the gate wins on is retrieval rounds avoided. The
    early-stopped subset — questions where the gate ran fewer rounds than the
    baseline on the SAME question — gets its own accuracy line, because that
    is where the gate actually intervened.
    """
    base_by_id = {r["question_id"]: r for r in baseline_records}
    pairs = [(g, base_by_id[g["question_id"]]) for g in records
             if g["question_id"] in base_by_id]
    gated_rounds = [int(g["rounds_used"]) for g, _ in pairs]
    base_rounds = [int(b["rounds_used"]) for _, b in pairs]
    g_total = int(sum(gated_rounds))
    b_total = int(sum(base_rounds))
    reduction = (b_total - g_total) / b_total if b_total else 0.0
    n_same = sum(1 for g, b in pairs if g["prediction"] == b["prediction"])
    early = [(g, b) for g, b in pairs
             if int(g["rounds_used"]) < int(b["rounds_used"])]
    early_stats = None
    if early:
        g_lab = label_records([g for g, _ in early])
        b_lab = label_records([b for _, b in early])
        g_em = float(sum(r["correct"] for r in g_lab) / len(g_lab))
        b_em = float(sum(r["correct"] for r in b_lab) / len(b_lab))
        early_stats = {"n": len(early), "gated_em": g_em,
                       "baseline_em": b_em, "delta_em": g_em - b_em}
    return {
        "n_paired": len(pairs),
        "n_unpaired_gated": len(records) - len(pairs),
        "gated_total": g_total,
        "baseline_total": b_total,
        "gated_mean": (g_total / len(pairs)) if pairs else 0.0,
        "baseline_mean": (b_total / len(pairs)) if pairs else 0.0,
        "reduction_frac": reduction,
        "identical_predictions": {"n_same": n_same, "n": len(pairs)},
        "early_stopped": early_stats,
    }


def _partition_by_generator(records: list[dict]) -> list[tuple]:
    """Split records by generator id, sorted, missing (None) last."""
    by_gen: dict = {}
    for r in records:
        by_gen.setdefault(r.get("generator"), []).append(r)
    return sorted(by_gen.items(), key=lambda kv: (kv[0] is None, str(kv[0])))


def _print_population_sections(records: list[dict],
                               baseline_subset: list[dict] | None,
                               gen, n_bins: int,
                               tune_records: list[dict] | None = None,
                               coverage_target: float = 0.9,
                               applied_split: str = "test",
                               crc_alphas: list[float] | None = None) -> dict:
    """Accuracy → cost → baseline blocks for ONE provenance-uniform population.

    Returns its section dict (nulls when unreportable). Prints nothing for
    accuracy-derived sections when the generator is unproven — omission, not
    banner. Cost/latency always print. The caller prints banners/headers and
    guarantees uniformity; blending never happens here by construction.
    """
    reportable = gen not in NON_REPORTABLE_GENERATORS
    section: dict = {
        "generator": gen,
        "verdict": "reportable" if reportable else f"unreportable: generator is {gen!r}",
        "em_f1": None, "calibration": None, "reliability_bins": None,
        "risk_coverage": None, "operating_point": None, "crc": None,
        "baseline": None,
    }
    if reportable:
        conf, corr = _conf_correct(records)
        cal = calibration_summary(conf, corr, n_bins=n_bins)
        curve = risk_coverage_curve(conf, corr)
        bins = reliability_bins(conf, corr, n_bins=n_bins)
        em_f1 = evaluate_results(
            [{"prediction": r["prediction"], "gold": r["gold"], "type": r["type"]}
             for r in records]
        )
        _print_accuracy(em_f1)
        _print_calibration(cal, bins)
        _print_risk_coverage(cal, curve, conf, corr)
        if tune_records is not None:
            section["operating_point"] = _tune_operating_point(
                tune_records, conf, corr, coverage_target, applied_split)
        if crc_alphas:
            if tune_records is not None:
                t_conf, t_corr = _conf_correct(tune_records)
                section["crc"] = _print_crc_block(
                    t_conf, t_corr, _record_keys(tune_records),
                    conf, corr, _record_keys(records), crc_alphas,
                    in_sample=False)
            else:
                keys = _record_keys(records)
                section["crc"] = _print_crc_block(
                    conf, corr, keys, conf, corr, keys, crc_alphas,
                    in_sample=True)
        section.update(em_f1=em_f1, calibration=cal, reliability_bins=bins,
                       risk_coverage={k: v.tolist() for k, v in curve.items()})

    print("\n--- Cost / latency ---")
    cost_summary = summarize_cost(records)
    print(format_cost_table(cost_summary))
    # Finding 2: latency is NOT comparable across arms — gated wall-clock
    # includes up to 3 Jev decision calls per question, baseline includes
    # none. Comparing them penalizes the gate for deciding. Rounds (baseline
    # section) is the honest cost proxy.
    print("  note: latency not comparable across arms (gated includes Jev "
          "decision calls); compare rounds, not wall-clock.")
    section["cost"] = cost_summary

    if baseline_subset:
        if not reportable:
            print("\n--- Baseline cost/latency (accuracy comparison omitted) ---")
            print(format_cost_table(summarize_cost(baseline_subset)))
        else:
            bl_em_f1 = evaluate_results(
                [{"prediction": r["prediction"], "gold": r["gold"], "type": r["type"]}
                 for r in baseline_subset]
            )
            bl_cost = summarize_cost(baseline_subset)
            policy = baseline_subset[0].get("policy", "fixed_iteration")
            print(f"\n--- Baseline comparison ({policy}, n={len(baseline_subset)}) ---")
            bov = bl_em_f1["overall"]
            ov = em_f1["overall"]
            print(f"  baseline    EM {bov['em']:.4f}  F1 {bov['f1']:.4f}")
            print(f"  sufficiency EM {ov['em']:.4f}  F1 {ov['f1']:.4f}")
            print(f"  delta EM {ov['em'] - bov['em']:+.4f}   "
                  f"delta F1 {ov['f1'] - bov['f1']:+.4f}")
            paired = _paired_rounds_summary(records, baseline_subset)
            print(f"  rounds (retrieval): gated total {paired['gated_total']} "
                  f"(mean {paired['gated_mean']:.2f}) vs baseline total "
                  f"{paired['baseline_total']} (mean {paired['baseline_mean']:.2f})  "
                  f"{-paired['reduction_frac']:+.1%}")
            same = paired["identical_predictions"]
            print(f"  identical predictions: {same['n_same']}/{same['n']} "
                  f"({same['n_same'] / same['n']:.1%})" if same["n"] else
                  "  identical predictions: n/a (no paired records)")
            if paired["n_unpaired_gated"]:
                print(f"  note: {paired['n_unpaired_gated']} gated records "
                      "have no baseline pair and are excluded above")
            early = paired["early_stopped"]
            if early:
                print(f"  early-stopped (gated fewer rounds): n={early['n']}")
                print(f"    on those: gated EM {early['gated_em']:.4f} vs "
                      f"baseline EM {early['baseline_em']:.4f}  "
                      f"({early['delta_em']:+.4f})")
            splits_seen = sorted({r.get("split", "?") for r in records})
            print(f"  scope: {'+'.join(splits_seen)} split(s), one generator "
                  f"({gen}), production (untuned) gate — a rounds saving "
                  "at equal accuracy holds on THIS split, not generally.")
            print("  baseline cost/latency:")
            print(format_cost_table(bl_cost))
            if bl_cost["input_tokens"]["total"] == 0:
                # Finding 3: baseline token accounting is unpopulated on this
                # run (generator tokens were deliberately not recorded
                # mid-run to keep cost semantics consistent). 0 is missing
                # data, not free.
                print("  note: baseline input_tokens is UNPOPULATED — 0 does "
                      "not mean free. Rounds above is the honest cost proxy. "
                      "First item for the next records run: record generator "
                      "tokens (usage-dict answer_fn form) BEFORE test-split "
                      "records are generated.")
            section["baseline"] = {"policy": policy, "em_f1": bl_em_f1,
                                   "cost": bl_cost, "paired": paired}
    return section


def print_report(records: list[dict], baseline_records: list[dict] | None,
                 n_bins: int, out_path: Path | None,
                 tune_records: list[dict] | None = None,
                 coverage_target: float = 0.9,
                 applied_split: str = "test",
                 crc_alphas: list[float] | None = None) -> dict:
    """Assemble and print the full eval report. Returns the report dict.

    Single provenance: the original flat report (parity mismatch still raises
    via _check_generator_parity). Mixed provenance: disclosed per-subset
    sections, never averaged — the parity guard's refusal of BLENDED numbers
    stands, and this is the display path for the disclosed case. Cost is
    reported per subset (regimes differ); the operating-point transfer lives
    on the test evaluation, not here.
    """
    subsets = _partition_by_generator(records)
    base_by_gen = dict(_partition_by_generator(baseline_records or []))

    print("=" * 72)
    print("JevRAG - eval sufficiency - hotpotqa")
    print("=" * 72)
    print(f"\nrecords: {len(records)}  "
          f"(val={sum(r['split'] == 'val' for r in records)}, "
          f"test={sum(r['split'] == 'test' for r in records)})")

    if len(subsets) == 1 and len(base_by_gen) <= 1:
        gated_gen, baseline_gen = _check_generator_parity(records, baseline_records)
        print(f"generator: {gated_gen}  "
              f"policy: {records[0].get('policy', 'sufficiency_jev')}")
        if baseline_records:
            print(f"baseline: generator {baseline_gen}  "
                  f"policy: {baseline_records[0].get('policy', 'fixed_iteration')}  "
                  f"n={len(baseline_records)}")
        reportable = gated_gen not in NON_REPORTABLE_GENERATORS
        if not reportable:
            print(f"\n!!! UNREPORTABLE - generator provenance is {gated_gen!r}. "
                  "Accuracy, calibration, risk/coverage, and baseline-delta "
                  "numbers are OMITTED: they would measure the generator (or "
                  "nothing), not the gate. Cost/latency below are real and "
                  "still shown.")
        section = _print_population_sections(
            records, baseline_records, gated_gen, n_bins,
            tune_records=tune_records, coverage_target=coverage_target,
            applied_split=applied_split, crc_alphas=crc_alphas)
        report = {"dataset": "hotpotqa", "n_records": len(records),
                  "mode": "single", **section}
    else:
        # Mixed provenance: disclose per subset, average nothing. A blended
        # number over two accuracy populations is exactly what this project
        # exists to avoid producing — a real, investigated hosting-provenance
        # gap (not a bug) is exactly the case this guards against.
        print("mixed provenance — disclosed per subset, never averaged "
              "(blended numbers refused by design):")
        for gen, sub in subsets:
            print(f"  subset {gen!r} (n={len(sub)}, "
                  f"policy {sub[0].get('policy', 'sufficiency_jev')})")
        for gen in base_by_gen:
            if gen not in dict(subsets):
                print(f"  note: baseline subset {gen!r} has no gated "
                      "counterpart — ignored, not compared")
        print("operating-point transfer is reported with the test evaluation "
              "(--split test); val subsets show accuracy/calibration only.")
        sections = {}
        for gen, sub in subsets:
            base_sub = base_by_gen.get(gen)
            print(f"\n--- subset: {gen!r} (n={len(sub)}) ---")
            if base_sub is None:
                print(f"  note: no baseline records share generator {gen!r} "
                      "— comparison omitted for this subset")
            if gen in NON_REPORTABLE_GENERATORS:
                print(f"\n!!! UNREPORTABLE subset — generator {gen!r}; "
                      "accuracy-derived sections omitted below.")
            sections[str(gen)] = _print_population_sections(
                sub, base_sub, gen, n_bins)
        verdict = ("reportable" if all(s["verdict"] == "reportable"
                                       for s in sections.values())
                   else "unreportable: a subset generator is unproven")
        report = {"dataset": "hotpotqa", "n_records": len(records),
                  "mode": "mixed-subsets", "verdict": verdict,
                  "subsets": sections, "cost": None, "operating_point": None,
                  "baseline": None, "em_f1": None, "calibration": None,
                  "reliability_bins": None, "risk_coverage": None}

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport written to {out_path}")
    return report


def _print_calibration_block(conf: np.ndarray, corr: np.ndarray, n_bins: int) -> dict:
    """Shared AURC/risk-coverage/ECE/Brier block for the non-sufficiency
    primitives below. Same math as sufficiency's report (`_conf_correct` +
    `calibration_summary` + `risk_coverage_curve`), reused rather than
    reimplemented — same instruction every primitive's own build followed.
    """
    cal = calibration_summary(conf, corr, n_bins=n_bins)
    curve = risk_coverage_curve(conf, corr)
    bins = reliability_bins(conf, corr, n_bins=n_bins)
    _print_calibration(cal, bins)
    _print_risk_coverage(cal, curve, conf, corr)
    return {"calibration": cal, "reliability_bins": bins,
            "risk_coverage": {k: v.tolist() for k, v in curve.items()}}


def cmd_eval_chunk_boundary(args: argparse.Namespace) -> int:
    """Report on chunk-boundary records (jevrag/primitives/chunk_boundary.py).

    Confidence = jev_confidence (P(this boundary is a real split)), scored
    directly against the ground-truth label (is_split) — same convention
    as scripts/eval_chunk_boundary.py's own report (`correct = is_split`,
    verified against that script directly, not assumed). This is
    calibration of the split-probability itself, not "did Jev's own
    thresholded call agree with the label" — a different, narrower
    question this report doesn't compute. One-shot decision, no rounds, no
    baseline arm the way sufficiency has fixed_iteration — the
    cosine-similarity baseline is a separate, non-Jev score reported
    alongside for the ranking comparison, not folded into one blended
    number.
    """
    records = load_records_generic(Path(args.records),
                                    {"doc_id", "boundary_index", "is_split",
                                     "jev_confidence", "cosine_score"})
    conf = np.array([float(r["jev_confidence"]) for r in records])
    label = np.array([int(r["is_split"]) for r in records], dtype=float)
    correct = label

    print("=" * 72)
    print("JevRAG - eval chunk-boundary")
    print("=" * 72)
    print(f"\nrecords: {len(records)}  doc(s): "
          f"{sorted({r['doc_id'] for r in records})}")
    print(f"policy: {records[0].get('policy', 'chunk_boundary_jev')}")

    section = _print_calibration_block(conf, correct, args.bins)

    # Same convention as the Jev arm above: score the raw cosine value
    # directly against the ground-truth label, no derived agreement flag.
    cos = np.array([float(r["cosine_score"]) for r in records])
    cos_thresholded_accuracy = float(
        np.mean(((cos >= args.cosine_threshold).astype(float) == label))
    )
    cos_aurc = aurc(cos, label)
    print("\n--- Cosine-similarity baseline (non-Jev ranking score) ---")
    print(f"  thresholded accuracy @ {args.cosine_threshold:.2f}: "
          f"{cos_thresholded_accuracy:.4f}")
    print(f"  AURC: {cos_aurc:.4f}   (Jev AURC above: {section['calibration']['aurc']:.4f})")
    print("  note: reported for comparison, not blended with the Jev "
          "numbers above — same guard sufficiency's report uses for its "
          "baseline arm.")

    report = {"primitive": "chunk_boundary", "n_records": len(records),
              "base_rate": float(np.mean(correct)), **section,
              "cosine_baseline": {
                  "threshold": args.cosine_threshold,
                  "thresholded_accuracy": cos_thresholded_accuracy,
                  "aurc": cos_aurc,
              }}
    _write_report(report, args.out)
    return 0


def cmd_eval_context_selection(args: argparse.Namespace) -> int:
    """Report on context-selection records (jevrag/primitives/context_selection.py).

    Confidence = jev_confidence, correct = the passage's real relevance
    label (`relevant`), scored directly — same convention verified against
    scripts/eval_context_selection.py's own report (`correct = [int(r["relevant"])
    for r in rows]`), not a derived "did Jev's own thresholded selection
    agree with the label" flag. Per-passage rows only by default (the
    like-for-like arm against the rag-jev adapter) — batched rows use a
    different cost_scope and would need collapsing before any token/latency
    sum; --arm lets the caller pick which to report on rather than
    silently mixing them.
    """
    records = load_records_generic(Path(args.records),
                                    {"query_id", "relevant",
                                     "jev_confidence", "jev_selected"})
    if args.arm:
        records = [r for r in records if r.get("arm") == args.arm]
        if not records:
            print(f"error: no records with arm={args.arm!r}", file=sys.stderr)
            return 2
    conf = np.array([float(r["jev_confidence"]) for r in records])
    correct = np.array([int(r["relevant"]) for r in records], dtype=float)

    print("=" * 72)
    print("JevRAG - eval context-selection")
    print("=" * 72)
    arms = sorted({r.get("arm", "?") for r in records})
    print(f"\nrecords: {len(records)}  arm(s): {arms}  "
          f"queries: {len({r['query_id'] for r in records})}")
    print(f"policy: {records[0].get('policy', 'context_selection_jev')}")

    section = _print_calibration_block(conf, correct, args.bins)

    jev_selected = np.array([bool(r["jev_selected"]) for r in records])
    tp = int(np.sum(jev_selected & (correct == 1.0)))
    fp = int(np.sum(jev_selected & (correct == 0.0)))
    fn = int(np.sum(~jev_selected & (correct == 1.0)))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print("\n--- Selection precision/recall at the recorded operating point ---")
    print(f"  precision: {precision:.4f}   recall: {recall:.4f}   "
          f"selected: {int(jev_selected.sum())}/{len(records)}")

    bm25_baseline = None
    if "bm25_confidence" in records[0]:
        bm25_conf = np.array([float(r["bm25_confidence"]) for r in records])
        bm25_aurc = aurc(bm25_conf, correct)
        print("\n--- BM25 baseline (non-Jev ranking score) ---")
        print(f"  AURC: {bm25_aurc:.4f}   (Jev AURC above: "
              f"{section['calibration']['aurc']:.4f})")
        print("  note: reported for comparison only — not blended with "
              "Jev numbers.")
        bm25_baseline = {"aurc": bm25_aurc}

    report = {"primitive": "context_selection", "n_records": len(records),
              "arm": args.arm, "base_rate_relevant": float(np.mean(correct)),
              "precision": precision, "recall": recall,
              "bm25_baseline": bm25_baseline, **section}
    _write_report(report, args.out)
    return 0


def cmd_eval_answer_abstain(args: argparse.Namespace) -> int:
    """Report on answer-abstain records (jevrag/primitives/answer_abstain.py).

    Confidence = the grounding probability. Correct = the EM-proxy label —
    this is a proxy for "grounded," not ground truth, and the report says
    so. The headline number is the selective effect (passed-subset EM vs
    overall), not the calibration numbers alone.
    """
    records = load_records_generic(Path(args.records),
                                    {"question_id", "confidence", "action",
                                     "correct"})
    conf = np.array([float(r["confidence"]) for r in records])
    correct = np.array([int(r["correct"]) for r in records], dtype=float)
    passed = np.array([r["action"] == "pass" for r in records])

    print("=" * 72)
    print("JevRAG - eval answer-abstain")
    print("=" * 72)
    print(f"\nrecords: {len(records)}  policy: "
          f"{records[0].get('policy', 'answer_abstain_jev')}")
    print("note: 'correct' is an EM-proxy for 'grounded', not verified "
          "ground truth — a disclosed limitation of this proxy label.")

    section = _print_calibration_block(conf, correct, args.bins)

    overall_em = float(np.mean(correct))
    abstain_rate = float(np.mean(~passed))
    passed_em = float(np.mean(correct[passed])) if passed.any() else float("nan")
    print("\n--- Selective effect (the actual headline number) ---")
    print(f"  overall EM        : {overall_em:.4f}")
    print(f"  abstain rate      : {abstain_rate:.4f}")
    print(f"  passed-subset EM  : {passed_em:.4f}   "
          f"(delta {passed_em - overall_em:+.4f})")

    report = {"primitive": "answer_abstain", "n_records": len(records),
              "overall_em": overall_em, "abstain_rate": abstain_rate,
              "passed_em": passed_em, **section}
    _write_report(report, args.out)
    return 0


def cmd_eval_cache_trust(args: argparse.Namespace) -> int:
    """Report on cache-trust records (jevrag/primitives/cache_trust.py).

    Confidence = serve_from_cache probability. Correct = whether the cached
    answer was actually right (real EM against gold). Fallback rows
    (fallback_used=True, confidence=None) are excluded from calibration,
    matching cache_trust.py's own NaN-confidence convention — never
    scored, never fabricated.
    """
    records = load_records_generic(Path(args.records),
                                    {"query_id", "confidence", "action",
                                     "correct", "fallback_used"})
    scored = [r for r in records if not r.get("fallback_used")
              and r.get("confidence") is not None]
    n_fallback = len(records) - len(scored)
    if not scored:
        print("error: no scoreable records (all fallback/no-confidence)",
              file=sys.stderr)
        return 2
    conf = np.array([float(r["confidence"]) for r in scored])
    correct = np.array([int(r["correct"]) for r in scored], dtype=float)

    print("=" * 72)
    print("JevRAG - eval cache-trust")
    print("=" * 72)
    print(f"\nrecords: {len(records)}  scored: {len(scored)}  "
          f"fallback/unscored: {n_fallback}")
    print(f"policy: {records[0].get('policy', 'cache_trust_jev')}")

    section = _print_calibration_block(conf, correct, args.bins)

    served = np.array([r["action"] == "serve" for r in scored])
    served_em = float(np.mean(correct[served])) if served.any() else float("nan")
    print("\n--- Serve accuracy (the actual headline number) ---")
    print(f"  served EM (of what was actually served) : {served_em:.4f}   "
          f"(n={int(served.sum())})")
    print(f"  serve rate                               : "
          f"{float(np.mean(served)):.4f}")

    report = {"primitive": "cache_trust", "n_records": len(records),
              "n_fallback": n_fallback, "served_em": served_em, **section}
    _write_report(report, args.out)
    return 0


def load_records_generic(path: Path, required_keys: set) -> list[dict]:
    """Load JSONL records for the non-sufficiency primitives.

    Same loud-validation spirit as `load_records` (sufficiency's own
    loader), simpler required-key set since these primitives don't share
    sufficiency's INTERFACE.md contract (no rounds_used/generator parity
    concept for a one-shot decision).
    """
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = required_keys - set(rec)
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: record missing required keys {sorted(missing)}"
                )
            records.append(rec)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def _record_keys(records: list[dict]) -> list[str]:
    """Stable per-row tie-break keys for CRC.

    question_id when it is present and unique (every primitive's records carry
    it). Otherwise the row index — which breaks ties reproducibly within one
    file but is not a property of the row, and CRC's tie-break needs to be one,
    or a fresh question could not follow the same rule. The fallback is
    announced rather than assumed.
    """
    ids = [str(r["question_id"]) for r in records if "question_id" in r]
    if len(ids) == len(records) and len(set(ids)) == len(ids):
        return ids
    print("  note: CRC tie-break keys fall back to row indices (records lack a "
          "unique question_id) — reproducible within this file, not per row "
          "across files", file=sys.stderr)
    return [str(i) for i in range(len(records))]


def _print_crc_block(
    cal_conf: np.ndarray,
    cal_corr: np.ndarray,
    cal_keys: list,
    eval_conf: np.ndarray,
    eval_corr: np.ndarray,
    eval_keys: list,
    alphas: list[float],
    *,
    in_sample: bool,
) -> list[dict]:
    """CRC per-decision thresholds — the error-budget counterpart to the
    coverage-target operating point. Calibrated on the tune side (val when
    --tune-records is given), read out on the reported split. With no separate
    tune records the read-out is in-sample and says so: the bound then holds by
    construction via the monotone envelope, which proves mechanics, not the
    guarantee.

    Ties are broken by a deterministic hash of each row's key (see
    jevrag/eval/crc.py): the threshold is in decision space, so `threshold=`
    prints the raw confidence of the k-hat-th row while the answer rule is
    `score + eps*offset(key) >= threshold`. Without that, quantized confidences
    let the value rule answer whole tie blocks and overshoot k-hat.
    """
    print("\n--- CRC thresholds (conformal risk control, B=1) ---")
    print("  tie policy: deterministic sha256(question_id) tie-break; answer "
          "iff score + 1e-06*offset(key) >= threshold")
    if in_sample:
        print(
            "  calibrated and read out on the SAME records (no --tune-records):\n"
            "  in-sample the bound holds by construction — this proves mechanics,\n"
            "  not the exchangeability guarantee"
        )
    rows = []
    for alpha in alphas:
        calib = crc_calibrate(cal_conf, cal_corr, alpha, keys=cal_keys)
        ro = crc_readout(eval_conf, eval_corr, eval_keys, calib["threshold"])
        holds = bool(ro["risk"] <= alpha) if np.isfinite(ro["risk"]) else None
        if calib["abstain_all"]:
            print(
                f"  alpha={alpha:.3f}: abstain-all (no k satisfies the bound; "
                f"B/(n+1)={calib['b_correction']:.4f}) -> coverage {ro['coverage']:.4f}"
            )
        else:
            risk_s = f"{ro['risk']:.4f}" if np.isfinite(ro["risk"]) else "nan"
            print(
                f"  alpha={alpha:.3f}: threshold={calib['threshold_score']:.4f} "
                f"(k-hat={calib['k_hat']}/{calib['n_cal']}, "
                f"answered-cal={calib['n_answered_cal']}, "
                f"B/(n+1)={calib['b_correction']:.4f}) -> "
                f"coverage {ro['coverage']:.4f}, selective risk {risk_s} "
                f"vs alpha (holds: {holds})"
            )
        rows.append(
            {**calib,
             "eval_coverage": ro["coverage"],
             "eval_risk": ro["risk"],
             "eval_n_answered": ro["n_answered"],
             "guarantee_holds_empirically": holds,
             "in_sample": in_sample}
        )
    return rows



def _write_report(report: dict, out: str | None) -> None:
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport written to {out_path}")


def cmd_eval_sufficiency(args: argparse.Namespace) -> int:
    splits = split_lookup()

    records = load_records(Path(args.records))
    join_splits(records, splits)

    baseline_records = None
    if args.baseline_records:
        baseline_records = load_records(Path(args.baseline_records))
        join_splits(baseline_records, splits)

    tune_records = None
    if args.tune_records:
        tune_records = load_records(Path(args.tune_records))
        join_splits(tune_records, splits)
        # The operating point is tuned on val by definition. If the tune file
        # carries other splits, use only val — and say so.
        n_tune = len(tune_records)
        tune_records = [r for r in tune_records if r.get("split") == "val"]
        if len(tune_records) < n_tune:
            print(f"note: tune file has {n_tune} records, "
                  f"{len(tune_records)} on val — tuning on val only",
                  file=sys.stderr)
        if not tune_records:
            print("note: no val records in tune file — skipping operating point",
                  file=sys.stderr)
            tune_records = None

    split = args.split  # 'test' (default), 'val', or 'all'
    if split != "all":
        records = [r for r in records if r["split"] == split]
        if baseline_records:
            baseline_records = [r for r in baseline_records if r["split"] == split]
        if not records:
            print(f"error: no sufficiency records in split {split!r}", file=sys.stderr)
            return 2

    out_path = Path(args.out) if args.out else None
    report = print_report(records, baseline_records, n_bins=args.bins,
                          out_path=out_path, tune_records=tune_records,
                          coverage_target=args.coverage_target,
                          applied_split=split, crc_alphas=args.crc_alpha)
    # Finding 4: an eval whose accuracy numbers are unreportable is a failed
    # eval. Exit 0 would let a script or CI treat it as success — same spirit
    # as omitting the numbers instead of bannering them.
    return 1 if report["verdict"] != "reportable" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="run the calibration-first eval harness")
    eval_sub = p_eval.add_subparsers(dest="primitive", required=True)

    p_suf = eval_sub.add_parser(
        "sufficiency", help="evidence-sufficiency primitive (V1)"
    )
    p_suf.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa"])
    p_suf.add_argument(
        "--records", default="records/sufficiency_hotpotqa.jsonl",
        help="decision-path records JSONL (INTERFACE.md shape)",
    )
    p_suf.add_argument(
        "--baseline-records", default="records/fixed_iter3_hotpotqa.jsonl",
        help="fixed-iteration baseline records JSONL (skipped if file absent)",
    )
    p_suf.add_argument(
        "--split", default="test", choices=["val", "test", "all"],
        help="frozen split to report on (default: test - thresholds tuned on val)",
    )
    p_suf.add_argument("--bins", type=int, default=10, help="ECE bin count")
    p_suf.add_argument(
        "--tune-records", default=None,
        help="VAL-split records used to pick the operating threshold; when given, "
             "the report includes the operating point (tuned on val, applied to "
             "the records being evaluated)",
    )
    p_suf.add_argument(
        "--coverage-target", type=float, default=0.9,
        help="target coverage for the val-tuned operating threshold (default 0.9)",
    )
    p_suf.add_argument(
        "--crc-alpha", type=float, action="append", default=None,
        metavar="ALPHA",
        help="error budget for a CRC-calibrated threshold (repeatable, e.g. "
             "--crc-alpha 0.1 --crc-alpha 0.2). Calibrated on the tune/val "
             "records, read out on the reported split; with no --tune-records "
             "the read-out is in-sample (bound holds by construction)",
    )
    p_suf.add_argument("--out", default=None, help="write the full report JSON here")
    p_suf.set_defaults(func=cmd_eval_sufficiency)

    p_cb = eval_sub.add_parser(
        "chunk-boundary", help="chunk-boundary primitive (V1.1)"
    )
    p_cb.add_argument(
        "--records", default="outputs/chunk_boundary_wikipedia.jsonl",
        help="chunk-boundary records JSONL (jevrag/primitives/chunk_boundary.py shape)",
    )
    p_cb.add_argument("--bins", type=int, default=10, help="ECE bin count")
    p_cb.add_argument(
        "--cosine-threshold", type=float, default=0.5,
        help="split threshold for the non-Jev cosine baseline (default 0.5)",
    )
    p_cb.add_argument("--out", default=None, help="write the full report JSON here")
    p_cb.set_defaults(func=cmd_eval_chunk_boundary)

    p_cs = eval_sub.add_parser(
        "context-selection", help="context-selection primitive"
    )
    p_cs.add_argument(
        "--records", default="outputs/context_selection_hotpotqa_val.jsonl",
        help="context-selection records JSONL "
             "(jevrag/primitives/context_selection.py shape)",
    )
    p_cs.add_argument(
        "--arm", default="per_passage", choices=["per_passage", "batched", None],
        help="which arm to report on when the file has both (default: per_passage, "
             "the like-for-like arm against the rag-jev adapter)",
    )
    p_cs.add_argument("--bins", type=int, default=10, help="ECE bin count")
    p_cs.add_argument("--out", default=None, help="write the full report JSON here")
    p_cs.set_defaults(func=cmd_eval_context_selection)

    p_aa = eval_sub.add_parser(
        "answer-abstain", help="answer-abstain primitive"
    )
    p_aa.add_argument(
        "--records", default="outputs/answer_abstain_val.jsonl",
        help="answer-abstain records JSONL "
             "(jevrag/primitives/answer_abstain.py shape)",
    )
    p_aa.add_argument("--bins", type=int, default=10, help="ECE bin count")
    p_aa.add_argument("--out", default=None, help="write the full report JSON here")
    p_aa.set_defaults(func=cmd_eval_answer_abstain)

    p_ct = eval_sub.add_parser(
        "cache-trust", help="cache-trust primitive"
    )
    p_ct.add_argument(
        "--records", default="outputs/cache_trust_val.jsonl",
        help="cache-trust records JSONL (jevrag/primitives/cache_trust.py shape)",
    )
    p_ct.add_argument("--bins", type=int, default=10, help="ECE bin count")
    p_ct.add_argument("--out", default=None, help="write the full report JSON here")
    p_ct.set_defaults(func=cmd_eval_cache_trust)

    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if getattr(args, "baseline_records", None) and not Path(args.baseline_records).exists():
        print(f"note: baseline records {args.baseline_records} not found - "
              "skipping baseline comparison", file=sys.stderr)
        args.baseline_records = None
    if getattr(args, "tune_records", None) and not Path(args.tune_records).exists():
        print(f"note: tune records {args.tune_records} not found - "
              "skipping operating point", file=sys.stderr)
        args.tune_records = None
    if not Path(args.records).exists():
        print(f"error: records file {args.records} not found. The decision path "
              "produces it; see INTERFACE.md.", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except ValueError as e:
        # Loud refusal, not a traceback: bad records, unknown ids, mixed or
        # mismatched generators. Exit 2 like the other usage errors above.
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
