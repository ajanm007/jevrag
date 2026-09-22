"""Report on a shadow-ledger JSONL file — the observe-mode readout.

Reads the ledger a ShadowDecision wrapper wrote and prints what the
challenger would have done: call counts, per-backend / per-question
breakdown, confidence distribution, and cost (via the shared
``jevrag.eval.cost`` machinery — reused, not reimplemented). With
``--records``, positionally joins the ledger against the labeled records
file it was replayed from (same order — valid exactly when the ledger was
produced by replaying that file, as ``scripts/demo_shadow_mode.py`` does;
anything else is a mismatch this script refuses) and adds the shared
``jevrag.eval.calibration`` summary plus challenger-vs-production action
agreement at ``--threshold``.

Usage:
    py scripts/report_shadow_ledger.py --ledger outputs/shadow_logprob_demo.jsonl
    py scripts/report_shadow_ledger.py --ledger <ledger> --records <records> \\
        [--threshold 0.5] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from jevrag.eval.calibration import calibration_summary  # noqa: E402
from jevrag.eval.cost import format_cost_table, summarize_cost  # noqa: E402

#: Ledger keys every row must carry (writer contract in jevrag/backends/shadow.py).
REQUIRED_LEDGER_KEYS = {
    "seq", "ts", "wrapper", "backend", "questions",
    "result", "confidence", "state_summary",
}


def load_ledger(path: Path) -> list[dict]:
    """Load ledger rows, loudly — a corrupt observation log is a real finding."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: bad JSON: {e}")
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: row is not an object")
            missing = REQUIRED_LEDGER_KEYS - set(row)
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: row missing ledger keys {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no ledger rows found")
    return rows


def confidence_values(rows: list[dict]) -> tuple[list[float], int]:
    """Flatten scalar confidences; multi-question dict rows are refused loudly.

    Returns (values, n_nan). Dict-valued confidences would need a per-question
    breakdown this minimal report doesn't do — refusing beats silently
    averaging things that were never one number.
    """
    values: list[float] = []
    n_nan = 0
    for row in rows:
        conf = row["confidence"]
        if isinstance(conf, dict):
            raise ValueError(
                f"ledger seq={row.get('seq')}: multi-question dict confidence "
                "needs a per-question report — not supported by this script")
        values.append(float(conf))
    arr = np.asarray(values, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    return [float(v) for v in arr[~np.isnan(arr)]], n_nan


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None,
                "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def histogram(values: list[float], bins: int = 10) -> list[str]:
    """Text decile histogram — distribution shape at a glance."""
    if not values:
        return ["(no values)"]
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    widest = max(1, int(counts.max()))
    lines = []
    for i in range(bins):
        bar = "#" * max(1, int(40 * counts[i] / widest)) if counts[i] else ""
        lines.append(f"  [{edges[i]:.1f},{edges[i + 1]:.1f}) "
                     f"{int(counts[i]):>5} {bar}")
    return lines


def summarize(rows: list[dict]) -> dict:
    """Pure ledger summary (everything the printed report needs)."""
    backends: dict[str, int] = {}
    questions: dict[str, dict] = {}
    for row in rows:
        backends[row["backend"]] = backends.get(row["backend"], 0) + 1
        for q in row["questions"]:
            entry = questions.setdefault(q["name"], {"kind": q["kind"], "n": 0})
            entry["n"] += 1
    values, n_nan = confidence_values(rows)
    # Cost via the shared machinery: ledger rows already carry the same
    # per-call latency/token fields decision-path records do.
    pseudo_records = [
        {"latency_ms": r.get("latency_ms"), "input_tokens": r.get("input_tokens")}
        for r in rows
    ]
    return {
        "n_calls": len(rows),
        "n_wrappers": len({r["wrapper"] for r in rows}),
        "backends": backends,
        "questions": questions,
        "confidence": distribution(values),
        "n_nan_confidence": n_nan,
        "histogram": histogram(values),
        "cost": summarize_cost(pseudo_records),
    }


def load_labeled_records(path: Path) -> list[dict]:
    """Minimal labeled-records loader: needs question-agnostic confidence/correct/action."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = {"confidence", "correct", "action"} - set(rec)
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: record missing {sorted(missing)}")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path}: no records found")
    return rows


def print_report(summary: dict) -> None:
    print("=" * 72)
    print("JevRAG - shadow ledger report")
    print("=" * 72)
    print(f"\ncalls observed: {summary['n_calls']}  "
          f"(wrappers: {summary['n_wrappers']})")
    print("per backend:")
    for name, n in sorted(summary["backends"].items()):
        print(f"  {name:<30} {n}")
    print("per question:")
    for name, q in sorted(summary["questions"].items()):
        print(f"  {name:<30} kind={q['kind']:<10} n={q['n']}")
    c = summary["confidence"]
    print("\nchallenger confidence distribution:")

    def f(v, nd=4):
        return "—" if v is None else f"{v:.{nd}f}"

    print(f"  n={c['n']} (NaN excluded: {summary['n_nan_confidence']})  "
          f"mean={f(c['mean'])} p50={f(c['p50'])} p95={f(c['p95'])} "
          f"min={f(c['min'])} max={f(c['max'])}")
    print("\n".join(summary["histogram"]))
    print("\n--- Challenger cost (shared cost harness over ledger rows) ---")
    print(format_cost_table(summary["cost"]))
    print("\nnote: observation only — none of the above affected production.")


def print_calibration_join(rows: list[dict], records: list[dict],
                           threshold: float) -> dict:
    """Positional join (ledger[i] <-> records[i]) + calibration + agreement."""
    if len(rows) != len(records):
        raise ValueError(
            f"positional join refused: ledger has {len(rows)} rows but "
            f"records file has {len(records)} — the ledger was not produced "
            "by replaying this file in order")
    conf = np.array([float(r["confidence"]) for r in rows])
    corr = np.array([float(r["correct"]) for r in records], dtype=float)
    cal = calibration_summary(conf, corr)
    prod_action = np.array([r["action"] for r in records])
    chall_action = np.where(conf >= threshold, "pass", "abstain")
    agree = prod_action == chall_action
    prod_pass = prod_action == "pass"
    chall_pass = chall_action == "pass"
    prod_em = float(corr[prod_pass].mean()) if prod_pass.any() else float("nan")
    chall_em = float(corr[chall_pass].mean()) if chall_pass.any() else float("nan")
    print("\n--- Calibration join (positional: ledger[i] <-> records[i]) ---")
    print(f"  challenger AURC {cal['aurc']:.4f} (oracle {cal['oracle_aurc']:.4f}, "
          f"random {cal['random_aurc']:.4f})")
    print(f"  ECE {cal['ece']:.4f}  Brier {cal['brier']:.4f} "
          f"(skill {cal['brier_skill']:.4f})  base rate {cal['base_rate_correct']:.4f}")
    print(f"\n--- What-if agreement @ threshold {threshold:.2f} "
          "(challenger vs production) ---")
    print(f"  agreement rate: {float(agree.mean()):.4f} "
          f"({int(agree.sum())}/{len(rows)})")
    print(f"  both pass: {int((prod_pass & chall_pass).sum())}  "
          f"both abstain: {int((~prod_pass & ~chall_pass).sum())}")
    print(f"  challenger-only pass: {int((~prod_pass & chall_pass).sum())}  "
          f"challenger-only abstain: {int((prod_pass & ~chall_pass).sum())}")
    print(f"  passed-subset EM: challenger {chall_em:.4f} "
          f"(n={int(chall_pass.sum())}) vs production {prod_em:.4f} "
          f"(n={int(prod_pass.sum())}) vs overall {float(corr.mean()):.4f}")
    print("  note: 'correct' is the EM proxy, with its disclosed limits — "
          "same caveat as every answer-abstain number.")
    return {
        "threshold": threshold,
        "calibration": cal,
        "agreement_rate": float(agree.mean()),
        "challenger_passed_em": chall_em,
        "production_passed_em": prod_em,
        "overall_em": float(corr.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True, help="shadow ledger JSONL")
    ap.add_argument("--records", default=None,
                    help="labeled records file the ledger replayed (enables "
                         "the calibration + what-if join)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="pass/abstain threshold for the what-if agreement "
                         "(default 0.5)")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args(argv)

    try:
        rows = load_ledger(Path(args.ledger))
        summary = summarize(rows)
        print_report(summary)
        report = {"ledger": str(args.ledger), **summary,
                  "cost": summary["cost"],
                  "histogram": summary["histogram"]}
        if args.records:
            records = load_labeled_records(Path(args.records))
            report["join"] = print_calibration_join(rows, records,
                                                    args.threshold)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
