"""README-table aggregator (packet V03-A6, readout fix V03-A11): result files
-> markdown rows.

Reads one or more result files written by ``scripts/run_suite.py`` (header +
rows, per ``docs/suite_records_contract.md``) and emits the markdown rows for the
README "Results at a glance" table: gated vs fixed vs CRC, both CRC risk
numbers, cost. Metrics are recomputed from the rows with
``run_suite.split_metrics`` (single source of truth) — EXCEPT the gated_crc
risks: stored gated_crc confidences already carry the deploy-time shift, so
re-jittering them with ``crc_readout`` fails on sub-eps ties (packet V03-A11).
The gated_crc row is therefore read as the live decision (recorded confidence
>= recorded threshold, no second jitter); the ``gated`` readout is unchanged.
Abstain thresholds come from the sibling ``.trace.json`` when present; a
stale/None gated_crc threshold (halted run) is rebuilt deterministically from
the file's calibrate gated rows and verified against the live served count
before any CRC row prints. Nothing is invented: unavailable cells print ``n/a``
with the reason in the notes column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevrag.pipeline import JOINT_ABSTAIN  # noqa: E402
from scripts.run_suite import (  # noqa: E402
    build_crc_specs,
    calibrate_rule,
    load_result_rows,
    split_metrics,
)

#: Notes-column provenance labels for the gated_crc readout.
RECORDED_LABEL = "live decision at recorded threshold"
REBUILT_LABEL = ("threshold rebuilt from calibrate rows "
                 "(run halted before trace)")


def fmt(value, ndigits: int = 3) -> str:
    """Numbers fixed-point; None -> n/a; never invents."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and value != value:  # NaN
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{ndigits}f}"


def load_thresholds(results_path: str) -> dict:
    """Abstain thresholds per arm from the sibling trace file, else {}."""
    trace_path = (results_path[:-len(".jsonl")] + ".trace.json"
                  if results_path.endswith(".jsonl") else None)
    if trace_path and Path(trace_path).exists():
        try:
            return (json.loads(open(trace_path, encoding="utf-8").read())
                    .get("abst_thresholds") or {})
        except (ValueError, AttributeError):
            return {}
    return {}


def recorded_abstain_confs(rows: list[dict]):
    """(confs, correct, n_missing): recorded abstain confidences of CRC rows.

    Rows missing the signal are counted, never invented: they stay in the
    coverage denominator as unanswered (the ``crc_readout`` NaN accounting).
    Error rows are excluded here (shown separately as ``n errors``).
    """
    confs, correct, missing = [], [], 0
    for r in rows:
        if r.get("outcome") == "error":
            continue
        conf = ((r.get("abstain") or {}).get("record") or {}).get(
            "confidence")
        if conf is None:
            missing += 1
            continue
        try:
            confs.append(float(conf))
        except (TypeError, ValueError):
            missing += 1
            continue
        correct.append(1.0 if r.get("em") else 0.0)
    return confs, correct, missing


def live_crc_readout(rows: list[dict], threshold: float):
    """Plain readout of the live gated_crc decision (packet V03-A11).

    The arm answered iff recorded confidence >= threshold — the recorded
    value already carries the deploy-time shift, so there is no second
    jitter (re-jittering fails on sub-eps ties: two rows tied at the raw
    level differ by < 2*TIE_BREAK_EPS after the live shift). ``>=`` matches
    the live arm (``action_for`` passes at equality, as does the adapter's
    decision-space comparison). threshold=+inf (abstain-all) answers nothing.
    Returns (coverage, selective_risk, marginal_risk, n, n_missing_signal);
    zero answered -> risks None (n/a, never 0.0).
    """
    confs, correct, missing = recorded_abstain_confs(rows)
    n = len(confs) + missing
    if not n:
        return None, None, None, 0, 0
    answered = [c >= threshold for c in confs]
    n_ans = sum(answered)
    if not n_ans:
        return 0.0, None, None, n, missing
    n_wrong_ans = sum(1 for a, e in zip(answered, correct) if a and not e)
    sel = n_wrong_ans / n_ans
    cov = n_ans / n
    return cov, sel, cov * sel, n, missing


def resolve_gated_crc_threshold(results_path: str, all_rows: list[dict],
                                group_rows: list[dict], alpha: float):
    """Threshold + provenance for one gated_crc row group.

    The recorded trace value wins. Otherwise the threshold is rebuilt
    deterministically from the file's calibrate-half gated rows with the
    runner's own spec/threshold builders (no re-implemented math), and the
    rebuilt threshold must reproduce the arm's live served count (rows with
    recorded confidence >= threshold vs rows with outcome == answer).
    Returns (threshold, label, ok, detail); ok=False means print no CRC row.
    """
    recorded = load_thresholds(results_path).get("gated_crc")
    if recorded is not None:
        return float(recorded), RECORDED_LABEL, True, ""
    cal = [r for r in all_rows
           if r.get("split") == "calibrate" and r.get("arm") == "gated"
           and r.get("outcome") != "error"
           and r.get("sufficiency") and r.get("abstain")]
    if not cal:
        return None, "", False, ("no recorded threshold and no usable "
                                  "calibrate gated rows to rebuild from")
    try:
        spec = build_crc_specs(cal, alpha)[JOINT_ABSTAIN]
        rule = calibrate_rule(JOINT_ABSTAIN, spec)
    except ValueError as exc:
        return None, "", False, f"threshold rebuild failed: {exc}"
    thr = float("inf") if rule.abstain_all else float(rule.threshold)
    confs, _, missing = recorded_abstain_confs(group_rows)
    if missing:
        return None, "", False, (f"rebuild unverifiable: {missing} rows "
                                  "missing the recorded signal")
    live_served = sum(1 for r in group_rows
                      if r.get("outcome") == "answer")
    pred_served = sum(1 for c in confs if c >= thr)
    if pred_served != live_served:
        return None, "", False, (f"rebuilt threshold predicts {pred_served} "
                                  f"served but rows show {live_served}; no "
                                  "CRC row")
    return thr, REBUILT_LABEL, True, ""


def table_for_results(results_path: str) -> list[dict]:
    """One result dict per (arm, split) found in the file."""
    header, rows = load_result_rows(results_path)
    dataset = (header or {}).get("dataset", Path(results_path).stem)
    try:
        alpha = float((header or {}).get("alpha", 0.3))
    except (TypeError, ValueError):
        alpha = 0.3
    thresholds = load_thresholds(results_path)
    by_arm_split: dict = {}
    last: dict = {}
    for r in rows:
        last[(r.get("arm"), r.get("split"), r.get("question_id"))] = r   # last row wins
    for (arm, split, _), r in last.items():
        by_arm_split.setdefault((arm, split), []).append(r)
    all_rows = list(last.values())
    split_counts: dict = {}
    for (arm, split), v in by_arm_split.items():
        split_counts.setdefault(arm, {})[split] = len(v)
    out = []
    for (arm, split), v in sorted(by_arm_split.items()):
        m = split_metrics(v, thresholds.get(arm))
        partial = ""
        if split == "report":
            n_cal = split_counts.get(arm, {}).get("calibrate", len(v))
            if len(v) < n_cal:
                partial = (f"partial run: report n={len(v)} < "
                           f"calibrate n={n_cal}")
        if arm == "gated_crc":
            thr, label, ok, detail = resolve_gated_crc_threshold(
                results_path, all_rows, v, alpha)
            if ok:
                cov, sel, marg, _, missing = live_crc_readout(v, thr)
                m = dict(m, coverage=cov, selective_risk=sel,
                         marginal_risk=marg, risk_source=label)
                if missing:
                    m["risk_source"] = (
                        label + f"; {missing} rows missing abstain signal "
                        "counted unanswered")
            else:
                m = dict(m, coverage=None, selective_risk=None,
                         marginal_risk=None,
                         risk_source=f"no CRC row: {detail}")
        out.append({"dataset": dataset, "arm": arm, "split": split,
                    "metrics": m,
                    "threshold_nota": arm not in thresholds,
                    "partial": partial})
    return out


def markdown_rows(tables: list[list[dict]]) -> str:
    """Full markdown table (header + one row per arm/split)."""
    lines = ["| dataset | arm | split | n | served% | EM | F1 | coverage | "
             "sel_risk P(wrong\\|ans) | marg_risk P(ans&wrong) | AURC | ECE | "
             "rounds | $ | notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for table in tables:
        for t in table:
            m = t["metrics"]
            n = m["n"]
            served_pct = (100.0 * m["n_served"] / n) if n else None
            notes = []
            if t["threshold_nota"]:
                notes.append("no recorded threshold")
            if t.get("partial"):
                notes.append(t["partial"])
            if m.get("risk_source") not in ("crc_readout",
                                            "definitional_no_gate"):
                notes.append(str(m.get("risk_source", "n/a")))
            if m["n_errors"]:
                notes.append(f"{m['n_errors']} errors")
            lines.append(
                f"| {t['dataset']} | {t['arm']} | {t['split']} | {n} | "
                f"{fmt(served_pct, 1)} | {m['em_overall']}/{n} | "
                f"{fmt(m['f1_mean'])} | {fmt(m.get('coverage'))} | "
                f"{fmt(m.get('selective_risk'))} | "
                f"{fmt(m.get('marginal_risk'))} | {fmt(m.get('aurc'))} | "
                f"{fmt(m.get('ece'))} | {m['rounds']} | "
                f"{fmt(m['cost_usd'], 4)} | {'; '.join(notes) or '—'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+",
                    help="*_suite_results.jsonl file(s)")
    args = ap.parse_args()
    print(markdown_rows([table_for_results(p) for p in args.results]),
          end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
