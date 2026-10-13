"""Single-source CRC shift test (packet V03-A10, tracker v0.3 MUST-4).

Calibrate CRC on one source dataset's calibrate-half gated rows (v0.3: HotpotQA,
``--limit 150 --with-crc``), then apply those calibrated rule(s) to each target
dataset's REPORT-half gated rows and report coverage and selective risk against
the alpha target. Offline only: reads result files, recomputes thresholds with
the same ``run_suite`` builders the runner uses, applies them with the
tie-broken ``crc_readout``. No Jev calls, no network.

What this is: a single-source shift across closed-pool multi-hop datasets
(shift in difficulty more than in task type). The source rule is extracted from
the SOURCE file's calibrate gated rows only — target calibrate rows never feed
any figure (calibrate/report isolation).

Rule applicability, stated: the ``answer_abstain`` rule maps to a real operating
characteristic (the terminal pass/abstain verdict read at the source's
decision-space threshold). The ``sufficiency`` rule is reported as a pure
signal-shift diagnostic: in a live run it would change stopping (hence the
rows themselves), which an offline re-readout cannot reproduce.

Full-file guard: every input must carry the ``gated_crc`` arm. Checkpoints run
gated+fixed only (a checkpoint's few-row calibration must never seed report
rows — the A6 mixture lesson), so a file without ``gated_crc`` is refused as a
checkpoint offered where a full file is required. Exit 0 on success, 2 on
missing files / missing CRC spec / checkpoint input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevrag.eval.crc import crc_readout  # noqa: E402
from jevrag.pipeline import JOINT_ABSTAIN, JOINT_SUFFICIENCY  # noqa: E402
from scripts.run_suite import (  # noqa: E402
    build_crc_specs,
    calibrate_rule,
    crc_spec_sha,
    load_result_rows,
)
from scripts.suite_table import (  # noqa: E402
    fmt,
    live_crc_readout,
    resolve_gated_crc_threshold,
)

JOINTS = (JOINT_SUFFICIENCY, JOINT_ABSTAIN)
EXIT_DATA = 2

#: Per-row signal each CRC rule reads (see run_suite.build_crc_specs).
SIGNAL_BLOCK = {JOINT_SUFFICIENCY: "sufficiency", JOINT_ABSTAIN: "abstain"}


def err(msg: str) -> int:
    print(f"shift_test: {msg}", file=sys.stderr)
    return EXIT_DATA


def dedupe(rows: list[dict]) -> list[dict]:
    """Last row wins per (question_id, arm), mirroring run_suite metrics."""
    latest: dict = {}
    for r in rows:
        latest[(str(r.get("question_id")), str(r.get("arm")))] = r
    return list(latest.values())


def dataset_of(header: dict | None, path: str) -> str:
    if header and header.get("dataset"):
        return str(header["dataset"])
    return Path(path).stem


def require_full_file(path: str, header: dict | None,
                       rows: list[dict]) -> tuple[str, list[str]] | int:
    """Full-file guard: refuse checkpoint files (no gated_crc arm)."""
    arms = sorted({str(r.get("arm")) for r in rows})
    if "gated_crc" not in arms:
        return err(
            f"checkpoint file offered where a full file is required: {path} "
            f"has arms {arms} but no 'gated_crc' arm (checkpoints run "
            "gated+fixed only; need the full --limit 150 --with-crc file).")
    return dataset_of(header, path), arms


def usable_cal_gated(rows: list[dict]) -> list[dict]:
    """Source calibration set: calibrate-half gated rows carrying both joint
    blocks (same filter run_suite.resolve_abst_thresholds uses)."""
    return [r for r in rows
            if r.get("split") == "calibrate" and r.get("arm") == "gated"
            and r.get("outcome") != "error"
            and r.get("sufficiency") and r.get("abstain")]


def signal_rows(rows: list[dict], joint: str):
    """(scores, correct, keys, n_skipped) for one joint on report gated rows.

    Rows missing the joint's block are skipped and counted, never invented.
    """
    block = SIGNAL_BLOCK[joint]
    scores, correct, keys = [], [], []
    skipped = 0
    for r in rows:
        b = r.get(block) or {}
        conf = (b.get("record") or {}).get("confidence")
        if conf is None:
            skipped += 1
            continue
        try:
            scores.append(float(conf))
        except (TypeError, ValueError):
            skipped += 1
            continue
        correct.append(1.0 if r.get("em") else 0.0)
        keys.append(str(r.get("question_id")))
    return scores, correct, keys, skipped


def readout(rule, scores, correct, keys) -> tuple:
    """(coverage, selective_risk, marginal_risk); abstain-all -> (0.0, n/a)."""
    if rule.abstain_all:
        return 0.0, None, None
    if not scores:
        return None, None, None
    ro = crc_readout(scores, correct, keys, float(rule.threshold))
    sel = ro.get("risk")
    if sel is not None and sel != sel:  # NaN -> n/a, never 0.0
        sel = None
    marg = ro.get("marginal_risk")
    if marg is not None and marg != marg:
        marg = None
    return ro.get("coverage"), sel, marg


def indistrib_context(path: str, rows: list[dict], alpha: float):
    """Target's own in-distribution CRC row: gated_crc/report read as the live
    decision at the recorded threshold (recorded confidence >= threshold, no
    second jitter — same readout as suite_table so both tables agree). A
    stale/None trace threshold is rebuilt from the file's calibrate gated
    rows and verified against the live served count. Returns (info|None,
    note); None means print the n/a row with the reason.
    """
    crc_rep = [r for r in rows
               if r.get("arm") == "gated_crc" and r.get("split") == "report"
               and r.get("outcome") != "error"]
    if not crc_rep:
        return None, "no gated_crc/report rows in file"
    thr, label, ok, detail = resolve_gated_crc_threshold(
        path, rows, crc_rep, alpha)
    if not ok:
        return None, detail
    cov, sel, marg, _, missing = live_crc_readout(crc_rep, thr)
    info = {"n": len(crc_rep),
            "n_served": sum(1 for r in crc_rep
                            if r.get("outcome") == "answer"),
            "coverage": cov, "selective_risk": sel,
            "marginal_risk": marg, "n_missing": missing,
            "n_errors": sum(1 for r in rows
                            if r.get("arm") == "gated_crc"
                            and r.get("split") == "report"
                            and r.get("outcome") == "error")}
    return info, label


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Apply a source dataset's CRC calibration to target "
                    "report-half gated rows (offline shift test).")
    ap.add_argument("--source", required=True,
                    help="source result file (calibration source, e.g. "
                         "hotpotqa.jsonl)")
    ap.add_argument("--targets", nargs="+", required=True,
                    help="one or more target result files")
    ap.add_argument("--alpha", type=float, default=None,
                    help="shift alpha target (default: source header alpha, "
                         "else 0.3)")
    ap.add_argument("--joint", choices=[*JOINTS, "all"], default="all",
                    help="which calibrated rule(s) to apply")
    args = ap.parse_args(argv)

    try:
        src_header, src_rows = load_result_rows(args.source)
    except (FileNotFoundError, OSError):
        return err(f"missing file: {args.source}")
    except ValueError as exc:
        return err(f"unreadable source file {args.source}: {exc}")
    if not Path(args.source).exists():
        return err(f"missing file: {args.source}")
    src_rows = dedupe(src_rows)
    full = require_full_file(args.source, src_header, src_rows)
    if isinstance(full, int):
        return full
    src_name, _ = full
    alpha = args.alpha
    if alpha is None:
        try:
            alpha = float((src_header or {})["alpha"])
        except (KeyError, TypeError, ValueError):
            alpha = 0.3
    if not 0.0 < alpha < 1.0:
        return err(f"alpha must be in (0, 1), got {args.alpha}")

    cal = usable_cal_gated(src_rows)
    if not cal:
        return err(f"missing CRC spec: {args.source} has no usable "
                   "calibrate-half gated rows (need split=calibrate, "
                   "arm=gated, non-error, with sufficiency+abstain blocks).")
    try:
        specs = build_crc_specs(cal, alpha)
    except ValueError as exc:
        return err(f"missing CRC spec: {args.source}: {exc}")
    joints = list(JOINTS) if args.joint == "all" else [args.joint]
    rules: dict = {}
    for joint in joints:
        try:
            rules[joint] = calibrate_rule(joint, specs[joint])
        except ValueError as exc:
            print(f"shift_test: note: no {joint} rule from {args.source} "
                  f"({exc}); that rule is skipped.", file=sys.stderr)
    if not rules:
        return err(f"missing CRC spec: {args.source}: no joint calibrated.")
    try:
        sha = crc_spec_sha({j: specs[j] for j in rules},
                           {j: rules[j] for j in rules}, alpha)
    except Exception:  # noqa: BLE001 — sha is provenance, never fatal
        sha = "n/a"

    targets: list[tuple[str, str, list[dict]]] = []
    for t in args.targets:
        try:
            t_header, t_rows = load_result_rows(t)
        except (FileNotFoundError, OSError):
            return err(f"missing file: {t}")
        except ValueError as exc:
            return err(f"unreadable target file {t}: {exc}")
        if not Path(t).exists():
            return err(f"missing file: {t}")
        t_rows = dedupe(t_rows)
        full_t = require_full_file(t, t_header, t_rows)
        if isinstance(full_t, int):
            return full_t
        t_name, _ = full_t
        try:
            t_alpha = float((t_header or {}).get("alpha", alpha))
        except (TypeError, ValueError):
            t_alpha = alpha
        targets.append((t, t_name, t_rows, t_alpha))

    print(f"# CRC shift test (offline, no Jev calls): {src_name} -> "
          f"{', '.join(n for _, n, _, _ in targets)}")
    print("# Single-source shift across closed-pool multi-hop datasets "
          "(shift in difficulty more than in task type).")
    print(f"# Source calibration: {args.source} n_cal_gated={len(cal)} "
          f"alpha={alpha} spec_sha={sha[:12]}...")
    for joint, rule in rules.items():
        thr = "inf (abstain-all)" if rule.abstain_all \
            else f"{rule.threshold:.6f}"
        print(f"#   rule {joint}: threshold={thr} "
              f"threshold_score={getattr(rule, 'threshold_score', float('nan')):.6f} "
              f"k_hat={rule.k_hat} n_cal={rule.n_cal}")
    if JOINT_SUFFICIENCY in rules:
        print("# Note: the sufficiency row is a signal-shift diagnostic only "
              "(live stopping would change the rows); only answer_abstain "
              "maps to an operating characteristic.")
    print("| dataset | n_report | rule | coverage | sel_risk P(wrong\\|ans) | "
          "marg_risk P(ans&wrong) | alpha | notes |")
    print("|---|---|---|---|---|---|---|---|")
    for t, t_name, t_rows, t_alpha in targets:
        rep = [r for r in t_rows
               if r.get("split") == "report" and r.get("arm") == "gated"
               and r.get("outcome") != "error"]
        n_err = sum(1 for r in t_rows
                    if r.get("split") == "report" and r.get("arm") == "gated"
                    and r.get("outcome") == "error")
        for joint, rule in rules.items():
            scores, correct, keys, skipped = signal_rows(rep, joint)
            cov, sel, marg = readout(rule, scores, correct, keys)
            notes = []
            if skipped:
                notes.append(f"{skipped} rows missing "
                             f"{SIGNAL_BLOCK[joint]} signal skipped")
            if n_err:
                notes.append(f"{n_err} error rows excluded")
            if rule.abstain_all:
                notes.append("source rule abstains on everything")
            print(f"| {t_name} | {len(rep)} "
                  f"| shift:{joint} ({src_name}-calibrated) "
                  f"| {fmt(cov)} | {fmt(sel)} | {fmt(marg)} | {fmt(alpha)} "
                  f"| {'; '.join(notes) or '—'} |")
        m, note = indistrib_context(t, t_rows, t_alpha)
        if m is None:
            print(f"| {t_name} | n/a | gated_crc-infile (own calibration) "
                  f"| n/a | n/a | n/a | {fmt(t_alpha)} | {note} |")
        else:
            extra = f"{m['n_served']}/{m['n']} served; {note}"
            if m.get("n_missing"):
                extra += (f"; {m['n_missing']} rows missing abstain signal "
                          "counted unanswered")
            if m.get("n_errors"):
                extra += f"; {m['n_errors']} errors"
            print(f"| {t_name} | {m['n']} "
                  f"| gated_crc-infile (own calibration) "
                  f"| {fmt(m.get('coverage'))} "
                  f"| {fmt(m.get('selective_risk'))} "
                  f"| {fmt(m.get('marginal_risk'))} | {fmt(t_alpha)} "
                  f"| {extra} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
