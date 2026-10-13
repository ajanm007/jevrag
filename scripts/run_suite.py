"""Generic closed-pool suite runner (packet V03-A6, tracker item I-5 core).

Reads one ``<dataset>_suite_records.jsonl`` per
``tempnotes/SUITE_RECORDS_CONTRACT.md`` (header + question rows; the Cline
seat produces those files in parallel — this runner codes to the contract,
tested against a fixture, and does not import their loader modules).

Modes: ``per_question`` (a BM25 over that question's own paragraphs) and
``shared`` (one pool file, one index across questions). Arms: ``gated``,
``fixed``, ``gated_crc`` (report half only; CRC calibrate-on-calibrate /
evaluate-on-report at alpha 0.3, untuned). One PipelineConfig per arm per
question set — never shared across arms (the A4 free-ride lesson).

Resume-safe (rows appended as they go; restarts skip finished (question,
arm) pairs), cost-gated (``--limit`` checkpoint prints per-arm $/question
and a full-n projection including the CRC re-run; the remaining run is
refused past ``--budget`` without ``--yes``), fail-closed (first 402 or any
non-retryable generator error halts with rows-so-far kept).

Money is printed at the dashboard-implied rate below, not at the pipeline
module's 0.042 launch-pricing constant — the two differ and both are stated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402
from scripts.run_pipeline_docbench_pilot import (  # noqa: E402
    GeneratorHalted,
    OverwriteRefused,
    checked_out_paths,
    fail_closed_answer_fn,
    stage_split,
)

from jevrag._rag_gate import rag_gate_root  # noqa: E402
from jevrag.benchmarks.docbench import build_bm25_index  # noqa: E402
from jevrag.benchmarks.hotpotqa import exact_match, f1_score  # noqa: E402
from jevrag.decision import JevDecision  # noqa: E402
from jevrag.eval.calibration import expected_calibration_error  # noqa: E402
from jevrag.eval.crc import crc_readout  # noqa: E402
from jevrag.pipeline import (  # noqa: E402
    JOINT_ABSTAIN,
    JOINT_SUFFICIENCY,
    CrcSpec,
    PipelineConfig,
    answer_question,
    answer_question_fixed,
    build_pipeline_from_chunks,
    calibrate_rule,
)
from jevrag._vendor import selective as _selective  # noqa: E402

#: Dollars per million Jev input tokens, dashboard-implied (Anmol, 2026-10-01;
#: TypeSafe stats lag). Differs from jevrag.decision's 0.042 launch-pricing
#: constant on purpose; projections and prints here use this one, stated.
SUITE_USD_PER_MTOK = 0.041

GENERATOR_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ARMS_CALIBRATE = ("gated", "fixed")
ARMS_REPORT = ("gated", "fixed", "gated_crc")


def suite_cost_usd(input_tokens: int) -> float:
    """Jev dollars at the dashboard-implied suite rate."""
    return input_tokens / 1_000_000 * SUITE_USD_PER_MTOK


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file (result-header provenance)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_alias_aware(prediction: str, answer: str,
                      aliases: list[str]) -> dict:
    """EM/F1 against answer + aliases, one evaluator path (packet A5 lesson).

    Same vendored ``exact_match``/``f1_score`` the HotpotQA path uses (and
    the same ANY-match / BEST-F1 semantics as closed_pool.score_with_aliases,
    reimplemented here in ~10 lines so this runner does not import the
    Cline seat's in-flight loader module). Returns {"em", "f1", "matched"}.
    """
    gold: list[str] = []
    for a in [answer, *(aliases or [])]:
        key = str(a).strip()
        if key and key not in gold:
            gold.append(key)
    if not gold:
        raise ValueError("suite row has no gold answer or aliases")
    pred = prediction or ""
    matched = next((g for g in gold if exact_match(pred, g)), None)
    return {"em": matched is not None,
            "f1": max((f1_score(pred, g) for g in gold), default=0.0),
            "matched": matched}


def load_suite_records(path: str | Path) -> tuple[dict, list[dict]]:
    """Header + rows with contract validation (dangling ids are build errors).

    Assumptions about the contract (stated; the file itself is not edited):
    ids unique; split in {calibrate, report}; per_question rows carry a
    non-empty ``paragraphs`` list of {id, title, text}; shared rows may omit
    paragraphs and the header carries ``pool_file``; ``aliases`` defaults to
    []; ``hops``/``type`` optional.
    """
    header: dict | None = None
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if lineno == 1 and "__header__" in obj:
                header = dict(obj["__header__"])
                continue
            rows.append(obj)
    if header is None:
        raise ValueError(f"{path}: first line must be the __header__ object")
    if header.get("pool_mode") not in ("per_question", "shared"):
        raise ValueError(f"{path}: bad pool_mode {header.get('pool_mode')!r}")
    seen_ids: set[str] = set()
    for r in rows:
        for key in ("id", "question", "answer", "split"):
            if key not in r:
                raise ValueError(f"{path}: row missing {key!r}: "
                                 f"{str(r)[:100]}")
        if r["id"] in seen_ids:
            raise ValueError(f"{path}: duplicate id {r['id']!r}")
        if r["split"] not in ("calibrate", "report"):
            raise ValueError(f"{path}: row {r['id']} bad split {r['split']!r}")
        seen_ids.add(str(r["id"]))
        if header["pool_mode"] == "per_question":
            paras = r.get("paragraphs") or []
            if not paras:
                raise ValueError(
                    f"{path}: per_question row {r['id']} has no paragraphs")
            pids = {str(p["id"]) for p in paras}
            dangling = [s for s in (r.get("supporting_ids") or [])
                        if str(s) not in pids]
            if dangling or not (r.get("supporting_ids") or []):
                raise ValueError(
                    f"{path}: row {r['id']} bad supporting_ids "
                    f"(dangling={dangling}) — a build error, not a warning")
    return header, rows


def load_pool(pool_file: str | Path) -> list[dict]:
    """Shared-mode pool: [{id, title, text}] with contract validation."""
    pool = [json.loads(l) for l in open(pool_file, encoding="utf-8")
            if l.strip()]
    for i, p in enumerate(pool):
        if not isinstance(p, dict) or not all(k in p for k in
                                              ("id", "title", "text")):
            raise ValueError(f"{pool_file}: pool row {i} must carry "
                             "id/title/text")
    if not pool:
        raise ValueError(f"{pool_file}: empty pool")
    return [{"id": str(p["id"]), "title": str(p["title"]),
             "text": str(p["text"])} for p in pool]


def load_result_rows(path: str | Path) -> tuple[dict | None, list[dict]]:
    """Existing results header + rows (resume); missing file -> (None, [])."""
    p = Path(path)
    if not p.exists():
        return None, []
    header, rows = None, []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "__header__" in obj and header is None:
                header = obj["__header__"]
            else:
                rows.append(obj)
    return header, rows


def crc_spec_sha(specs: dict, rules: dict, alpha: float) -> str:
    """Identity of one CRC calibration (packet V03-A7 §1).

    sha256 over the calibrate-half keys, scores, labels, alpha and the
    derived thresholds. Stored on every gated_crc row; on resume a
    gated_crc row counts as finished only if its sha matches the current
    calibration, so a checkpoint's few-row specs can never silently mix
    with the full run's operating point.
    """
    canonical = {
        "alpha": alpha,
        "joints": {
            joint: {
                "keys": list(spec.keys),
                "scores": [float(x) for x in spec.scores],
                "correct": [float(x) for x in spec.correct],
                "threshold": (None if rules[joint].abstain_all
                              else float(rules[joint].threshold)),
                "abstain_all": bool(rules[joint].abstain_all),
            }
            for joint, spec in sorted(specs.items())
        },
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


def resolve_grid(header: dict, rows: list[dict],
                 k_arg: int | None, cap_arg: int | None) -> dict:
    """Per-dataset retrieval grid (packet V03-A7 §2).

    Small per-question pools saturate under the old k=5/cap=30 global (a
    10-paragraph pool is fully judged in round 1 — A4's degeneracy again),
    so defaults derive from the records: per_question with median pool ≤12
    → k=2/cap=12 (pools 4/8/12); ≤24 → k=3/cap=18 (6/12/18); shared keeps
    k=5/cap=30 (10/20/30). Explicit flags always win; the values actually
    used land in the result header.
    """
    if k_arg is not None or cap_arg is not None:
        auto_k, auto_cap = None, None
    else:
        auto_k, auto_cap = None, None
    if header.get("pool_mode") == "shared":
        auto_k, auto_cap = 5, 30
    else:
        sizes = sorted(len(r.get("paragraphs") or []) for r in rows)
        med = sizes[len(sizes) // 2] if sizes else 0
        if med <= 12:
            auto_k, auto_cap = 2, 12
        elif med <= 24:
            auto_k, auto_cap = 3, 18
        else:
            auto_k, auto_cap = 5, 30
    return {"per_round_k": k_arg if k_arg is not None else auto_k,
            "candidate_pool_cap": cap_arg if cap_arg is not None else auto_cap,
            "grid_source": ("explicit-flags" if k_arg is not None
                            or cap_arg is not None else "header-derived")}


def degeneracy_report(rows: list[dict]) -> dict:
    """Two degeneracy fractions per arm + overall (packet V03-A7 §3).

    new_candidates==0 tracks pool growth; evidence_ids equal to the previous
    round's tracks whether the loop's kept set changes. A selector that
    keeps returning the same evidence from a growing pool trips only the
    second — judging growth alone would miss it.
    """
    out: dict = {}
    groups: dict = {"all": rows}
    for r in rows:
        groups.setdefault(f"arm:{r.get('arm')}", []).append(r)
    for name, rs in groups.items():
        later_new, later_ev = [], []
        for row in rs:
            rounds = (row.get("selection") or {}).get("rounds", [])
            for prev, cur in zip(rounds, rounds[1:]):
                later_new.append(cur)
                later_ev.append((prev, cur))
        out[name] = {
            "n_later_rounds": len(later_new),
            "frac_no_new": (sum(1 for r in later_new
                                if not r.get("new_candidates"))
                            / len(later_new)) if later_new else 0.0,
            "frac_same_evidence": (sum(
                1 for p, c in later_ev
                if list(p.get("evidence_ids", []))
                == list(c.get("evidence_ids", []))) / len(later_ev)
                if later_ev else 0.0),
        }
    return out


def plan_pairs(rows: list[dict], limit: int | None,
               with_crc: bool = True) -> list[tuple[dict, str]]:
    """Ordered (row, arm) plan: calibrate gated+fixed, then report + CRC.

    ``limit`` = first N rows per split (file order = stratified order).
    Checkpoints (limit set) run gated+fixed only unless ``--with-crc`` —
    a checkpoint's few-row CRC specs must never seed report-half rows
    (the A6 mixture lesson).
    """
    cal = [r for r in rows if r["split"] == "calibrate"]
    rep = [r for r in rows if r["split"] == "report"]
    if limit is not None:
        cal, rep = cal[:limit], rep[:limit]
    report_arms = ARMS_REPORT if (with_crc or limit is None) else (
        "gated", "fixed")
    plan = ([(r, a) for r in cal for a in ARMS_CALIBRATE]
            + [(r, a) for r in rep for a in report_arms])
    return plan


def make_arm_configs(backends: dict, answer_fn, base: dict) -> dict[str, Any]:
    """One PipelineConfig per arm (never shared — the A4 free-ride lesson)."""
    cfgs = {}
    for arm in ("gated", "fixed", "gated_crc"):
        cfgs[arm] = PipelineConfig(
            chunk_decision=backends["chunk"],
            selection_decision=backends["selection"],
            sufficiency_decision=backends["sufficiency"],
            abstain_decision=backends["abstain"],
            cache_decision=backends["cache"],
            answer_fn=answer_fn,
            memoize_selection=True,
            memoize_chunk=True,
            per_round_k=base["per_round_k"],
            candidate_pool_cap=base["candidate_pool_cap"],
            oversample=base.get("oversample", 2),
            max_rounds=base.get("max_rounds", 3),
        )
    return cfgs


def pool_chunks_for_row(row: dict, pool: list[dict] | None,
                        mode: str) -> list[dict]:
    """[{title, text}] pool for one question in either mode."""
    if mode == "per_question":
        return [{"title": str(p.get("title", p["id"])),
                 "text": str(p["text"])} for p in row["paragraphs"]]
    assert pool is not None
    return [{"title": str(p["title"]), "text": str(p["text"])} for p in pool]


def build_crc_specs(gated_cal_rows: list[dict], alpha: float) -> dict:
    """CrcSpec per joint from calibrate-half gated rows (alias-aware EM).

    Scores: sufficiency joint = gate confidence at the stopping point;
    abstain joint = check_answer confidence. Keys: stable question ids.
    """
    specs = {}
    for joint, conf_key in ((JOINT_SUFFICIENCY, ("sufficiency",)),
                            (JOINT_ABSTAIN, ("abstain",))):
        scores, correct, keys = [], [], []
        for r in gated_cal_rows:
            block = r.get(conf_key[0])
            if not block:
                continue
            conf = float(block["record"]["confidence"])
            em = bool(r.get("em", False))
            scores.append(conf)
            correct.append(1.0 if em else 0.0)
            keys.append(str(r["question_id"]))
        if len(set(keys)) != len(keys):
            raise ValueError(f"CRC keys not unique for {joint}")
        specs[joint] = CrcSpec(alpha=alpha, scores=scores, correct=correct,
                               keys=keys)
    return specs


def resolve_abst_thresholds(all_rows: list[dict], alpha: float) -> dict:
    """Readout thresholds, rebuilt deterministically on every invocation.

    A previous shape rebuilt them only when gated_crc pairs were missing,
    so identical rows metricked differently across invocations (None vs a
    fallback 0.5). Now: specs + rules from ALL calibrate gated rows on disk
    (milliseconds, offline), every time. gated_crc reads out at its own
    calibrated decision-space threshold; abstain-all (or no calibrate rows)
    yields None, i.e. risks reported n/a rather than at a wrong threshold.
    """
    from jevrag.primitives.answer_abstain import (  # noqa: E402
        DEFAULT_ABSTAIN_THRESHOLD)
    thresholds: dict[str, float | None] = {
        "gated": float(DEFAULT_ABSTAIN_THRESHOLD), "fixed": None}
    cal = [r for r in all_rows
           if r.get("split") == "calibrate" and r.get("arm") == "gated"
           and r.get("outcome") != "error"
           and r.get("sufficiency") and r.get("abstain")]
    if any(r.get("arm") == "gated_crc" for r in all_rows) and cal:
        specs = build_crc_specs(cal, alpha)
        rule = calibrate_rule(JOINT_ABSTAIN, specs[JOINT_ABSTAIN])
        thresholds["gated_crc"] = (
            None if rule.abstain_all else float(rule.threshold))
    else:
        thresholds["gated_crc"] = None
    return thresholds


def split_metrics(rows: list[dict], abst_threshold: float | None) -> dict:
    """One result object per (arm, split): counts, EM/F1, AURC, ECE, risks.

    Risks: for gated arms via ``crc_readout`` on abstain confidences at the
    arm's own operating threshold (the calibrated decision-space value for
    gated_crc, the plain default for gated — each consistent with the
    confidences it is read against); for fixed, definitional coverage=1
    numbers labeled as such (no gate signal exists). AURC/ECE need spread in
    both classes; failures yield None ("n/a"), never a fabricated number.
    """
    ok = [r for r in rows if r.get("outcome") != "error"]
    served = [r for r in ok if r["outcome"] == "answer"]
    withheld = [r for r in ok if r["outcome"] != "answer"]
    em = [bool(r.get("em", False)) for r in ok]
    m: dict = {
        "n": len(rows), "n_ok": len(ok),
        "n_served": len(served), "n_withheld": len(withheld),
        "n_errors": len(rows) - len(ok),
        "em_served": sum(1 for r in served if r.get("em")),
        "em_withheld": sum(1 for r in withheld if r.get("em")),
        "em_overall": sum(em),
        "f1_mean": (sum(float(r.get("f1", 0.0)) for r in ok) / len(ok)
                    if ok else 0.0),
        "rounds": sum(int((r.get("sufficiency") or {}).get("record", {})
                          .get("rounds_used", r.get("rounds_used", 0))
                          or 0) for r in ok),
        "jev_calls": sum(int(r.get("jev_calls", 0)) for r in ok),
        "jev_input_tokens": sum(int(r.get("jev_input_tokens", 0))
                                for r in ok),
        "sel_live": sum(int(r.get("selection_calls", 0)) for r in ok),
        "sel_hits": sum(int((r.get("selection") or {}).get("cache_hits", 0))
                        for r in ok),
        "wall_ms_mean": (sum(float(r.get("wall_ms_ask",
                                         r.get("wall_ms", 0.0)))
                             for r in ok) / len(ok) if ok else 0.0),
    }
    m["cost_usd"] = suite_cost_usd(m["jev_input_tokens"])
    confs = [float(r["abstain"]["record"]["confidence"]) for r in ok
             if r.get("abstain")]
    if confs and len(confs) == len(ok):
        try:
            m["aurc"] = float(_selective.aurc(
                __import__("numpy").asarray(confs),
                __import__("numpy").asarray(
                    [1.0 if e else 0.0 for e in em])))
        except Exception:  # noqa: BLE001 — single-class etc: n/a
            m["aurc"] = None
        try:
            m["ece"] = float(expected_calibration_error(confs, em))
        except Exception:  # noqa: BLE001 — n/a, never fabricated
            m["ece"] = None
    else:
        m["aurc"], m["ece"] = None, None
    if ok and all(r.get("abstain") for r in ok) and abst_threshold is not None \
            and abst_threshold != float("inf"):
        try:
            ro = crc_readout(
                [float(r["abstain"]["record"]["confidence"]) for r in ok],
                [1.0 if e else 0.0 for e in em],
                [str(r["question_id"]) for r in ok], float(abst_threshold))
            m["coverage"] = ro.get("coverage")
            m["selective_risk"] = ro.get("risk")  # P(wrong | answered)
            m["marginal_risk"] = ro.get("marginal_risk")  # P(ans. & wrong)
            m["risk_source"] = "crc_readout"
        except Exception as exc:  # noqa: BLE001 — n/a with reason
            m.update(coverage=None, selective_risk=None, marginal_risk=None,
                     risk_source=f"crc_readout_failed: {exc}")
    elif ok and not any(r.get("abstain") for r in ok):
        err = 1.0 - (sum(em) / len(em))
        m.update(coverage=1.0, selective_risk=err, marginal_risk=err,
                 risk_source="definitional_no_gate")
    else:
        m.update(coverage=None, selective_risk=None, marginal_risk=None,
                 risk_source="n/a")
    by_hops: dict = {}
    for r in ok:
        if r.get("hops") is None:
            continue
        by_hops.setdefault(str(r["hops"]), []).append(r)
    m["by_hops"] = {h: {"n": len(v),
                        "em": sum(1 for r in v if r.get("em")),
                        "served": sum(1 for r in v
                                      if r["outcome"] == "answer")}
                    for h, v in sorted(by_hops.items())}
    return m


def project_cost(finished: list[dict], remaining: dict[str, int]) -> dict:
    """Per-arm $/q from finished rows × remaining counts (incl. CRC re-run).

    ``remaining`` maps arm -> unfinished question count; the gated_crc
    re-run of the report half is part of the remaining by construction
    (gated_crc rows only exist unfinished). Prints belong to the caller.
    """
    proj = {}
    for arm, n_left in remaining.items():
        toks = [int(r.get("jev_input_tokens", 0)) for r in finished
                if r.get("arm") == arm]
        per_q = (sum(toks) / len(toks)) if toks else 0.0
        proj[arm] = {"per_q_tokens": per_q,
                     "n_left": n_left,
                     "tokens": per_q * n_left,
                     "usd": suite_cost_usd(int(per_q * n_left))}
    arm_toks = sum(v["tokens"] for v in proj.values())
    proj["total_usd"] = sum(v["usd"] for v in proj.values())
    proj["total_tokens"] = arm_toks
    return proj


def guard_or_resume(out_prefix: str, overwrite: bool,
                    identity: dict) -> dict[str, Path]:
    """Overwrite guard with a resume exception (packet A2 lesson, refined).

    Existing non-empty outputs are refused UNLESS they carry a header with
    the identical run identity (same records file + same config) — that is
    a resume, which appends only missing (question, arm) pairs. Anything
    else needs --overwrite. Returns resolved paths.
    """
    rows_path = REPO_ROOT / f"{out_prefix}.jsonl"
    if overwrite or not rows_path.exists() or rows_path.stat().st_size == 0:
        return checked_out_paths(out_prefix, overwrite)
    old_header, _ = load_result_rows(rows_path)
    old = old_header or {}
    keys = ("dataset", "pool_mode", "generator", "alpha", "per_round_k",
            "candidate_pool_cap", "oversample", "rounds", "records_sha256")
    if all(old.get(k) == identity.get(k) for k in keys):
        return checked_out_paths(out_prefix, True)  # resolve only
    raise OverwriteRefused(
        "refusing to overwrite non-empty outputs from a DIFFERENT run "
        f"({rows_path}) — pass --overwrite to allow it.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True,
                    help="path to <dataset>_suite_records.jsonl")
    ap.add_argument("--pool-file", default=None,
                    help="override the header's shared pool file")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="checkpoint: first N rows per split only")
    ap.add_argument("--budget", type=float, default=2.50,
                    help="refuse the remaining run past this $ projection")
    ap.add_argument("--yes", action="store_true",
                    help="proceed past a budget refusal")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--per-round-k", type=int, default=None,
                    help="override the header-derived default "
                    "(shared k=5; per_question k=2/k=3 by pool size)")
    ap.add_argument("--cap", type=int, default=None,
                    help="override the header-derived candidate cap")
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--with-crc", action="store_true",
                    help="include gated_crc in a --limit checkpoint "
                    "(default: checkpoints run gated+fixed only, so a "
                    "few-row calibration can never seed report rows)")
    ap.add_argument("--redo-stale-crc", action="store_true",
                    help="re-run gated_crc rows whose crc_spec_sha no longer "
                    "matches the current calibration (appended; metrics "
                    "take the last row per question)")
    ap.add_argument("--allow-degenerate", action="store_true")
    args = ap.parse_args()

    from openai import OpenAI
    from scripts.produce_records import make_answer_fn

    header, rows = load_suite_records(args.records)
    dataset = str(header.get("dataset", "unknown"))
    mode = str(header["pool_mode"])
    pool = None
    if mode == "shared":
        pool = load_pool(args.pool_file or header["pool_file"])
        print(f"shared pool: {len(pool)} paragraphs", flush=True)
    subsets = {"calibrate": [r for r in rows if r["split"] == "calibrate"],
               "report": [r for r in rows if r["split"] == "report"]}
    print(f"records: {dataset} {mode}: "
          f"{len(subsets['calibrate'])} calibrate + "
          f"{len(subsets['report'])} report", flush=True)

    grid = resolve_grid(header, rows, args.per_round_k, args.cap)
    print(f"grid: per_round_k={grid['per_round_k']} "
          f"cap={grid['candidate_pool_cap']} ({grid['grid_source']})",
          flush=True)

    paths = guard_or_resume(args.out_prefix, args.overwrite, {
        "dataset": str(header.get("dataset", "unknown")),
        "pool_mode": str(header.get("pool_mode", "")),
        "generator": f"openrouter:{GENERATOR_MODEL}",
        "alpha": args.alpha, "per_round_k": grid["per_round_k"],
        "candidate_pool_cap": grid["candidate_pool_cap"],
        "oversample": args.oversample,
        "rounds": args.rounds,
        "records_sha256": sha256_file(args.records),
    })
    out_path, trace_path = paths["rows"], paths["trace"]
    old_header, done_rows = load_result_rows(out_path)
    done = {(str(r.get("question_id")), str(r.get("arm")))
            for r in done_rows}
    if old_header is not None:
        print(f"resume: {len(done_rows)} finished (question, arm) pairs "
              f"already on disk", flush=True)

    plan = plan_pairs(rows, args.limit, with_crc=args.with_crc)
    if args.limit is not None and not args.with_crc:
        print("checkpoint runs gated+fixed only (gated_crc skipped: "
              "calibrate first, full run only)", flush=True)
    missing = [(r, a) for (r, a) in plan
               if (str(r["id"]), a) not in done]
    if not missing:
        print("nothing missing — computing metrics from disk", flush=True)
    else:
        # Budget gate on the missing work (measured $/q when available).
        if args.limit is None:
            remaining = {}
            for r, a in missing:
                remaining[a] = remaining.get(a, 0) + 1
            if done_rows:
                proj = project_cost(done_rows, remaining)
                print(f"projection for remaining {sum(remaining.values())} "
                      f"rows: {proj['total_tokens']:.0f} tokens = "
                      f"${proj['total_usd']:.4f} (budget ${args.budget:.2f})",
                      flush=True)
                if proj["total_usd"] > args.budget and not args.yes:
                    print("REFUSED: projection exceeds --budget; pass --yes "
                          "to proceed.", flush=True)
                    return 3
            elif not args.yes:
                print("no measured $/q yet (no finished rows): run a "
                      "--limit checkpoint first, or pass --yes.", flush=True)
                return 3

    generator_tag = f"openrouter:{GENERATOR_MODEL}"
    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = (rag_gate_root() / "data" / "prompt_template.txt"
                ).read_text(encoding="utf-8")
    answer_fn = fail_closed_answer_fn(
        make_answer_fn(or_client, GENERATOR_MODEL, template), generator_tag)
    jev_key = load_jev_key()
    backends = {j: JevDecision(api_key=jev_key)
                for j in ("chunk", "selection", "sufficiency", "abstain",
                          "cache")}
    print(f"generator: {generator_tag}", flush=True)

    base = {"per_round_k": grid["per_round_k"],
            "candidate_pool_cap": grid["candidate_pool_cap"],
            "oversample": args.oversample, "max_rounds": args.rounds}
    t0 = time.perf_counter()
    wrote_header = old_header is not None
    out_f = open(out_path, "a", encoding="utf-8")
    try:
        if not wrote_header:
            out_f.write(json.dumps({"__header__": {
                "packet": "V03-A6", "dataset": dataset,
                "pool_mode": mode, "generator": generator_tag,
                "alpha": args.alpha, "per_round_k": grid["per_round_k"],
                "candidate_pool_cap": grid["candidate_pool_cap"],
                "grid_source": grid["grid_source"],
                "oversample": args.oversample, "rounds": args.rounds,
                "records_file": str(args.records),
                "records_sha256": sha256_file(args.records),
                "pipeline_sha256": sha256_file(
                    REPO_ROOT / "jevrag" / "pipeline.py"),
            }}, ensure_ascii=False) + "\n")
            out_f.flush()
            wrote_header = True

        # Contexts: shared -> one ctx per arm; per_question -> one ctx per
        # (arm, question), configs shared per arm (memo keys are per
        # question+chunk, safe across questions; never across arms).
        cfgs = make_arm_configs(backends, answer_fn, base)
        ctx_cache: dict = {}

        def get_ctx(arm: str, row: dict):
            key = (arm, row["id"]) if mode == "per_question" else (arm, "*")
            if key not in ctx_cache:
                chunks = pool_chunks_for_row(
                    row, pool, mode) if mode == "per_question" else \
                    [{"title": str(p["title"]), "text": str(p["text"])}
                     for p in pool]
                cfg = cfgs[arm if arm != "gated_crc" else "gated_crc"]
                if arm == "gated_crc":
                    cfg.crc_specs = crc_specs_map
                ctx_cache[key] = build_pipeline_from_chunks(
                    cfg, f"{dataset}-{row['id']}", chunks)
            return ctx_cache[key]

        crc_specs_map: dict = {}
        crc_rules_map: dict = {}
        current_sha: dict[str, str | None] = {"sha": None}
        # Calibrate-half gated rows (finished + fresh) feed the CRC specs;
        # run calibrate missing pairs first so specs exist before report.
        # Staleness (packet V03-A7 §1): a gated_crc row counts as finished
        # only if its crc_spec_sha matches the CURRENT calibration. Stale
        # rows are reported and re-run only with --redo-stale-crc (appended;
        # metrics take the last row per question).
        cal_missing = [(r, a) for (r, a) in missing
                       if r["split"] == "calibrate"]
        rep_planned = [(r, a) for (r, a) in missing
                       if r["split"] == "report"]
        fresh_rows: list[dict] = []

        def run_some(pairs) -> None:
            for r, a in pairs:
                ctx = get_ctx(a, r)
                q = {"id": str(r["id"]), "question": str(r["question"]),
                     "answer": str(r.get("answer", "")),
                     "type": str(r.get("type", "unknown"))}
                t1 = time.perf_counter()
                try:
                    if a == "fixed":
                        out = answer_question_fixed(
                            q, ctx, n_rounds=args.rounds,
                            generator_tag=generator_tag)
                    else:
                        out = answer_question(q, ctx)
                        out["generator"] = generator_tag
                except GeneratorHalted:
                    raise
                except Exception as exc:  # noqa: BLE001 — error rows
                    out = {"question_id": q["id"], "question": q["question"],
                           "gold": q["answer"], "outcome": "error",
                           "answer": "", "prediction": "", "source": "error",
                           "generator": generator_tag,
                           "error": f"{type(exc).__name__}: {exc}",
                           "error_trace": traceback.format_exc(limit=5),
                           "sufficiency": None, "abstain": None,
                           "selection_calls": 0, "jev_calls": 0,
                           "jev_input_tokens": 0,
                           "wall_ms": (time.perf_counter() - t1) * 1000.0}
                out["wall_ms_ask"] = out.get(
                    "wall_ms_ask", (time.perf_counter() - t1) * 1000.0)
                out["arm"] = a
                out["split"] = r["split"]
                out["dataset"] = dataset
                out["hops"] = r.get("hops")
                if a == "gated_crc":
                    out["crc_spec_sha"] = current_sha["sha"]
                sc = score_alias_aware(
                    out.get("prediction", ""), str(r.get("answer", "")),
                    list(r.get("aliases") or []))
                out["em"], out["f1"], out["matched_alias"] = (
                    sc["em"], sc["f1"], sc["matched"])
                try:
                    out["stage"] = stage_split(out)
                except Exception:  # noqa: BLE001 — error rows
                    out["stage"] = {}
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                out_f.flush()
                fresh_rows.append(out)

        run_some(cal_missing)
        # Specs AND rules from ALL calibrate gated rows (disk + fresh), so
        # the sha below identifies this exact calibration.
        cal_gated = [r for r in done_rows + fresh_rows
                     if r.get("split") == "calibrate"
                     and r.get("arm") == "gated"
                     and r.get("outcome") != "error"
                     and r.get("sufficiency") and r.get("abstain")]
        if any(a == "gated_crc" for r, a in plan
                if r["split"] == "report"):
            crc_specs_map.update(build_crc_specs(cal_gated, args.alpha))
            for joint in (JOINT_SUFFICIENCY, JOINT_ABSTAIN):
                crc_rules_map[joint] = calibrate_rule(
                    joint, crc_specs_map[joint])
            current_sha["sha"] = crc_spec_sha(
                crc_specs_map, crc_rules_map, args.alpha)
            print(f"CRC specs from {len(cal_gated)} calibrate gated rows "
                  f"(alpha={args.alpha}, sha={current_sha['sha'][:12]}…)",
                  flush=True)
            for joint, rule in crc_rules_map.items():
                print(f"  rule {joint}: abstain_all={rule.abstain_all} "
                      f"threshold={rule.threshold:.6f}", flush=True)
        # Stale scan over FINISHED gated_crc rows (not the missing set — a
        # stale row looks done until its sha is compared). Re-run only with
        # the flag (appended; metrics take the last row per question).
        rep_missing = [(r, a) for (r, a) in rep_planned
                       if (str(r["id"]), a) not in done]
        disk_crc = {(str(r.get("question_id"))): r for r in done_rows
                    if str(r.get("arm")) == "gated_crc"}
        for r, a in plan_pairs(rows, args.limit, with_crc=True):
            if a != "gated_crc" or r["split"] != "report":
                continue
            old = disk_crc.get(str(r["id"]))
            if old is not None and current_sha["sha"] is not None \
                    and str(old.get("crc_spec_sha")) != current_sha["sha"] \
                    and (r, a) not in rep_missing:
                print(f"  STALE gated_crc {r['id']}: "
                      f"sha {str(old.get('crc_spec_sha'))[:12]}… != "
                      f"{current_sha['sha'][:12]}…", flush=True)
                if args.redo_stale_crc:
                    rep_missing.append((r, a))
        run_some(rep_missing)
    except GeneratorHalted as exc:
        print(f"\nHALTED: {exc}\nrows so far kept; no fallback ran.",
              flush=True)
        return 1
    finally:
        out_f.close()

    # Metrics + checkpoint economics from the full on-disk set.
    # Readout thresholds are resolved from disk rows (deterministic across
    # invocations), never from this run's in-memory state. Redone rows
    # (--redo-stale-crc appends) metrick last-wins per (question, arm).
    _, all_rows = load_result_rows(out_path)
    latest: dict = {}
    for r in all_rows:
        latest[(str(r.get("question_id")), str(r.get("arm")))] = r
    all_rows = list(latest.values())
    abst_thresholds = resolve_abst_thresholds(all_rows, args.alpha)
    by_arm_split: dict = {}
    for r in all_rows:
        by_arm_split.setdefault((r.get("arm"), r.get("split")), []).append(r)
    metrics = {f"{a}/{s}": split_metrics(v, abst_thresholds.get(a))
               for (a, s), v in sorted(by_arm_split.items())}
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({"packet": "V03-A6", "dataset": dataset,
                   "outcomes_path": str(out_path),
                   "abst_thresholds": {k: v for k, v in
                                       abst_thresholds.items()},
                   "metrics": metrics}, f, indent=2)

    for key, m in metrics.items():
        print(f"{key}: n={m['n']} served={m['n_served']} "
              f"EM {m['em_overall']}/{m['n']} F1 {m['f1_mean']:.3f} "
              f"sel_risk={m['selective_risk']} marg_risk={m['marginal_risk']} "
              f"AURC={m['aurc']} ECE={m['ece']} rounds={m['rounds']} "
              f"Jev {m['jev_calls']} calls "
              f"(${m['cost_usd']:.4f})", flush=True)
    if args.limit is not None:
        # Checkpoint projection for the full n (both splits, all arms,
        # gated_crc re-run of the report remainder included by construction).
        full_plan = plan_pairs(rows, None, with_crc=True)
        have = {(str(r.get("question_id")), str(r.get("arm")))
                for r in all_rows}
        left = {}
        for r, a in full_plan:
            if (str(r["id"]), a) not in have:
                left[a] = left.get(a, 0) + 1
        proj = project_cost(all_rows, left)
        done_toks = sum(int(r.get("jev_input_tokens", 0)) for r in all_rows)
        print(f"checkpoint: {len(all_rows)} rows on disk "
              f"(measured {done_toks} Jev tokens = "
              f"${suite_cost_usd(done_toks):.4f} at "
              f"${SUITE_USD_PER_MTOK}/MTok); "
              f"projection for remaining {sum(left.values())} rows: "
              f"{proj['total_tokens']:.0f} tokens = "
              f"${proj['total_usd']:.4f} "
              f"(budget ${args.budget:.2f})", flush=True)
    # Degeneracy, per dataset (and per arm): pool growth AND evidence change.
    deg = degeneracy_report(all_rows)
    bad = {k: v for k, v in deg.items()
           if v["frac_no_new"] > 0.5 or v["frac_same_evidence"] > 0.5}
    for k, v in deg.items():
        print(f"degeneracy {k}: rounds 2+ with zero new candidates = "
              f"{v['frac_no_new']:.3f} (n={v['n_later_rounds']}), with "
              f"unchanged evidence = {v['frac_same_evidence']:.3f}",
              flush=True)
    print(f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}",
          flush=True)
    if bad and not args.allow_degenerate:
        print(f"DEGENERATE: {sorted(bad)} exceed 50% on growth or evidence "
              f"stagnation.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
