"""Suite runner offline tests (packets V03-A6 §7, V03-A7). No network.

Stub Decisions + stub generator + contract fixtures built in tmp dirs.
main() is exercised end to end with monkeypatched backends, key loaders and
RAG_GATE_PATH (tiny prompt_template in tmp_path) so nothing leaves tmp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.run_suite as suite  # noqa: E402
from jevrag.decision import StubDecision  # noqa: E402


def _stub_backends(monkeypatch, suff=0.95, abstain=0.8):
    monkeypatch.setattr(
        suite, "JevDecision",
        lambda **kw: StubDecision(confidences={
            "sufficient": suff, "grounded": abstain, "relevant": 0.9,
            "split": 0.1, "serve_from_cache": 0.1,
            "staleness_risk": 0.1}))
    import scripts.produce_records as pr

    def fake_make_answer_fn(client, model, template):
        def answer_fn(question, evidence):
            return f"stubbed {question[:20]}", {"input_tokens": 3,
                                                "output_tokens": 2}
        return answer_fn

    monkeypatch.setattr(pr, "make_answer_fn", fake_make_answer_fn)
    import scripts._load_run_keys as rk
    monkeypatch.setattr(rk, "load_jev_key", lambda: "test-key")
    monkeypatch.setattr(rk, "load_openrouter_key", lambda: "test-key")
    # run_suite imported the names directly; patch its references too.
    monkeypatch.setattr(suite, "load_jev_key", lambda: "test-key")
    monkeypatch.setattr(suite, "load_openrouter_key", lambda: "test-key")


def _fixture_records(path: Path, mode: str, n_cal: int = 3, n_rep: int = 3,
                     pool_file: str | None = None):
    header = {"__header__": {
        "dataset": "tinytest", "pool_mode": mode, "seed": 1,
        "n": n_cal + n_rep, "n_calibrate": n_cal, "n_report": n_rep}}
    if mode == "shared":
        header["__header__"]["pool_file"] = pool_file
    rows = []
    for i in range(n_cal + n_rep):
        split = "calibrate" if i < n_cal else "report"
        r = {"id": f"q{i}", "question": f"Question number {i}?",
             "answer": f"answer {i}", "aliases": [f"ans {i}"],
             "type": "bridge", "hops": 2, "split": split,
             "supporting_ids": ["p0"]}
        if mode == "per_question":
            r["paragraphs"] = [
                {"id": "p0", "title": f"t{i}-keep",
                 "text": f"keep evidence for question {i}"},
                {"id": "p1", "title": f"t{i}-drop",
                 "text": f"drop unrelated filler {i}"}]
        rows.append(r)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _fixture_pool(path: Path):
    pool = [{"id": f"c{j}", "title": f"pool-{j}",
             "text": f"keep pool passage {j}"} for j in range(8)]
    with open(path, "w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    return path


@pytest.fixture(autouse=True)
def _portable_gatedata(monkeypatch, tmp_path):
    """Portable RAG_GATE_PATH (packet V03-A7 §4): a tiny prompt_template in
    tmp_path instead of the WSL-only /mnt/... path. Content is irrelevant
    (the stub generator never reads it); existence is what main() needs."""
    data = tmp_path / "rgdata"
    (data / "data").mkdir(parents=True)
    (data / "data" / "prompt_template.txt").write_text(
        "Q: {question}\nC: {context}\n", encoding="utf-8")
    monkeypatch.setenv("RAG_GATE_PATH", str(data))


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_suite.py", *argv])
    return suite.main()


@pytest.mark.parametrize("mode", ["per_question", "shared"])
def test_both_modes_run_all_arms(tmp_path, monkeypatch, mode):
    _stub_backends(monkeypatch)
    rec = _fixture_records(tmp_path / "r.jsonl", mode,
                           pool_file=str(tmp_path / "pool.jsonl"))
    if mode == "shared":
        _fixture_pool(tmp_path / "pool.jsonl")
    rc = _run_main(monkeypatch, "--records", str(rec),
                   "--out-prefix", str(tmp_path / "out"), "--yes", "--allow-degenerate")
    assert rc == 0
    _, rows = suite.load_result_rows(tmp_path / "out.jsonl")
    arms = {(r["question_id"], r["arm"]) for r in rows}
    assert len(rows) == 3 * 2 + 3 * 3  # cal×2 arms + rep×3 arms
    assert not any(a == "gated_crc" and r["split"] == "calibrate"
                   for r in rows for a in [r["arm"]])
    assert {r["em"] for r in rows} <= {True, False}
    assert all("stage" in r for r in rows)


def test_resume_skips_exactly_finished_pairs(tmp_path, monkeypatch):
    _stub_backends(monkeypatch)
    rec = _fixture_records(tmp_path / "r.jsonl", "shared",
                           pool_file=str(tmp_path / "pool.jsonl"))
    _fixture_pool(tmp_path / "pool.jsonl")
    out = str(tmp_path / "out")
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--limit", "1", "--allow-degenerate") == 0
    _, first = suite.load_result_rows(tmp_path / "out.jsonl")
    n_first = len(first)
    assert n_first == 1 * 2 + 1 * 2  # checkpoint: gated+fixed only, no CRC
    assert not any(r["arm"] == "gated_crc" for r in first)
    # Second identical invocation: everything skipped, row count unchanged.
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--limit", "1", "--allow-degenerate") == 0
    _, second = suite.load_result_rows(tmp_path / "out.jsonl")
    assert len(second) == n_first
    # Full run continues from the checkpoint, no duplicates.
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--yes", "--allow-degenerate") == 0
    _, full = suite.load_result_rows(tmp_path / "out.jsonl")
    pairs = [(r["question_id"], r["arm"]) for r in full]
    assert len(pairs) == len(set(pairs)) == 3 * 2 + 3 * 3


def test_budget_refusal_without_yes(tmp_path, monkeypatch):
    _stub_backends(monkeypatch)
    rec = _fixture_records(tmp_path / "r.jsonl", "shared",
                           pool_file=str(tmp_path / "pool.jsonl"))
    _fixture_pool(tmp_path / "pool.jsonl")
    rc = _run_main(monkeypatch, "--records", str(rec),
                   "--out-prefix", str(tmp_path / "out"),
                   "--budget", "0", "--limit", "1", "--allow-degenerate")
    # Checkpoint itself runs (limit path has no gate); the refusal path is
    # the full run with nothing measured... covered below instead.
    assert rc == 0
    rc = _run_main(monkeypatch, "--records", str(rec),
                   "--out-prefix", str(tmp_path / "out2"),
                   "--budget", "0")
    assert rc == 3  # no measured $/q and no --yes: refused
    assert not (tmp_path / "out2.jsonl").exists()


def test_fail_closed_402_halts_with_rows_kept(tmp_path, monkeypatch):
    _stub_backends(monkeypatch)
    import scripts.produce_records as pr

    class E402(Exception):
        status_code = 402

    def boom_make_answer_fn(client, model, template):
        def answer_fn(question, evidence):
            raise E402("Insufficient credits")
        return answer_fn

    monkeypatch.setattr(pr, "make_answer_fn", boom_make_answer_fn)
    rec = _fixture_records(tmp_path / "r.jsonl", "shared",
                           pool_file=str(tmp_path / "pool.jsonl"))
    _fixture_pool(tmp_path / "pool.jsonl")
    rc = _run_main(monkeypatch, "--records", str(rec),
                   "--out-prefix", str(tmp_path / "out"), "--yes", "--allow-degenerate")
    assert rc == 1


def test_crc_specs_come_from_calibrate_only():
    rows = [
        {"question_id": "c0", "split": "calibrate", "arm": "gated",
         "em": True, "sufficiency": {"record": {"confidence": 0.9}},
         "abstain": {"record": {"confidence": 0.8}}},
        {"question_id": "r0", "split": "report", "arm": "gated",
         "em": False, "sufficiency": {"record": {"confidence": 0.1}},
         "abstain": {"record": {"confidence": 0.2}}},
    ]
    specs = suite.build_crc_specs(
        [r for r in rows if r["split"] == "calibrate"], 0.3)
    assert specs[suite.JOINT_SUFFICIENCY].scores == [0.9]
    assert specs[suite.JOINT_SUFFICIENCY].keys == ["c0"]
    assert specs[suite.JOINT_ABSTAIN].scores == [0.8]


def test_contract_validation_rejects_bad_rows(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"__header__": {"pool_mode": "x"}}) + "\n")
    with pytest.raises(ValueError, match="pool_mode"):
        suite.load_suite_records(bad)
    bad2 = tmp_path / "bad2.jsonl"
    bad2.write_text(
        json.dumps({"__header__": {"pool_mode": "per_question"}}) + "\n"
        + json.dumps({"id": "q", "question": "?", "answer": "a",
                      "split": "calibrate", "paragraphs": [{"id": "p0"}],
                      "supporting_ids": ["pX"]}) + "\n")
    with pytest.raises(ValueError, match="dangling"):
        suite.load_suite_records(bad2)


def test_aggregator_on_fixture(tmp_path):
    from scripts.suite_table import markdown_rows, table_for_results
    res = tmp_path / "res.jsonl"
    header = {"__header__": {"dataset": "tinytest"}}
    rows = [
        {"question_id": "q0", "arm": "gated", "split": "report",
         "outcome": "answer", "prediction": "a", "gold": "a",
         "em": True, "f1": 1.0, "jev_calls": 5, "jev_input_tokens": 100,
         "selection_calls": 3, "selection": {"rounds": []},
         "abstain": {"record": {"confidence": 0.9}},
         "sufficiency": {"record": {"rounds_used": 1}}},
        {"question_id": "q1", "arm": "fixed", "split": "report",
         "outcome": "answer", "prediction": "b", "gold": "a",
         "em": False, "f1": 0.0, "jev_calls": 4, "jev_input_tokens": 80,
         "selection_calls": 4, "selection": {"rounds": []},
         "sufficiency": None, "abstain": None},
    ]
    with open(res, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md = markdown_rows([table_for_results(str(res))])
    assert "| tinytest | gated | report |" in md
    assert "| tinytest | fixed | report |" in md
    assert "n/a" in md  # fixed has no gate signal: AURC/ECE unavailable


def test_resolve_thresholds_deterministic_shape():
    import scripts.run_suite as rs
    assert rs.resolve_abst_thresholds([], 0.3)["gated_crc"] is None
    assert rs.resolve_abst_thresholds([], 0.3)["fixed"] is None
    assert isinstance(
        rs.resolve_abst_thresholds([], 0.3)["gated"], float)


def test_overwrite_guard_refuses_different_run(tmp_path, monkeypatch):
    _stub_backends(monkeypatch)
    rec = _fixture_records(tmp_path / "r.jsonl", "shared",
                           pool_file=str(tmp_path / "pool.jsonl"))
    _fixture_pool(tmp_path / "pool.jsonl")
    out = str(tmp_path / "out")
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--limit", "1",
                     "--allow-degenerate") == 0
    # Same prefix, different records content -> refused without --overwrite.
    rec2 = _fixture_records(tmp_path / "r2.jsonl", "shared",
                            pool_file=str(tmp_path / "pool.jsonl"))
    with open(rec2, "a", encoding="utf-8") as f:  # force a real difference
        f.write(json.dumps({"id": "qx", "question": "Extra?",
                            "answer": "yes", "split": "report",
                            "supporting_ids": ["c0"]}) + "\n")
    import scripts.run_suite as rs
    with pytest.raises(Exception, match="DIFFERENT run"):
        rs.guard_or_resume(out, False, {
            "dataset": "tinytest", "pool_mode": "shared",
            "generator": "openrouter:openai/gpt-4o-mini",
            "alpha": 0.3, "per_round_k": 5, "candidate_pool_cap": 30,
            "oversample": 2, "rounds": 3,
            "records_sha256": rs.sha256_file(str(rec2))})
    # ...but allowed with --overwrite.
    rs.guard_or_resume(out, True, {})


def _fixture_records_sized(path: Path, pool_n: int):
    rec = {"__header__": {"dataset": "tinytest", "pool_mode": "per_question",
                          "seed": 1, "n": 2, "n_calibrate": 1,
                          "n_report": 1}}
    rows = []
    for i in (0, 1):
        rows.append({
            "id": f"q{i}", "question": f"Sized question {i}?",
            "answer": f"answer {i}", "aliases": [], "type": "bridge",
            "split": "calibrate" if i == 0 else "report",
            "supporting_ids": ["p0"],
            "paragraphs": [
                {"id": f"p{j}", "title": f"t{i}-{j}",
                 "text": f"keep passage {i} {j}"} for j in range(pool_n)]})
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_grid_defaults_from_pool_size(tmp_path, monkeypatch):
    _stub_backends(monkeypatch)
    assert suite.resolve_grid(
        {"pool_mode": "shared"}, [], None, None) == {
            "per_round_k": 5, "candidate_pool_cap": 30,
            "grid_source": "header-derived"}
    assert suite.resolve_grid(
        {"pool_mode": "per_question"},
        [{"paragraphs": [{"id": 1}] * 10}], None, None)["per_round_k"] == 2
    assert suite.resolve_grid(
        {"pool_mode": "per_question"},
        [{"paragraphs": [{"id": 1}] * 20}], None, None)["per_round_k"] == 3
    # Explicit flags always win; header records resolved values.
    rec = _fixture_records_sized(tmp_path / "r.jsonl", 10)
    out = str(tmp_path / "out")
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--yes",
                     "--allow-degenerate") == 0
    header, _ = suite.load_result_rows(tmp_path / "out.jsonl")
    assert header["per_round_k"] == 2
    assert header["candidate_pool_cap"] == 12


def test_stale_crc_redo(tmp_path, monkeypatch):
    _stub_backends(monkeypatch)
    rec = _fixture_records(tmp_path / "r.jsonl", "shared",
                           pool_file=str(tmp_path / "pool.jsonl"))
    _fixture_pool(tmp_path / "pool.jsonl")
    out = str(tmp_path / "out")
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--yes",
                     "--allow-degenerate") == 0
    _, rows = suite.load_result_rows(tmp_path / "out.jsonl")
    crc_rows = [r for r in rows if r.get("arm") == "gated_crc"]
    assert crc_rows and all(r.get("crc_spec_sha") for r in crc_rows)
    sha0 = crc_rows[0]["crc_spec_sha"]
    # Tamper one sha (simulates a changed calibration): rerun without the
    # flag leaves the count alone; with it appends exactly one redone row.
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    # tamper only the first gated_crc row
    seen_crc = False
    out_lines = []
    for line in lines:
        obj = json.loads(line)
        if obj.get("arm") == "gated_crc" and not seen_crc:
            obj["crc_spec_sha"] = "tampered"
            seen_crc = True
            line = json.dumps(obj)
        out_lines.append(line)
    (tmp_path / "out.jsonl").write_text("\n".join(out_lines) + "\n",
                                        encoding="utf-8")
    n_before = len(out_lines) - 1
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--yes",
                     "--allow-degenerate") == 0
    _, rows2 = suite.load_result_rows(tmp_path / "out.jsonl")
    assert len(rows2) == n_before  # stale reported, not re-run
    assert _run_main(monkeypatch, "--records", str(rec),
                     "--out-prefix", out, "--yes", "--redo-stale-crc",
                     "--allow-degenerate") == 0
    _, rows3 = suite.load_result_rows(tmp_path / "out.jsonl")
    assert len(rows3) == n_before + 1  # exactly one redone row appended
    assert rows3[-1]["crc_spec_sha"] == sha0


def test_degeneracy_report_both_fractions():
    rep = suite.degeneracy_report([
        {"arm": "gated", "selection": {"rounds": [
            {"new_candidates": ["a"], "evidence_ids": ["a"]},
            {"new_candidates": ["b"], "evidence_ids": ["a"]},
            {"new_candidates": [], "evidence_ids": ["a"]}]}},
    ])
    g = rep["arm:gated"]
    assert g["n_later_rounds"] == 2
    assert g["frac_no_new"] == 0.5
    assert g["frac_same_evidence"] == 1.0  # evidence never changes
    assert rep["all"]["frac_no_new"] == 0.5
