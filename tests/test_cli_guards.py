"""Tests for the CLI's report guards: provenance header, generator parity,
unreportable-generator omission, tune presentation, and the generator prompt cap.
"""

import json

import pytest

from jevrag.__main__ import (
    NON_REPORTABLE_GENERATORS,
    REQUIRED_KEYS,
    _check_generator_parity,
    _tune_operating_point,
    main,
    print_report,
)
from jevrag.benchmarks import hotpotqa

GEN = "openrouter:qwen/qwen3-8b"


def make_record(confidence=0.9, prediction="yes", generator=GEN,
                policy="sufficiency_jev", qid="q1", rounds=2, tokens=500,
                gold="yes"):
    return {
        "question_id": qid,
        "question": "Q?",
        "gold": gold,
        "prediction": prediction,
        "confidence": confidence,
        "rounds_used": rounds,
        "latency_ms": 100.0,
        "input_tokens": tokens,
        "type": "bridge",
        "policy": policy,
        "generator": generator,
        "split": "val",
    }


def test_required_keys_includes_generator():
    assert "generator" in REQUIRED_KEYS


def test_parity_ok_same_generator():
    gated = [make_record()]
    base = [make_record(confidence=None, policy="fixed_iteration_3")]
    assert _check_generator_parity(gated, base) == (GEN, GEN)


def test_parity_no_baseline():
    assert _check_generator_parity([make_record()], None) == (GEN, None)


def test_parity_mismatch_refuses():
    gated = [make_record(generator="openrouter:qwen/qwen3-8b")]
    base = [make_record(generator="openrouter:other-model")]
    with pytest.raises(ValueError, match="parity violated"):
        _check_generator_parity(gated, base)


def test_parity_mixed_arm_refuses():
    gated = [make_record(generator="a"), make_record(generator="b")]
    with pytest.raises(ValueError, match="mixes generators"):
        _check_generator_parity(gated, None)


def test_fallback_report_omits_numbers(capsys):
    recs = [make_record(generator="extractive_fallback") for _ in range(4)]
    report = print_report(recs, None, n_bins=2, out_path=None)
    out = capsys.readouterr().out
    assert "UNREPORTABLE" in out
    assert "extractive_fallback" in out  # provenance in the header
    assert report["verdict"].startswith("unreportable")
    assert report["em_f1"] is None
    assert report["calibration"] is None
    assert report["risk_coverage"] is None
    assert report["baseline"] is None
    # Cost/latency are real under any generator and stay.
    assert report["cost"]["n_records"] == 4
    assert "Cost / latency" in out
    assert "--- Accuracy" not in out
    assert "--- Calibration" not in out
    assert "--- Risk / coverage" not in out


def test_unknown_generator_also_unreportable(capsys):
    recs = [make_record(generator="unknown")]
    report = print_report(recs, None, n_bins=2, out_path=None)
    assert report["verdict"].startswith("unreportable")
    assert "UNREPORTABLE" in capsys.readouterr().out


def test_reportable_generator_shows_numbers(capsys):
    recs = [make_record(confidence=c, prediction=p)
            for c, p in [(0.9, "yes"), (0.2, "no"), (0.8, "yes"), (0.4, "no")]]
    report = print_report(recs, None, n_bins=2, out_path=None)
    out = capsys.readouterr().out
    assert report["verdict"] == "reportable"
    assert f"generator: {GEN}" in out
    assert report["em_f1"] is not None
    assert report["calibration"] is not None


def test_tune_reports_val_and_applied_side(capsys):
    import numpy as np

    recs = [make_record(confidence=c, prediction=p)
            for c, p in [(0.9, "yes"), (0.2, "no"), (0.8, "yes"), (0.4, "no"),
                         (0.95, "yes"), (0.1, "no")]]
    conf = np.array([r["confidence"] for r in recs])
    corr = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    op = _tune_operating_point(recs, conf, corr, target_coverage=0.9)
    out = capsys.readouterr().out
    assert op["tuned_on"] == "val"
    assert op["target_coverage"] == 0.9
    assert "val" in op and "applied" in op  # both sides, the transfer gap
    assert "production gate" in out  # untuned numbers labeled, not replaced


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_cli_refuses_mismatched_files(tmp_path, capsys, monkeypatch):
    import jevrag.__main__ as cli

    monkeypatch.setattr(cli, "split_lookup", lambda: {"q1": "val"})
    gated = tmp_path / "gated.jsonl"
    base = tmp_path / "base.jsonl"
    _write_jsonl(gated, [make_record()])
    _write_jsonl(base, [make_record(generator="other:model")])
    rc = main(["eval", "sufficiency", "--records", str(gated),
               "--baseline-records", str(base), "--split", "all"])
    assert rc == 2
    assert "parity violated" in capsys.readouterr().err


@pytest.mark.skipif(
    not hotpotqa.dataset_path().exists(),
    reason=(
        "cmd_eval_sufficiency resolves the real HotpotQA split lookup before "
        "reaching the generator guard this test checks; needs the real data "
        "file (see test_hotpotqa.py's skip reason) even though the assertion "
        "itself is only about the generator field"
    ),
)
def test_cli_rejects_record_without_generator(tmp_path, capsys):
    gated = tmp_path / "gated.jsonl"
    rec = make_record()
    del rec["generator"]
    _write_jsonl(gated, [rec])
    rc = main(["eval", "sufficiency", "--records", str(gated),
               "--baseline-records", "none", "--split", "all"])
    assert rc == 2
    assert "generator" in capsys.readouterr().err


# --- Findings 1-4 (REVIEW_02) ------------------------------------------------

def _paired_fixture():
    gated = [
        make_record(qid="q1", confidence=0.9, prediction="yes", rounds=1),
        make_record(qid="q2", confidence=0.4, prediction="no", rounds=2),
        make_record(qid="q3", confidence=0.8, prediction="yes", rounds=3),
        make_record(qid="q4", confidence=0.95, prediction="x", gold="y",
                    rounds=1),
    ]
    base = [
        make_record(qid="q1", confidence=None, prediction="yes", rounds=3,
                    policy="fixed_iteration_3", tokens=0),
        make_record(qid="q2", confidence=None, prediction="yes", rounds=3,
                    policy="fixed_iteration_3", tokens=0),
        make_record(qid="q3", confidence=None, prediction="yes", rounds=3,
                    policy="fixed_iteration_3", tokens=0),
        make_record(qid="q4", confidence=None, prediction="z", gold="y",
                    rounds=3, policy="fixed_iteration_3", tokens=0),
    ]
    return gated, base


def test_paired_rounds_summary_numbers(capsys):
    gated, base = _paired_fixture()
    report = print_report(gated, base, n_bins=2, out_path=None)
    paired = report["baseline"]["paired"]
    assert paired["n_paired"] == 4
    assert paired["n_unpaired_gated"] == 0
    assert (paired["gated_total"], paired["baseline_total"]) == (7, 12)
    assert paired["reduction_frac"] == pytest.approx((12 - 7) / 12)
    assert paired["identical_predictions"] == {"n_same": 2, "n": 4}
    early = paired["early_stopped"]
    assert early["n"] == 3
    assert early["gated_em"] == pytest.approx(1 / 3)
    assert early["baseline_em"] == pytest.approx(2 / 3)
    out = capsys.readouterr().out
    assert "rounds (retrieval)" in out
    assert "-41.7%" in out
    assert "early-stopped" in out
    assert "scope:" in out and "THIS split" in out


def test_latency_note_and_zero_token_note(capsys):
    gated, base = _paired_fixture()
    print_report(gated, base, n_bins=2, out_path=None)
    out = capsys.readouterr().out
    assert "not comparable across arms" in out  # Finding 2, no delta printed
    assert "UNPOPULATED" in out  # Finding 3: 0 is missing data, not free


def test_unreportable_cli_exits_nonzero(tmp_path, capsys, monkeypatch):
    import jevrag.__main__ as cli

    monkeypatch.setattr(cli, "split_lookup", lambda: {"q1": "val"})
    gated = tmp_path / "gated.jsonl"
    _write_jsonl(gated, [make_record(generator="extractive_fallback")])
    rc = main(["eval", "sufficiency", "--records", str(gated),
               "--baseline-records", str(tmp_path / "missing.jsonl"),
               "--split", "all"])
    assert rc == 1
    assert "UNREPORTABLE" in capsys.readouterr().out


def test_reportable_cli_exits_zero(tmp_path, capsys, monkeypatch):
    import jevrag.__main__ as cli

    monkeypatch.setattr(cli, "split_lookup", lambda: {"q1": "val"})
    gated = tmp_path / "gated.jsonl"
    _write_jsonl(gated, [make_record()])
    rc = main(["eval", "sufficiency", "--records", str(gated),
               "--baseline-records", str(tmp_path / "missing.jsonl"),
               "--split", "all"])
    assert rc == 0


# --- PACKET 04: mixed-provenance subsets + pooled tune -------------------------

KAGGLE_GEN = "kaggle-hf:Qwen/Qwen3-8B"


def _mixed_fixture():
    gated = [
        make_record(qid="m1", confidence=0.9, prediction="yes", rounds=1,
                    generator=GEN),
        make_record(qid="m2", confidence=0.8, prediction="yes", rounds=2,
                    generator=GEN),
        make_record(qid="m3", confidence=0.7, prediction="no", gold="no",
                    rounds=1, generator=KAGGLE_GEN),
        make_record(qid="m4", confidence=0.6, prediction="no", gold="yes",
                    rounds=3, generator=KAGGLE_GEN),
    ]
    base = [
        make_record(qid="m1", confidence=None, prediction="yes", rounds=3,
                    policy="fixed_iteration_3", tokens=500, generator=GEN),
        make_record(qid="m2", confidence=None, prediction="yes", rounds=3,
                    policy="fixed_iteration_3", tokens=500, generator=GEN),
        make_record(qid="m3", confidence=None, prediction="no", gold="no",
                    rounds=3, policy="fixed_iteration_3", tokens=700,
                    generator=KAGGLE_GEN),
        make_record(qid="m4", confidence=None, prediction="no", gold="yes",
                    rounds=3, policy="fixed_iteration_3", tokens=700,
                    generator=KAGGLE_GEN),
    ]
    return gated, base


def _lookup_for(*record_lists):
    return {r["question_id"]: "val" for lst in record_lists for r in lst}


def test_mixed_val_discloses_subsets_no_blend(tmp_path, capsys, monkeypatch):
    import jevrag.__main__ as cli

    gated, base = _mixed_fixture()
    monkeypatch.setattr(cli, "split_lookup", lambda: _lookup_for(gated, base))
    gp, bp = tmp_path / "g.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(gp, gated)
    _write_jsonl(bp, base)
    rc = main(["eval", "sufficiency", "--records", str(gp),
               "--baseline-records", str(bp), "--split", "all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mixed provenance" in out
    assert "never averaged" in out
    assert f"--- subset: {GEN!r} (n=2) ---" in out
    assert f"--- subset: {KAGGLE_GEN!r} (n=2) ---" in out
    # No blended overall accuracy section outside subset banners.
    assert out.count("--- Accuracy") == 2


def test_mixed_with_fallback_subset_omits_only_it(tmp_path, capsys, monkeypatch):
    import jevrag.__main__ as cli

    gated, base = _mixed_fixture()
    gated = [dict(r) for r in gated]
    for r in gated:
        if r["generator"] == KAGGLE_GEN:
            r["generator"] = "extractive_fallback"
    monkeypatch.setattr(cli, "split_lookup", lambda: _lookup_for(gated, base))
    gp = tmp_path / "g.jsonl"
    _write_jsonl(gp, gated)
    rc = main(["eval", "sufficiency", "--records", str(gp),
               "--baseline-records", str(tmp_path / "missing.jsonl"),
               "--split", "all"])
    assert rc == 1  # unreportable subset fails the eval mechanically
    out = capsys.readouterr().out
    assert "UNREPORTABLE subset" in out
    assert f"--- subset: {GEN!r} (n=2) ---" in out  # real subset still shown


def test_baseline_extra_generator_ignored_with_note(tmp_path, capsys, monkeypatch):
    import jevrag.__main__ as cli

    gated = [make_record(qid="e1"), make_record(qid="e2")]
    base = [make_record(qid="e1", confidence=None, policy="fixed_iteration_3"),
            make_record(qid="e9", confidence=None, policy="fixed_iteration_3",
                        generator="other:model")]
    monkeypatch.setattr(cli, "split_lookup",
                        lambda: _lookup_for(gated, base))
    gp, bp = tmp_path / "g.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(gp, gated)
    _write_jsonl(bp, base)
    rc = main(["eval", "sufficiency", "--records", str(gp),
               "--baseline-records", str(bp), "--split", "all"])
    assert rc == 0
    assert "no gated counterpart" in capsys.readouterr().out


def test_tune_pooled_with_per_subset_val_lines(capsys):
    import numpy as np

    gated, _ = _mixed_fixture()
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    corr = np.array([1.0, 1.0, 1.0, 0.0])
    op = _tune_operating_point(gated, conf, corr, 0.9, "test")
    out = capsys.readouterr().out
    assert op["pooled"] is True
    assert set(op["tune_generators"]) == {GEN, KAGGLE_GEN}
    assert set(op["val_subsets"]) == {GEN, KAGGLE_GEN}
    assert "pooled val" in out
    assert "per subset" in out
    assert "mean_conf" in out  # hosting-independence stays a checked claim


def test_tune_applied_to_val_says_pending_test_data(capsys):
    import numpy as np

    gated, _ = _mixed_fixture()
    conf = np.array([r["confidence"] for r in gated])
    corr = np.ones(4)
    _tune_operating_point(gated, conf, corr, 0.9, "val")
    out = capsys.readouterr().out
    assert "pending test data" in out


# --- Item 1: generator prompt cap -------------------------------------------

def _produce():
    import scripts.produce_records as pr
    return pr


def test_context_respects_total_budget():
    pr = _produce()
    docs = [{"title": f"D{i}", "text": "x" * 2000} for i in range(15)]
    ctx = pr.build_generator_context(docs)
    assert len(ctx) <= pr.GENERATOR_TOTAL_BUDGET_CHARS + 300  # titles/joins slack
    # Whole documents only: every non-final part ends at a doc boundary.
    parts = ctx.split("\n\n")
    for part in parts:
        assert part.startswith("[D")


def test_context_never_cuts_mid_word():
    pr = _produce()
    docs = [{"title": "A", "text": "SENTinel " * 500}]
    ctx = pr.build_generator_context(docs, per_doc_chars=100,
                                     total_budget_chars=10000)
    body = ctx.split("] ", 1)[1]
    assert body.endswith("…")
    assert " ".join(body[:-1].split()) == " ".join(["SENTinel"] * 11)


def test_context_empty_evidence():
    pr = _produce()
    assert pr.build_generator_context([]) == ""


def test_prompt_fits_squeezed_ceiling():
    """~2500 chars + template/question must stay well under the observed
    1096-token ceiling (chars/4 rule of thumb)."""
    pr = _produce()
    docs = [{"title": f"D{i}", "text": "y" * 2000} for i in range(15)]
    template = "Answer using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    prompt = template.replace(
        "{context}", pr.build_generator_context(docs)).replace("{question}", "Q?")
    assert len(prompt) / 4 < 1096
