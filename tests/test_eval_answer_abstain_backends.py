"""Tests for scripts/eval_answer_abstain_backends (packet V03-B1). All offline.

Covers the new code only: the logprob-capturing generator closure (including the
no-logprobs sentinel path), the two-label separation in `compare`, and the
structural acceptance tests that this script adds no new backend, re-implements
no metric, and touches no protected module.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
          / "eval_answer_abstain_backends.py")


def _load():
    """Import the script's functions without running main()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_v03b1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def _entry(value):
    return SimpleNamespace(logprob=value)


class _FakeCompletions:
    def __init__(self, logprob_entries, content="Brooklyn"):
        self._entries = logprob_entries
        self._content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._content),
                logprobs=SimpleNamespace(content=self._entries),
            )],
            usage=SimpleNamespace(prompt_tokens=123, completion_tokens=4),
        )


class _FakeClient:
    def __init__(self, entries):
        self.completions = _FakeCompletions(entries)
        self.chat = SimpleNamespace(completions=self.completions)
# --- the logprob-capturing generator closure ---

def test_captures_mean_logprob_and_token_count(mod):
    client = _FakeClient([_entry(-0.2), _entry(-0.4)])
    fn = mod.make_logprob_answer_fn(client, "m", "{context}|{question}")
    answer, usage = fn("Q?", [{"title": "t", "text": "x"}])
    assert answer == "Brooklyn"
    assert usage["input_tokens"] == 123 and usage["output_tokens"] == 4
    assert fn.LAST_LOGPROB["mean_logprob"] == pytest.approx(-0.3)
    assert fn.LAST_LOGPROB["n_logprob_tokens"] == 2


def test_requests_logprobs_and_same_generation_params(mod):
    """Logprobs are requested, and temperature/max_tokens match
    produce_records' generator so the two runs stay comparable."""
    client = _FakeClient([_entry(0.0)])
    fn = mod.make_logprob_answer_fn(client, "m", "{context}|{question}")
    fn("Q?", [{"title": "t", "text": "x"}])
    kwargs = client.completions.kwargs
    assert kwargs["logprobs"] is True
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 32


def test_absent_logprobs_yield_nan_not_a_fabricated_number(mod):
    """No logprob payload must not silently become a confident answer. NaN
    propagates and is dropped pairwise by compare(), per calibration.py's
    documented NaN convention."""
    client = _FakeClient([])
    fn = mod.make_logprob_answer_fn(client, "m", "{context}|{question}")
    fn("Q?", [{"title": "t", "text": "x"}])
    assert math.isnan(fn.LAST_LOGPROB["mean_logprob"])
    assert fn.LAST_LOGPROB["n_logprob_tokens"] == 0


def test_none_logprob_payload_is_handled(mod):
    client = _FakeClient(None)
    fn = mod.make_logprob_answer_fn(client, "m", "{context}|{question}")
    fn("Q?", [{"title": "t", "text": "x"}])
    assert math.isnan(fn.LAST_LOGPROB["mean_logprob"])


def _row(jev, lp, correct, in_evidence):
    return {"question_id": "q", "correct": int(correct),
            "in_evidence": in_evidence,
            "jev_confidence": jev, "logprob_confidence": lp,
            "jev_action": "pass" if jev >= 0.5 else "abstain",
            "logprob_action": "pass" if lp >= 0.5 else "abstain"}
# --- compare(): the two label proxies stay separate ---

def test_compare_keeps_the_two_proxies_separate(mod):
    rows = [
        _row(0.9, 0.9, 1, True),    # correct
        _row(0.8, 0.8, 0, True),    # wrong, present in evidence
        _row(0.1, 0.1, 0, False),   # wrong, absent -> failure-mode positive
        _row(0.2, 0.2, 1, True),
    ]
    out = mod.compare(rows, n_boot=50, seed=0)
    assert out["base_em"] == pytest.approx(0.5)
    assert out["failure_mode_split"]["n_positive"] == 1
    assert out["failure_mode_split"]["aurc"] is not None


def test_failure_mode_proxy_is_degenerate_when_one_class(mod):
    rows = [_row(0.9, 0.9, 1, True), _row(0.1, 0.1, 0, True)]
    out = mod.compare(rows, n_boot=50, seed=0)
    assert out["failure_mode_split"]["aurc"] is None


def test_perfect_and_inverted_signals_order_as_expected(mod):
    """Sanity on the metric wiring: a signal ranking correct answers above
    wrong ones must score a strictly lower AURC than its inverse."""
    rows = [_row(0.9, 0.1, 1, True), _row(0.1, 0.9, 0, True)]
    out = mod.compare(rows, n_boot=50, seed=0)
    assert (out["em_correctness"]["jev"]["aurc"]
            < out["em_correctness"]["logprob"]["aurc"])


def test_selective_risk_at_matched_coverage_reported_for_both_backends(mod):
    rows = [_row(0.9, 0.9, 1, True), _row(0.8, 0.2, 0, True),
            _row(0.7, 0.7, 1, True), _row(0.3, 0.3, 0, True)]
    out = mod.compare(rows, n_boot=50, seed=0)
    block = out["selective_risk_at_matched_coverage"]
    assert set(block) == {"0.50", "0.60", "0.70", "0.80", "0.90"}
    for entry in block.values():
        assert set(entry) == {"jev", "logprob"}


def test_operating_point_uses_backend_defaults(mod):
    rows = [_row(0.9, 0.9, 1, True), _row(0.1, 0.9, 0, True)]
    out = mod.compare(rows, n_boot=50, seed=0)
    ops = out["operating_point"]
    assert ops["jev"]["n_pass"] == 1
    assert ops["jev"]["passed_em"] == pytest.approx(1.0)
    assert ops["logprob"]["n_abstain"] == 0


def test_nan_backend_score_is_dropped_not_zero_filled(mod):
    rows = [_row(0.9, 0.9, 1, True), _row(float("nan"), 0.3, 0, True),
            _row(0.8, 0.8, 1, True)]
    out = mod.compare(rows, n_boot=50, seed=0)
    assert out["em_correctness"]["jev"]["n_dropped_nan"] == 1
    assert out["em_correctness"]["jev"]["aurc"] < out["random_aurc"]


# --- structural acceptance tests ---

def test_script_adds_no_new_backend():
    """The packet's point: the harness hosts a second existing backend. This
    script must not define one."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert classes == [], f"unexpected class definitions: {classes}"


def test_script_does_not_reimplement_the_metrics():
    """AURC must come from the project's vendored function."""
    funcs = [n.name for n in ast.parse(
        SCRIPT.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)]
    assert "aurc" not in funcs and "risk_coverage_curve" not in funcs


def test_bootstrap_is_delegated_not_reimplemented():
    """The wrapper must delegate to the established implementation rather than
    carry its own resampling loop -- otherwise p-values silently drift from
    every other comparison on record."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "eval_ragas_faithfulness import paired_bootstrap" in text
    body = text.split("def paired_bootstrap", 1)[1]
    assert "rng.integers" not in body.split("def ", 1)[0]


def test_script_does_not_write_the_abstain_threshold():
    """Thresholds stay at the primitive's default -- the no-tuning guard,
    checked against the source rather than trusted."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "DEFAULT_ABSTAIN_THRESHOLD" in text
    assert "threshold=0." not in text.replace("0.5 = ", "")