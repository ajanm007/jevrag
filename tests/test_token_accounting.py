"""Tests for generator token accounting.

Context: ``make_answer_fn`` used to return ``(answer, int)``, but
``sufficiency._call_answer_fn`` only reads **dict-form** usage. So neither arm
recorded generator tokens, and the baseline's ``input_tokens: 0`` meant
"unmeasured", not "free".

These tests use a fake OpenAI-shaped client, so they run offline and pin the
contract: the closure returns a usage dict, and those tokens reach
``record["input_tokens"]`` on BOTH arms.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.produce_records import build_generator_context, make_answer_fn  # noqa: E402

from jevrag.baselines.fixed_iteration import run_fixed_iteration  # noqa: E402
from jevrag.decision import DecisionResult  # noqa: E402
from jevrag.primitives.sufficiency import run_sufficiency  # noqa: E402

DOCS = [{"title": f"D{i}", "text": f"Text for document {i}. Second sentence."}
        for i in range(12)]
TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


# --------------------------------------------------------------------- fakes
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    def __init__(self, content, prompt_tokens, completion_tokens):
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _Completions:
    def __init__(self, prompt_tokens, completion_tokens, content):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self.content, self.prompt_tokens, self.completion_tokens)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    """Minimal stand-in for openai.OpenAI."""

    def __init__(self, prompt_tokens=1234, completion_tokens=7, content="an answer"):
        self.completions = _Completions(prompt_tokens, completion_tokens, content)
        self.chat = _Chat(self.completions)


class ScriptedDecision:
    """Decision backend with a known Jev token cost per call."""

    backend_name = "scripted"

    def __init__(self, confidences, tokens_per_call=200):
        self.confidences = list(confidences)
        self.tokens_per_call = tokens_per_call
        self.calls = 0

    def ask(self, state, questions):
        conf = self.confidences[min(self.calls, len(self.confidences) - 1)]
        self.calls += 1
        return DecisionResult(result=conf, confidence=conf, metadata={
            "backend": self.backend_name, "latency_ms": 100.0,
            "input_tokens": self.tokens_per_call, "cost_usd": 0.0,
        })


def retrieve_fn(question, n_docs):
    return DOCS[:n_docs]


QUESTION = {"id": "q1", "question": "Q?", "answer": "an answer", "type": "bridge"}


# ---------------------------------------------------------------------- tests
def test_answer_fn_returns_usage_dict():
    client = FakeClient(prompt_tokens=1234, completion_tokens=7)
    fn = make_answer_fn(client, "test-model", TEMPLATE)
    answer, usage = fn("Q?", DOCS[:5])

    assert answer == "an answer"
    assert isinstance(usage, dict), "usage must be dict-form; a bare int is the bug"
    assert usage["input_tokens"] == 1234
    assert usage["output_tokens"] == 7


def test_baseline_records_carry_generator_tokens():
    """The regression: baseline input_tokens used to be 0 for every record."""
    client = FakeClient(prompt_tokens=1500)
    fn = make_answer_fn(client, "test-model", TEMPLATE)

    rec = run_fixed_iteration(QUESTION, n_rounds=3, retrieve_fn=retrieve_fn,
                              answer_fn=fn)

    # Baseline makes no Jev calls, so all of its tokens are generator tokens.
    assert rec["input_tokens"] == 1500, "baseline generator tokens not recorded"
    assert rec["confidence"] is None


def test_gated_records_carry_jev_plus_generator_tokens():
    client = FakeClient(prompt_tokens=1500)
    fn = make_answer_fn(client, "test-model", TEMPLATE)
    decision = ScriptedDecision([0.3, 0.95])  # stops on round 2

    rec = run_sufficiency(
        question_id="q1", question="Q?", gold="an answer", qtype="bridge",
        decision=decision, retrieve_fn=retrieve_fn, answer_fn=fn,
        threshold=0.7, max_rounds=3,
    )

    assert rec["rounds_used"] == 2
    assert rec["input_tokens"] == 2 * 200 + 1500  # Jev rounds + one generation


def test_bare_int_usage_is_ignored_documenting_the_old_bug():
    """Pins WHY the closure must return a dict: a tuple's bare int is dropped."""
    def int_usage_fn(question, evidence):
        return "an answer", 999

    rec = run_fixed_iteration(QUESTION, n_rounds=1, retrieve_fn=retrieve_fn,
                              answer_fn=int_usage_fn)
    assert rec["input_tokens"] == 0


def test_plain_string_answer_still_supported():
    """A generator returning only a string stays valid (tokens counted as 0)."""
    rec = run_fixed_iteration(QUESTION, n_rounds=1, retrieve_fn=retrieve_fn,
                              answer_fn=lambda q, e: "an answer")
    assert rec["prediction"] == "an answer"
    assert rec["input_tokens"] == 0


def test_generator_context_is_capped_whole_docs_only():
    long_docs = [{"title": f"T{i}", "text": "word " * 200} for i in range(12)]
    ctx = build_generator_context(long_docs, per_doc_chars=300, total_budget_chars=2500)

    assert "…" in ctx                      # per-doc trim leaves its marker
    assert len(ctx) <= 2500 + 200          # budget plus last doc's title furniture
    assert ctx.count("[T") >= 1
    # Docs are joined with a blank-line separator; every content line is one
    # whole document (never a mid-concatenation fragment).
    for line in ctx.splitlines():
        if line.strip():
            assert line.startswith("[T")


def test_generator_context_keeps_short_evidence():
    ctx = build_generator_context(DOCS[:3])
    assert ctx.count("[D") == 3
