# JevRAG

A common decision substrate for RAG (retrieval-augmented generation)
pipelines: a shared abstraction, a swappable decision backend, and a
calibration-first evaluation harness.

```
state → Decision → confidence → action
```

## The idea

Every RAG pipeline has moments where it has to decide something mid-flight:
is this chunk of text a coherent unit or should it be split, is the evidence
gathered so far enough to answer the question, which retrieved passages
actually earn a slot in the context window, is a cached answer safe to
reuse, does the generated answer actually stay grounded in the evidence.
Most systems make these calls with a hardcoded threshold or a heuristic
buried somewhere in the code, untested and unmeasured.

JevRAG's idea is to make each of these a first-class, explicit decision:
hand a decision backend the relevant state, get back a typed answer with a
calibrated confidence, and let the calling code decide what to do with that
confidence rather than letting the model decide silently. The decision
backend itself is swappable — a different model, a trained classifier, a
simple rule — behind one interface. Jev, a decision-focused model from
TypeSafe AI, is the first backend actually wired up; it is not the point of
the project. The point is the substrate and, just as importantly, an
evaluation harness that reports honestly on whether a decision's confidence
means anything, rather than trusting a vendor's claim about it.

## Quickstart

What you need:

- Python 3.11+
- A **RAG-Gate checkout** — this project deliberately reuses RAG-Gate's
  tested risk-coverage/AURC math, EM/F1 scoring, and its prepared
  HotpotQA data (1000 questions, frozen val/test split, prebuilt BM25
  index) rather than reimplementing any of it. By default the code looks
  for it at `D:\Research\RAG-Gate`; point `RAG_GATE_PATH` at wherever
  your own checkout lives if that's not where it sits. Without this,
  nothing that touches HotpotQA will run.
- A **Jev API key** (TypeSafe AI) — needed to generate fresh records.
  Not needed if you're only evaluating records someone already gave you.
- An **OpenRouter key**, only if you want to generate your own fresh
  records rather than evaluate existing ones — it's the answer generator.

```bash
git clone <this-repo-url>
cd jevrag
pip install -e .
```

Create a `.env` file at the repo root with your keys:

```
JEV_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

**Fastest path to real numbers — generate a small batch, then evaluate it:**

```bash
python scripts/produce_records.py --split val --limit 20
jevrag eval sufficiency --dataset hotpotqa \
    --records records/sufficiency_hotpotqa_val.jsonl \
    --baseline-records records/fixed_iter3_hotpotqa_val.jsonl \
    --split val
```

The first command makes real Jev + generator calls for 20 HotpotQA
questions (cheap — a few cents) and writes two JSONL files under
`records/`. The second reads them back and prints the full report:
accuracy, calibration (ECE/Brier), a risk-coverage table, and a cost
breakdown — the same report shape the full n=700 results below came
from, just on a smaller, faster sample. Drop `--limit 20` (and switch
`--split val` to `--split test`) to reproduce the real headline numbers,
though that means the full 700-question run.

No Jev key yet? The `jevrag eval` commands never call a model
themselves — they only read a records file and report on it — so if you
already have a records file from someone else's run (`records/` and
`outputs/` aren't committed to this repo, since they're regenerated
output, but they're easy to share directly), you can run the eval step
immediately without any key at all.

## What's actually built and proven so far

**All five originally-scoped decisions are built end-to-end and evaluated
for real**, on the identical `Decision` abstraction and the identical
calibration harness, with zero modifications to either required by any of
them — verified by cryptographic hash comparison (sha256 of every
protected module, before and after each new primitive's first live run),
not just by claim. All five now run through one unified CLI.

- **evidence-sufficiency** — given a question and the evidence retrieved so
  far, is that enough to answer, or should retrieval run another round? An
  *iterative, multi-round* stopping decision, asked after every retrieval
  round.
- **chunk-boundary** — given a candidate split point in a document, is this
  a coherent boundary or should the text stay merged? A *one-shot,
  pre-retrieval* decision, the structural opposite of sufficiency's loop.
- **context-selection** — given a retrieved candidate passage, does it earn
  a slot in the context sent to the generator? A *one-shot, per-item
  filter/select* decision — evaluated two ways (see below).
- **answer-abstain** — given the question, the retrieved evidence, and the
  answer the generator actually produced, does that answer stay grounded in
  the evidence, or should it be suppressed? A *one-shot, post-generation*
  grounding check — the only one that looks at generated output rather
  than input.
- **cache-trust** — given a semantic-cache hit, is the cached answer safe to
  serve, or should the system regenerate? A *one-shot* decision over purely
  structural state (similarity score, entry age, scoping — never the
  cached content itself, by design). Extracted from real, shipped prior art
  and independently corroborated by a second, blind design attempt that
  converged on the same architecture without ever seeing the first.

Concretely, what exists:

- **A `Decision` protocol** (`jevrag/decision.py`) — `ask(state, questions) ->
  DecisionResult`, with Jev as the live backend and a swappable interface
  verified by plugging in a second, trivial backend without touching anything
  else. Also verified against a non-text, purely structural state and a
  multi-question single call — neither assumption held only for text-shaped,
  single-question decisions.
- **Five primitives** (`jevrag/primitives/`) — `sufficiency.py`,
  `chunk_boundary.py`, `context_selection.py`, `answer_abstain.py`,
  `cache_trust.py` (plus `cache_safety_check.py`, an independently-designed
  second implementation of the cache-trust decision, built blind without
  reading the first — see Results below).
- **A calibration-first evaluation harness** (`jevrag/eval/`) — computes AURC
  and risk-coverage curves (a standard method from the selective-prediction
  literature for scoring how well a confidence signal separates likely-correct
  from likely-wrong answers), plus ECE, MCE, Brier score and Brier skill score
  (standard calibration metrics: is a stated confidence of 0.9 actually right
  about 90% of the time?). This harness is what turns "the model said it was
  confident" into a number you can actually trust or distrust — and it is the
  same code, unmodified, behind every one of the five primitives above.
- **A fixed-iteration baseline** (`jevrag/baselines/fixed_iteration.py`) — the
  same retrieval and generation stack with the gate disabled, always running a
  fixed number of rounds. This is what the gated sufficiency version has to
  beat, or at least match more cheaply.
- **Two wrapped/reproduced external comparisons** — a real, third-party
  package (`rag-jev`, PyPI) wrapped via adapter for context-selection, and a
  real, independently-run reproduction of its own published benchmark result
  (see Results below); and a free, logprob-based confidence signal (`rag-gate`)
  benchmarked directly against answer-abstain on identical ground truth.
- **HotpotQA benchmark wiring** — a public multi-hop question-answering
  dataset, with a frozen 300/700 validation/test split, scored with standard
  exact-match and F1 metrics.
- **BEIR/SciFact benchmark wiring** — a scientific claim-verification
  retrieval benchmark, used specifically to reproduce and compare against a
  real external package's own published result on its own home benchmark,
  not just on a dataset of convenience.
- **DocBench wiring** (`jevrag/benchmarks/docbench.py`) — a second,
  structurally different document benchmark (real PDFs, not short-span QA).
  Used to independently test all five primitives on two separate real
  documents, with two different generators — see Results below.
- **A CLI** — `jevrag eval sufficiency|chunk-boundary|context-selection|
  answer-abstain|cache-trust`, one command per primitive that reads
  pre-generated records and prints the full report: accuracy, calibration,
  risk-coverage, cost, and (where applicable) a baseline comparison. Every
  primitive is reachable this way, not just the first one built.

## The full CLI, all five primitives

The Quickstart above covers sufficiency end to end. Every other primitive
follows the identical records-in/report-out shape — generate with that
primitive's own `scripts/eval_*.py` script, then evaluate:

```bash
jevrag eval chunk-boundary --records outputs/chunk_boundary_wikipedia.jsonl
jevrag eval context-selection --records outputs/context_selection_hotpotqa_val.jsonl
jevrag eval answer-abstain --records outputs/answer_abstain_val.jsonl
jevrag eval cache-trust --records outputs/cache_trust_val.jsonl
```

None of the `jevrag eval` commands call a model or retriever themselves —
generating records and evaluating them are kept deliberately separate, so
the same eval command works identically on a run from five minutes ago
or five days ago. Full flag reference for any primitive:
`jevrag eval <primitive> --help`. See `INTERFACE.md` for the exact
records shape sufficiency expects; each other primitive documents its own
shape in its module docstring.

## Results

### evidence-sufficiency (n=700 test questions) — the headline result

**The gate matches baseline accuracy while retrieving 38.5% less — and that
claim was actually tested, not just eyeballed.**

| | Gated (sufficiency) | Baseline (always 3 rounds) | Difference |
|---|---|---|---|
| Exact match | 0.4129 | 0.4057 | +0.0071 (not statistically significant) |
| F1 | 0.5398 | 0.5304 | +0.0094 |
| Total retrieval rounds | 1291 | 2100 | **−38.5%** |

The rounds reduction is a direct count — the gate simply makes fewer
retrieval calls, no statistics needed to know that's true. The small accuracy
edge is a different kind of claim, and it was checked with a paired
statistical test (McNemar's test, the standard test for comparing two methods
on the same set of questions): on the 19 questions where the two methods
actually disagreed, the result was **not statistically significant**
(p = 0.36). At this sample size, a handful of disagreeing questions isn't
enough to say the gate is genuinely more accurate rather than just tied.

So the number actually being claimed is: **equal accuracy, for over a third
less retrieval work.** That's still a real, useful result — it's exactly what
a good stopping decision should do — but it's reported as what the data
supports, not rounded up to "the gate wins."

**Is the model's stated confidence trustworthy?** Not fully, and this was
measured rather than assumed. The confidence Jev returns *ranks* good
evidence above bad evidence reasonably well (its AURC of 0.4470 sits well
between a perfect signal's 0.1724 and a useless signal's 0.5871). But its
absolute numbers are not well calibrated: on average, a stated confidence is
off by about 33 percentage points from the actual accuracy at that confidence
level (ECE = 0.3322), and it does *worse* than a naive constant guess by one
standard measure (Brier skill = −0.4486). The clearest example: on 354 of the
700 questions — half the dataset — the model reported roughly 95% confidence,
and was right only 53% of the time. So the model is often very sure of
itself, and wrong nearly half the time when it is. This kind of finding is
exactly what a calibration harness is for: catching a confidence signal that
sounds trustworthy but isn't, rather than repeating a vendor's claim.

**Does a threshold picked on one dataset transfer to another?** Tuning the
gate's cutoff to hit 90% coverage on the validation set and applying it
unchanged to the test set carries over reasonably well on coverage (90.3% to
91.3%) but loses about 5.6 points of accuracy in the process (49.5% to
43.8%). Thresholds tuned in one place don't always hold up somewhere else —
this is a known, general finding in this kind of work, and it holds here too.

A smaller, earlier batch of 300 validation questions was generated in two
batches using two different hosting setups for the same underlying model,
because the first hosting provider ran out of usable credit partway through.
The two batches disagreed on accuracy by about 15 percentage points — a real,
investigated (not corrupted-data, not a bug) difference in how the two
setups behave, reported as two disclosed subsets rather than averaged.

### chunk-boundary — the generalization test

The whole reason chunk-boundary exists: evidence-sufficiency alone can only
prove the abstraction works for *one* decision. Chunk-boundary is
structurally its opposite (one-shot vs. iterative), so plugging it into the
identical `Decision` protocol and calibration harness without touching
either is the actual acceptance test — and it passed, hash-verified before
and after the live run, on a 60-paragraph Wikipedia article with 119
candidate boundaries.

It essentially **tied** a simple cosine-similarity baseline on ranking
quality (AURC 0.2798 vs. 0.2894) there. But its calibration was dramatically
better than sufficiency's, on the identical backend: ECE 0.087 and Brier
skill +0.21, versus sufficiency's ECE 0.33 and Brier skill −0.45. **This is
the project's central calibration finding: Jev's confidence is not
uniformly trustworthy or untrustworthy — it depends on the specific
question being asked.** "Is Jev well-calibrated?" doesn't have one answer;
it depends what you ask it.

**Retested twice on real documents, with a genuinely more informative
picture than Wikipedia's alone.** A first real-document test used invalid
ground truth (PDF page breaks mistaken for real paragraph breaks) and was
correctly refused as uninterpretable rather than reported as a bad number.
Corrected with real, human-verified section boundaries on the same
document: Jev clearly beat cosine similarity this time (AURC 0.518 vs.
0.654, both baselines moving from noise to real signal together,
validating the fix), 100% recall on all 15 true section breaks, with every
ranking error running in the safe direction for ingestion chunking
(over-splitting inside long sections, never merging distinct topics). A
second real document (a different paper, section *and* subsection
boundaries) showed the pattern is real but not perfectly stable across
documents: recall dropped to 72% there, with the instability traced
specifically to subsection-level granularity — and even there, every
miss but one was a benign same-topic merge, never a dangerous cross-topic
one.

### context-selection — wrapped vs. rebuilt, on two real benchmarks

Real, already-shipped prior art exists for this decision (`rag-jev` on
PyPI, evaluated by its own authors on the BEIR/SciFact benchmark). Rather
than choosing between wrapping that package or building a from-scratch
implementation, both were built and compared directly on the same harness.

**On HotpotQA** (matched sample, n=192 candidate passages, same 20
questions, same candidate construction verified down to zero mismatches):

| | AURC | ECE | Brier skill |
|---|---|---|---|
| From-scratch (this repo) | 0.5412 | 0.1542 | +0.239 |
| Wrapped `rag-jev` adapter | 0.5550 | 0.1805 | +0.121 |

The AURC edge here was run through a paired significance test (paired
bootstrap, 10,000 resamples): **real, p = 0.038** — though the confidence
interval sits close enough to zero ([+0.0008, +0.0299]) that the honest
description is "real but modest," not a decisive win.

**On SciFact** (`rag-jev`'s own home benchmark, n=300 queries — the same
scale as their published evaluation): the from-scratch implementation's own
NDCG@10 reproduced `rag-jev`'s published headline number (0.7513) to
within 0.012, and edged the wrapped adapter's reproduction by a further
0.009 (0.7479 vs. 0.7390) — but **neither implementation actually beat the
published number**; both land within noise of it, and the small edge over
the wrapped adapter did not clear significance either (p = 0.234). This is
the honest reading: a real, small, reproducible edge for the from-scratch
build over the general-purpose wrapped package on HotpotQA specifically —
not a claim of beating third-party published work, and not a claim that
holds up equally on every benchmark tested.

### answer-abstain — a post-generation grounding check, benchmarked against a free alternative

Given the question, the retrieved evidence, and an already-generated
answer, does the answer actually stay grounded? On 30 real HotpotQA
questions: filtering out the gate's abstentions (27% abstain rate) lifts
the exact-match accuracy of what actually reaches the user from 0.53
(unfiltered) to 0.68 — a real, substantial quality improvement, not a
sampling artifact.

This decision was also benchmarked directly against a free, already-built
alternative: a logprob-based trust signal (mean token log-probability of
the generation, no extra model call needed) from a separate open-source
tool. The two signals turned out to be **essentially tied on ranking
quality** (AURC 0.154 vs. 0.163) — confirmed with a paired significance
test (p = 0.879, a confidence interval more than ten times wider than the
observed delta), not just judged as close by eye. The free signal is a
real, legitimate alternative, not something this approach clearly
obsoletes.

### cache-trust — extracted from real prior art, independently corroborated

Real, shipped prior art exists for this decision (a cache-safety gate
embedded in an open-source chat application's code) — not a standalone
package, so wrapping it meant real extraction work, not a thin adapter.
The extracted design (`cache_trust.py`) was ported faithfully: a purely
structural state (similarity score, margin, entry age, TTL, conversation
scoping — never the cached content itself, by the original design's own
privacy-motivated choice) and two typed questions in one call, the first
real test of both a non-text state and a multi-question call in this
project. Both worked cleanly on the first live attempt.

An easy first evaluation scored near-perfectly (AURC 0.146, Brier skill
+0.96) — honestly flagged as measuring "the gate reads a clean signal
correctly," not hard discrimination, since the test scenario was
naturally bimodal. A deliberate, harder follow-up built genuinely
ambiguous, near-threshold cases: performance held up close to oracle-level
in exactly that harder region (AURC 0.043 vs. an oracle of 0.041), and
every real error observed across both evaluations ran in the safe
direction — the gate has never served a wrong cached answer, only
occasionally regenerated when the cached one would have been fine.

**Independently corroborated, not just self-evaluated.** A second design,
built cold with zero exposure to the first implementation or its source
material, converged on the same core architecture — same structural-only
state, same two signal axes, same threshold-style verdict — real evidence
the design is the natural solution to this decision, not an arbitrary one.

### Real documents, two of them, all five primitives, two different generators

Beyond the paired-benchmark comparisons above, all five primitives were
run independently (not chained) against two real, complete documents — an
academic paper and a RAG research paper — using two different generators
(a low-volume OpenAI-compatible model, and Gemini, integrated for the
first time). Each run ended in an explicit, pre-agreed pass/fail verdict
per primitive against a stated numeric bar, not just raw numbers left for
interpretation. Context-selection passed cleanly both times with the
largest margins measured anywhere in the project; sufficiency and
answer-abstain passed but remain data-thin at this scale; chunk-boundary
passed on one document and failed the strict recall bar on the second, for
the reason described above. Cache-trust was correctly skipped both times
rather than forced through a scenario that would have added nothing new.

### The standing calibration finding, across every real measurement

Across every decision measured — different domains (multi-hop QA,
document structure, scientific claim verification, post-generation
grounding, cache safety), different implementations, the same Jev backend
throughout — calibration quality varies by more than a full Brier-skill
point, from −0.45 (worst: sufficiency, an iterative stopping decision with
a skewed outcome distribution) to +0.96 (best: cache-trust, on an
admittedly easy first scenario). **The pattern that holds across every
measurement: Jev's confidence reliably *ranks* correct above incorrect —
every single AURC measured has sat meaningfully between the oracle and
random baselines — but its *absolute* calibration is a property of the
specific decision and its base-rate structure, not a fixed trait of the
backend.** A sparse or skewed positive rate (sufficiency's stopping
decision, SciFact's ~5% relevance rate) consistently produces worse
absolute calibration than a more balanced one, regardless of which
decision or implementation is being scored.

## Reproducing this

The retrieval logic, the prompt shaping, and the sufficiency/baseline loops
all live in this repository and are portable — `scripts/produce_records.py`
runs against any OpenAI-compatible endpoint. The actual records behind the
sufficiency results above were generated using a locally-run open model on
a cloud GPU notebook; that notebook isn't included in this repository since
it's throwaway infrastructure specific to one hosting setup, not part of the
reusable library. The other primitives' results were generated with a mix
of hosted API calls (OpenRouter, Gemini) — small, cheap runs, reproducible
against any equivalent account.

## Project layout

```
jevrag/
  decision.py              # the Decision interface + the Jev backend
  _rag_gate.py             # single import bridge to RAG-Gate's tested selective.py/evaluator.py
  primitives/
    sufficiency.py         # iterative, multi-round stopping decision
    chunk_boundary.py      # one-shot, pre-retrieval split/merge decision
    context_selection.py   # one-shot, per-passage filter/select decision
    answer_abstain.py      # one-shot, post-generation grounding decision
    cache_trust.py         # one-shot, structural-state cache-serve decision (extraction)
    cache_safety_check.py  # independent second design of the cache-trust decision
  adapters/
    rag_jev_selector.py    # wraps the real rag-jev PyPI package for comparison
  eval/
    calibration.py         # AURC/risk-coverage + ECE/MCE/Brier — shared, unmodified across all five primitives
    cost.py                # token/latency accounting
  benchmarks/
    hotpotqa.py             # dataset loading + scoring
    scifact.py               # BEIR/SciFact loading + NDCG@10 scoring
    docbench.py               # real-document benchmark (two documents tested)
    llm_judge.py               # LLM-judge scoring for benchmarks without exact-match ground truth
  baselines/
    fixed_iteration.py      # the baseline evidence-sufficiency has to match or beat
  __main__.py               # the `jevrag eval <primitive>` command, all five wired in
scripts/
  produce_records.py                # generates sufficiency's records
  eval_chunk_boundary.py            # chunk-boundary's Wikipedia eval script
  eval_chunk_boundary_doc.py        # chunk-boundary's real-document eval script
  eval_context_selection.py         # context-selection's own eval script (from-scratch primitive)
  eval_rag_jev.py                    # context-selection's wrapped-adapter eval script
  eval_scifact_ragjev.py              # rag-jev's SciFact reproduction
  eval_scifact_context_selection.py   # SciFact head-to-head, from-scratch primitive
  eval_answer_abstain.py              # answer-abstain's own eval script
  eval_cache_trust.py                 # cache-trust's own eval script
  eval_cache_trust_hard.py            # cache-trust's near-threshold hardening eval
  eval_cache_safety_check.py          # the independent second cache-decision design's eval
  eval_docbench_primitives.py         # all five primitives, one real document, independent
  demo_docbench.py                     # DocBench exploratory demo (unscored)
tests/                                 # 163 tests across all five primitives
```

`records/`, `reports/`, and `outputs/` are regenerated by running the
pipeline, not committed to the repository.
