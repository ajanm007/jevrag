# Per-primitive results

The detailed per-decision write-ups — the numbers, the caveats, the failure
modes, and what each result does *not* claim. These sections are reproduced
verbatim from the README, which now carries only the
[Results at a glance](../README.md#results-at-a-glance) table and links here.

Nothing in this file has been edited for the move except the heading level
(`###` → `##`) so anchors land at the top level. Every number below is the same
number the README used, traceable to the same raw run outputs.

Contents:

- [evidence-sufficiency](#evidence-sufficiency-n700-test-questions--the-headline-result)
- [chunk-boundary](#chunk-boundary--the-generalization-test)
- [context-selection](#context-selection--wrapped-vs-rebuilt-on-two-real-benchmarks)
- [answer-abstain](#answer-abstain--a-post-generation-grounding-check-benchmarked-against-a-free-alternative)
- [cache-trust](#cache-trust--extracted-from-real-prior-art-independently-corroborated)

> **Note on the answer-abstain section below:** it reports the original n=30
> head-to-head. That result was **superseded** by the n=300 run — see the
> [Results at a glance](../README.md#results-at-a-glance) table in the README for
> the current numbers. The n=30 text is kept here as history, not as the
> project's current claim.

---

## evidence-sufficiency (n=700 test questions) — the headline result

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

## chunk-boundary — the generalization test

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

## context-selection — wrapped vs. rebuilt, on two real benchmarks

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

## answer-abstain — a post-generation grounding check, benchmarked against a free alternative

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

A third signal was added to that same comparison later: **RAGAS's
faithfulness metric** (an LLM-judge decomposition-and-verification method),
scored on the identical 30 questions against the same EM ground truth.
The headline is the confound, not the winner: with RAGAS's own default
judge (gpt-4o-mini — the same model that generated the answers, so a
self-judge), faithfulness ranks *worst* of the three (AURC 0.189); with a
different-vendor judge (Gemini 2.5 Flash) it ranks *best* (AURC 0.100, vs.
Jev's 0.154). The judge-model swing (0.09 AURC) is larger than any
pairwise difference between the signals themselves, and no pairwise
difference is statistically significant at n=30 (paired bootstrap, p ≥
0.07; the largest gap, RAGAS[Gemini] vs. logprob, has a CI touching zero).
Plainly: at this scale the abstention-gate comparison is **confound-limited,
not signal-limited** — RAGAS's ranking depends more on the judge model than
on its mechanism. Reusable script: `scripts/eval_ragas_faithfulness.py`
(needs the `ragas` extra and live judge calls).

**Security disclosure (2026-09-24):** the pinned `ragas==0.4.3` — every
version in the `ragas` extra's supported range — has an unpatched SSRF in
its multi-modal faithfulness collections module (attacker-controlled
`retrieved_contexts` can force the library to fetch arbitrary URLs/local
paths). No patched release exists yet; the vendor did not respond to
disclosure. The `ragas` extra is never installed by default or in CI, and
this project's own use of it only scores a fixed local benchmark dataset —
but if you install `jevrag[ragas]` yourself, do not pass untrusted or
externally-sourced text as `retrieved_contexts` until a fix ships. See the
`[ragas]` extra's comment in `pyproject.toml` for the exact vulnerable
functions.

## cache-trust — extracted from real prior art, independently corroborated

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
