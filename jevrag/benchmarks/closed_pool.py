"""Closed-pool benchmark layer for the v0.3 multi-dataset suite (D1).

A "closed pool" is ONE shared, deduplicated paragraph list built from the
sampled questions' own paragraph pools, indexed once with BM25. This is the
same shape as HotpotQA's existing 9769-document index, so a suite run and the
n=700 HotpotQA headline are comparable rather than a different experiment.

What this module deliberately does NOT do: download anything, build a Wikipedia
index, or reimplement scoring. Datasets are loaded by their adapters from a
local file path (the notebook downloads); normalization and EM/F1 come from the
vendored evaluator via :mod:`jevrag.benchmarks.hotpotqa`; BM25 comes from
:func:`jevrag.benchmarks.docbench.build_bm25_index`.

Sampling is seeded and frozen, and the seed plus the sampling rule are written
into the output header so a sample is reproducible and auditable — an eval whose
500 questions can't be regenerated is not an eval.

Honesty note on the sanity report: the pool-sanity numbers (BM25 recall of gold
supporting paragraphs) are a property of the DATASET and the POOL, not of any
decision backend. They tell us how hard a closed pool is. Low recall is a
finding to report, not a bug to tune away.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Default seed for the frozen sample. Written into every output header.
DEFAULT_SEED = 20261001

#: D3 default: 500 questions split 250 calibrate / 250 report.
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_CALIBRATE = 250
DEFAULT_REPORT = 250

#: Split labels, deliberately not "train"/"test": these are both evaluation
#: splits (calibration-fit vs held-out report), and calling either one a test
#: split invites exactly the tuning-on-test confusion this project keeps
#: having to disclose.
SPLIT_CALIBRATE = "calibrate"
SPLIT_REPORT = "report"

#: BM25 recall depths reported by the pool-sanity pass. k=15 is included
#: because the sufficiency loop reads 5 paragraphs per round for up to 3 rounds
#: (5 / 10 / 15), so 15 is a depth the pipeline actually uses.
RECALL_KS = (5, 10, 15, 20)

#: Pool modes. `shared` is one deduplicated pool over every sampled question's
#: paragraphs (MuSiQue-questionable, and comparable to HotpotQA's 9,769-doc
#: index). `per_question` retrieves only over the paragraphs the question itself
#: ships — MuSiQue's standard distractor setting, where the gold is always
#: present. They are DIFFERENT experiments and the mode travels in every header:
#: the same recall number means something completely different in each.
POOL_MODE_SHARED = "shared"
POOL_MODE_PER_QUESTION = "per_question"
DEFAULT_POOL_MODE = POOL_MODE_SHARED


def _canonical(paragraph: dict[str, Any]) -> tuple[str, str]:
    """The identity used for deduplication: (title, text), whitespace-collapsed.

    Normalizing whitespace before comparing means two copies of a paragraph that
    differ only in trailing spaces or newlines collapse to ONE pool entry —
    otherwise "dedup" silently under-deduplicates and the pool count drifts
    between runs on different platforms.
    """
    return (" ".join(str(paragraph.get("title", "")).split()),
            " ".join(str(paragraph.get("text", "")).split()))


class QuestionRecord(dict):
    """One sampled question in the common shape.

    A plain dict subclass (not a dataclass) so it serializes straight to JSONL
    and stays drop-in for the existing loop's question dicts, which are plain
    dicts with ``id``/``question``/``answer``/``type``.

    Keys:
        id: stable string id from the dataset.
        question: the question text.
        answer: gold answer string (the canonical/first form).
        aliases: every other acceptable answer form; EM matches any of them.
        type: dataset-specific type label, or "" when the dataset has none.
        hops: integer hop count when the dataset gives one, else None.
        split: SPLIT_CALIBRATE or SPLIT_REPORT.
        supporting_ids: paragraph ids in the closed pool that are gold for this
            question (empty for datasets with no supporting annotation).
    """

    def __init__(self, *, id: str, question: str, answer: str,
                 aliases: Sequence[str] = (), type: str = "",
                 hops: int | None = None, split: str = "",
                 supporting_ids: Sequence[str] = ()) -> None:
        super().__init__(
            id=id, question=question, answer=answer,
            aliases=list(aliases), type=type, hops=hops, split=split,
            supporting_ids=list(supporting_ids),
        )

    @property
    def all_answers(self) -> list[str]:
        """Gold answer first, then aliases — the full EM match set."""
        seen, out = set(), []
        for a in [self["answer"], *self["aliases"]]:
            key = str(a).strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out


def make_question(*, id: str, question: str, answer: str,
                  aliases: Iterable[str] = (), type: str = "",
                  hops: int | None = None, split: str = "",
                  supporting_ids: Iterable[str] = ()) -> QuestionRecord:
    """Constructor for the common record shape (used by every adapter)."""
    return QuestionRecord(id=id, question=question, answer=answer,
                          aliases=list(aliases), type=type, hops=hops,
                          split=split, supporting_ids=list(supporting_ids))


# PLACEHOLDER_SAMPLER
def stratify_key(record: dict[str, Any]) -> str:
    """Stratification bucket for one question.

    Prefers hop count (MuSiQue, and the multi-hop sets generally), then type
    (HotpotQA bridge/comparison, 2Wiki's four types), then a single bucket.
    Documented as a preference order rather than sniffed, because a dataset
    with both fields should stratify on the one that actually varies difficulty.
    """
    hops = record.get("hops")
    if hops is not None:
        return f"hops={hops}"
    qtype = str(record.get("type") or "").strip()
    return f"type={qtype}" if qtype else "type=unknown"


def sample_questions(records: Sequence[dict[str, Any]],
                     n: int = DEFAULT_SAMPLE_SIZE,
                     seed: int = DEFAULT_SEED
                     ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seeded, stratified sample of ``n`` questions, plus a header describing how.

    Deterministic for a given (input order, n, seed): the same seed returns the
    same questions in the same order, which is what makes the sample auditable.
    Stratification is proportional-within-bucket by largest-remainder, so a rare
    bucket is not silently rounded away to zero.

    Returns ``(sample, header)``. The header carries the seed, the rule, and the
    counts, and belongs in the output file so the sample can be regenerated and
    checked rather than taken on trust.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    pool = [dict(r) for r in records]
    if len(pool) < n:
        raise ValueError(
            f"asked for {n} questions but the dataset has {len(pool)}")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in pool:
        buckets.setdefault(stratify_key(r), []).append(r)

    rng = random.Random(seed)
    for key in sorted(buckets):
        # Sort by id first so the bucket's internal order does not depend on
        # dict/file ordering, then shuffle deterministically from the seed.
        ordered = sorted(buckets[key], key=lambda r: str(r.get("id", "")))
        rng.shuffle(ordered)
        buckets[key] = ordered

    total = len(pool)
    exact = {k: len(v) * n / total for k, v in buckets.items()}
    take = {k: int(v) for k, v in exact.items()}
    # Largest-remainder: hand leftover questions to the buckets with the biggest
    # fractional claim, ties broken by name for determinism.
    order = sorted(buckets, key=lambda k: (-(exact[k] - take[k]), k))
    for k in order[: n - sum(take.values())]:
        take[k] += 1

    sample: list[dict[str, Any]] = []
    for k in sorted(buckets):
        sample.extend(buckets[k][: min(take[k], len(buckets[k]))])

    # Top up deterministically if any bucket was short of its share.
    if len(sample) < n:
        chosen = {str(r.get("id", "")) for r in sample}
        leftovers = sorted((r for r in pool
                            if str(r.get("id", "")) not in chosen),
                           key=lambda r: str(r.get("id", "")))
        sample.extend(leftovers[: n - len(sample)])

    header = {
        "seed": seed,
        "requested_n": n,
        "sampled_n": len(sample),
        "source_n": total,
        "stratify_rule": ("proportional by hops when present, else by type; "
                          "largest-remainder allocation; within-bucket order "
                          "sorted by id then seeded shuffle"),
        "strata_counts": {k: len(v) for k, v in sorted(buckets.items())},
        "strata_taken": {k: min(take[k], len(buckets[k])) for k in sorted(buckets)},
    }
    return sample, header


def split_calibrate_report(sample: Sequence[dict[str, Any]],
                           n_calibrate: int = DEFAULT_CALIBRATE,
                           n_report: int = DEFAULT_REPORT,
                           seed: int = DEFAULT_SEED
                           ) -> tuple[list[dict[str, Any]],
                                      list[dict[str, Any]], dict[str, Any]]:
    """Split the sample into disjoint calibrate / report halves.

    Split AFTER sampling and stratified the same way, so both halves carry the
    same bucket mix — an unlucky split putting every 4-hop question on one side
    would make the halves non-comparable. Returns copies; input is not mutated.
    """
    if n_calibrate + n_report > len(sample):
        raise ValueError(
            f"split {n_calibrate}+{n_report} exceeds sample size {len(sample)}")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in sample:
        buckets.setdefault(stratify_key(r), []).append(dict(r))

    rng = random.Random(seed + 1)  # distinct from the sampling seed on purpose
    exact = {k: len(v) * n_calibrate / len(sample) for k, v in buckets.items()}
    take = {k: int(v) for k, v in exact.items()}
    order = sorted(buckets, key=lambda k: (-(exact[k] - take[k]), k))
    for k in order[: n_calibrate - sum(take.values())]:
        take[k] += 1

    calibrate: list[dict[str, Any]] = []
    for k in sorted(buckets):
        rows = sorted(buckets[k], key=lambda r: str(r.get("id", "")))
        rng.shuffle(rows)
        calibrate.extend(rows[: min(take[k], len(rows))])

    cal_ids = {str(r.get("id", "")) for r in calibrate}
    report = sorted((dict(r) for r in sample
                     if str(r.get("id", "")) not in cal_ids),
                    key=lambda r: str(r.get("id", "")))

    for r in calibrate:
        r["split"] = SPLIT_CALIBRATE
    for r in report:
        r["split"] = SPLIT_REPORT

    header = {
        "split_seed": seed + 1,
        "n_calibrate": len(calibrate),
        "n_report": len(report),
        "requested_calibrate": n_calibrate,
        "requested_report": n_report,
        "disjoint": not (cal_ids & {str(r.get("id", "")) for r in report}),
    }
    return calibrate, report, header


def build_pool(questions: Sequence[dict[str, Any]]
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the ONE shared, deduplicated paragraph pool for a question sample.

    Each question carries its own paragraphs (via the adapter's
    ``paragraphs_for`` call). All of them are concatenated, deduplicated on
    (title, text) with whitespace normalized, and given STABLE ids — the id is
    the index of the paragraph's first appearance in sorted-question order, so
    the same sample always yields the same ids on any machine. Stability is the
    whole point: a pool id that shifts between runs makes every downstream
    recall number incomparable.

    Returns ``(pool, header)`` where each pool entry is
    ``{"id": "p0", "title": ..., "text": ...}`` and the header records the
    paragraph counts (total seen, unique kept, duplicates removed).
    """
    pool: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    total_seen = 0

    # Sorted by question id so pool ids do not depend on input order.
    for q in sorted(questions, key=lambda r: str(r.get("id", ""))):
        for para in _paragraphs_for(q):
            total_seen += 1
            key = _canonical(para)
            if not key[1]:
                continue  # empty text: not retrievable, not a real paragraph
            if key in seen:
                continue
            pid = f"p{len(pool)}"
            seen[key] = pid
            pool.append({"id": pid, "title": key[0], "text": key[1]})

    header = {
        "n_questions": len(questions),
        "paragraphs_seen": total_seen,
        "paragraphs_unique": len(pool),
        "duplicates_removed": total_seen - len(pool),
        "dedup_key": "(title, text) with whitespace collapsed",
        "id_rule": "p<index of first appearance, questions visited in sorted-id order>",
    }
    return pool, header


def _paragraphs_for(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapters attach their per-question paragraphs under ``_paragraphs``.

    Kept as a private key rather than a public field because it is large and is
    stripped before the sample is written out — the written record keeps only
    ``supporting_ids``, which is what the pool actually needs afterwards.
    """
    return list(question.get("_paragraphs") or [])


def resolve_supporting_ids(questions: Sequence[dict[str, Any]],
                           pool: Sequence[dict[str, Any]]
                           ) -> list[dict[str, Any]]:
    """Rewrite each question's ``supporting_ids`` into closed-pool ids.

    Datasets identify gold paragraphs per question by their OWN local index. After
    deduplication those indices no longer address the shared pool, so each
    question's ``_supporting_local`` positions are mapped through the dedup table
    to the pool id they landed on. A gold paragraph whose text was empty and
    dropped therefore resolves to nothing — recorded, not hidden.
    """
    canonical_to_pid: dict[tuple[str, str], str] = {
        (p["title"], p["text"]): p["id"] for p in pool
    }
    out = []
    for q in questions:
        rec = dict(q)
        rec.pop("_paragraphs", None)
        ids: list[str] = []
        missing = 0
        for pos in rec.pop("_supporting_local", []) or []:
            paras = _paragraphs_for(q)
            if not isinstance(pos, int) or not (0 <= pos < len(paras)):
                missing += 1
                continue
            pid = canonical_to_pid.get(_canonical(paras[pos]))
            if pid is None:
                missing += 1
            else:
                ids.append(pid)
        rec["supporting_ids"] = ids
        rec["supporting_unresolved"] = missing
        out.append(rec)
    return out


def score_with_aliases(prediction: str, record: dict[str, Any]) -> dict[str, Any]:
    """EM and F1 against the full alias set, via the vendored evaluator.

    Normalization is NOT reimplemented — ``exact_match``/``f1_score`` come from
    RAG-Gate through :mod:`jevrag.benchmarks.hotpotqa`, the same functions the
    HotpotQA path uses. EM is "matches ANY acceptable answer"; F1 takes the BEST
    scoring alias, so a prediction matching a less-canonical alias is not
    penalized below one matching the primary form.

    Returns ``{"em": bool, "f1": float, "matched_alias": str | None}``.
    """
    from .hotpotqa import exact_match, f1_score

    # Accept either a QuestionRecord or a plain dict; read the fields directly
    # rather than reconstructing the record, which would be fragile.
    gold: list[str] = []
    for a in [record.get("answer"), *(record.get("aliases") or [])]:
        key = str(a).strip()
        if key and key not in gold:
            gold.append(key)
    if not gold:
        raise ValueError(f"question {record.get('id')!r} has no gold answer")

    pred = prediction or ""
    em = any(exact_match(pred, g) for g in gold)
    f1 = max((f1_score(pred, g) for g in gold), default=0.0)
    matched = next((g for g in gold if exact_match(pred, g)), None)
    return {"em": bool(em), "f1": float(f1), "matched_alias": matched}


def pool_sanity_report(questions: Sequence[dict[str, Any]],
                       pool: Sequence[dict[str, Any]],
                       retrieve_fn, ks: Sequence[int] = RECALL_KS,
                       pool_mode: str = POOL_MODE_SHARED
                       ) -> dict[str, Any]:
    """How hard is this pool? Two DISTINCT metrics, explicitly named.

    The earlier version shipped one name (``gold_recall_at_k``) for
    paragraph-level recall and another (``per_hop.recall_at_max_k``) for "all
    golds found", while the orientation text described the latter. Two different
    quantities under confusingly similar names means a reader picking the wrong
    one gets a number that is simply wrong. Now:

    - ``paragraph_recall_at_k[k]`` -- PARAGRAPH-level: of all gold supporting
      paragraphs across all questions, the fraction found in the top-k. Partial
      credit: 3 of 4 golds found contributes 0.75.
    - ``all_gold_found_at_k[k]`` -- QUESTION-level: the fraction of questions
      whose EVERY gold paragraph is in the top-k. All-or-nothing.

    Both are reported for every k, overall, per hop bucket, and per the
    dataset's own id prefix. They answer different questions and differ a lot in
    practice; neither substitutes for the other.

    Questions with no supporting annotation are excluded from both denominators
    and counted, never scored as misses. A question that DECLARED gold but
    resolved none of it is a build failure, reported separately.
    """
    ks = sorted(ks)
    max_k = max(ks) if ks else 1
    pool_ids = {p["id"] for p in pool}

    n_with_gold = gold_total = gold_in_pool = 0
    missing_detail: list[dict[str, Any]] = []
    para_hits = {k: 0 for k in ks}
    all_found = {k: 0 for k in ks}
    by_hop: dict[str, dict[str, Any]] = {}
    by_prefix: dict[str, dict[str, Any]] = {}

    def _bucket(store, key):
        return store.setdefault(str(key), {
            "questions": 0, "gold_total": 0,
            "para_hits": {k: 0 for k in ks},
            "all_found": {k: 0 for k in ks}})

    for q in questions:
        gold_ids = [g for g in q.get("supporting_ids", []) if g in pool_ids]
        if q.get("supporting_unresolved"):
            n_with_gold += 1
            missing_detail.append({
                "id": q.get("id"),
                "gold_ids": list(q.get("supporting_ids", [])),
                "resolved_ids": gold_ids,
                "unresolved": q["supporting_unresolved"]})
            continue
        if not q.get("supporting_ids"):
            continue
        n_with_gold += 1
        gold_total += len(q["supporting_ids"])
        gold_in_pool += len(gold_ids)
        if len(gold_ids) != len(q["supporting_ids"]):
            missing_detail.append({
                "id": q.get("id"),
                "gold_ids": list(q["supporting_ids"]),
                "resolved_ids": gold_ids,
                "unresolved": q.get("supporting_unresolved", 0)})
        if not gold_ids:
            continue

        # Per-question mode requires the caller to select this question's pool.
        if hasattr(retrieve_fn, "bind_question_id"):
            retrieve_fn.bind_question_id(str(q.get("id")))
        got = [c.get("id") for c in retrieve_fn(q["question"], max_k)]

        hop_b = _bucket(by_hop, q.get("hops") if q.get("hops") is not None
                        else "none")
        pfx_b = _bucket(by_prefix, q.get("id_prefix") or q.get("type")
                        or "unknown")
        for b in (hop_b, pfx_b):
            b["questions"] += 1
            b["gold_total"] += len(gold_ids)

        for k in ks:
            found = len([g for g in gold_ids if g in got[:k]])
            para_hits[k] += found
            af = int(found == len(gold_ids))
            all_found[k] += af
            for b in (hop_b, pfx_b):
                b["para_hits"][k] += found
                b["all_found"][k] += af

    def _finish(b):
        nq, gt = b["questions"], b["gold_total"]
        return {"questions": nq, "gold_paragraphs": gt,
                "paragraph_recall_at_k": {
                    str(k): (b["para_hits"][k] / gt) if gt else None for k in ks},
                "all_gold_found_at_k": {
                    str(k): (b["all_found"][k] / nq) if nq else None for k in ks}}

    return {
        "pool_mode": pool_mode,
        "n_questions": len(questions),
        "n_questions_with_gold": n_with_gold,
        "n_questions_without_gold": len(questions) - n_with_gold,
        "pool_size": len(pool),
        "gold_paragraphs_total": gold_total,
        "gold_paragraphs_in_pool": gold_in_pool,
        "gold_in_pool_rate": (gold_in_pool / gold_total) if gold_total else None,
        "questions_with_missing_gold": missing_detail[:50],
        "n_questions_with_missing_gold": len(missing_detail),
        "paragraph_recall_at_k": {
            str(k): (para_hits[k] / gold_total) if gold_total else None for k in ks},
        "all_gold_found_at_k": {
            str(k): (all_found[k] / n_with_gold) if n_with_gold else None
            for k in ks},
        "per_hop": {h: _finish(b) for h, b in sorted(by_hop.items())},
        "per_id_prefix": {p: _finish(b) for p, b in sorted(by_prefix.items())},
        "metric_definitions": {
            "paragraph_recall_at_k": (
                "PARAGRAPH-level. Of all gold supporting paragraphs across all "
                "questions with gold, the fraction retrieved in the top-k. "
                "Partial credit: 3 of 4 golds found contributes 0.75."),
            "all_gold_found_at_k": (
                "QUESTION-level. Fraction of questions whose EVERY gold "
                "supporting paragraph is in the top-k. All-or-nothing."),
        },
        "orientation": (
            "both metrics are over GOLD supporting paragraphs only; questions "
            "with no supporting annotation are excluded from both denominators "
            "and counted in n_questions_without_gold, never scored as misses"),
    }


def write_jsonl_with_header(path: str | Path, rows: Sequence[dict[str, Any]],
                            header: dict[str, Any]) -> None:
    """Write ``header`` as the first line, then one JSON object per row.

    The header line is how a sample stays auditable: seed, rule and counts
    travel with the data instead of living only in someone's notes.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__header__": header}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl_with_header(path: str | Path
                           ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inverse of :func:`write_jsonl_with_header`."""
    header: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "__header__" in rec:
                header = rec["__header__"]
            else:
                rows.append(rec)
    return header, rows


def build_index(pool: Sequence[dict[str, Any]]):
    """BM25 index over the shared pool, reusing docbench's builder verbatim."""
    from .docbench import build_bm25_index

    return build_bm25_index([{"title": p["title"], "text": p["text"]}
                             for p in pool])


def build_per_question_pools(questions: Sequence[dict[str, Any]]
                             ) -> tuple[dict[str, list[dict[str, Any]]],
                                        dict[str, Any]]:
    """Per-question pools, deduplicated WITHIN each question only.

    Each question keeps its own paragraphs (MuSiQue ships 20, gold included), so
    retrieval never has to distinguish its evidence from another question's.
    Cross-question duplicates are deliberately NOT collapsed here: sharing a
    paragraph between two questions' pools is what the per-question mode is
    supposed to prevent. Ids are the question's own `p0..pN`, scoped by question.

    Returns ``({qid: [para, ...]}, header)``.
    """
    pools: dict[str, list[dict[str, Any]]] = {}
    total = dupes = 0
    for q in sorted(questions, key=lambda r: str(r.get("id", ""))):
        seen: set[tuple[str, str]] = set()
        out: list[dict[str, Any]] = []
        for para in _paragraphs_for(q):
            total += 1
            key = _canonical(para)
            if not key[1]:
                continue
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            out.append({"id": f"p{len(out)}", "title": key[0], "text": key[1]})
        pools[str(q.get("id"))] = out

    sizes = [len(v) for v in pools.values()]
    header = {
        "pool_mode": POOL_MODE_PER_QUESTION,
        "n_questions": len(pools),
        "paragraphs_seen": total,
        "paragraphs_total_after_dedup": sum(sizes),
        "duplicates_removed_within_question": dupes,
        "paragraphs_per_question": {
            "min": min(sizes) if sizes else None,
            "max": max(sizes) if sizes else None,
            "mean": (sum(sizes) / len(sizes)) if sizes else None,
        },
        "dedup_key": "(title, text) with whitespace collapsed, WITHIN a question",
    }
    return pools, header


def make_per_question_retrieve_fn(pools: dict[str, list[dict[str, Any]]]):
    """``retrieve_fn`` that searches ONLY the question's own pool.

    Signature matches the shared version but keys on the question text via a
    lookup built from the caller's question ids -- see `bind_question_id` for how
    the caller wires a question to its pool. Kept deliberately explicit rather
    than guessing from the question string.
    """
    from .docbench import bm25_retrieve

    indexes: dict[str, Any] = {}
    documents: dict[str, list[dict[str, Any]]] = {}
    for qid, paras in pools.items():
        documents[qid] = paras
        indexes[qid] = build_index(paras)

    current = {"qid": None}

    def bind_question_id(qid: str) -> None:
        """Select which question's pool the next retrieve call searches."""
        current["qid"] = str(qid)

    def retrieve_fn(question: str, n_docs: int) -> list[dict[str, Any]]:
        qid = current["qid"]
        if qid is None or qid not in indexes:
            raise RuntimeError(
                "per_question retrieve_fn used before bind_question_id(qid); "
                "each question must select its own pool first")
        docs = documents[qid]
        return bm25_retrieve(question, indexes[qid], docs,
                             top_k=n_docs)["chunks"]

    retrieve_fn.bind_question_id = bind_question_id
    retrieve_fn.pool_mode = POOL_MODE_PER_QUESTION
    retrieve_fn.n_pools = len(pools)
    return retrieve_fn


def resolve_supporting_ids_per_question(questions: Sequence[dict[str, Any]],
                                        pools: dict[str, list[dict[str, Any]]]
                                        ) -> list[dict[str, Any]]:
    """Same contract as :func:`resolve_supporting_ids`, but ids are per-question.

    Public keys and strip-private behaviour are identical, so downstream code
    and the written sample file look the same in both modes.
    """
    out = []
    for q in questions:
        rec = dict(q)
        rec.pop("_paragraphs", None)
        qid = str(q.get("id"))
        pool = {(_canonical({"title": p["title"], "text": p["text"]})): p["id"]
                for p in pools.get(qid, [])}
        paras = _paragraphs_for(q)
        ids, missing = [], 0
        for pos in rec.pop("_supporting_local", []) or []:
            if not isinstance(pos, int) or not (0 <= pos < len(paras)):
                missing += 1
                continue
            pid = pool.get(_canonical(paras[pos]))
            if pid is None:
                missing += 1
            else:
                ids.append(pid)
        rec["supporting_ids"] = ids
        rec["supporting_unresolved"] = missing
        rec["pool_mode"] = POOL_MODE_PER_QUESTION
        out.append(rec)
    return out


def make_pool_retrieve_fn(bm25_index, pool: Sequence[dict[str, Any]],
                          top_k: int = 20):
    """Cumulative top-n ``retrieve_fn`` over the shared pool.

    Same shape as ``docbench.make_retrieve_fn`` (so the sufficiency loop consumes
    it unchanged), but each chunk carries its stable pool ``id`` so a retrieval
    can be scored against ``supporting_ids``.
    """
    from .docbench import bm25_retrieve

    documents = [{"id": p["id"], "title": p["title"], "text": p["text"]}
                 for p in pool]

    def retrieve_fn(question: str, n_docs: int) -> list[dict[str, Any]]:
        return bm25_retrieve(question, bm25_index, documents,
                             top_k=n_docs)["chunks"]

    retrieve_fn.pool_size = len(pool)
    retrieve_fn.top_k = top_k
    return retrieve_fn


# PLACEHOLDER_SANITY
    report = sorted((dict(r) for r in sample
                     if str(r.get("id", "")) not in cal_ids),
                    key=lambda r: str(r.get("id", "")))

    for r in calibrate:
        r["split"] = SPLIT_CALIBRATE
    for r in report:
        r["split"] = SPLIT_REPORT

    header = {
        "split_seed": seed + 1,
        "n_calibrate": len(calibrate),
        "n_report": len(report),
        "requested_calibrate": n_calibrate,
        "requested_report": n_report,
        "disjoint": not (cal_ids & {str(r.get("id", "")) for r in report}),
    }
    return calibrate, report, header


#: Keys that mark a record as still carrying private, pre-resolution state.
#: A row carrying any of these has NOT been through resolve_supporting_ids.
PRE_RESOLUTION_KEYS = ("_paragraphs", "_supporting_local")


class UnresolvedRowsError(ValueError):
    """Raised when records handed to the writer are still pre-resolution.

    The pre-resolution-rows defect (rows whose `supporting_ids` is empty because
    the writer was fed the raw sample instead of the resolved records) bit this
    project three separate times: Notebook 0 shipped a 500-row sample with empty
    gold ids, and the B6 and B7 build scripts both tripped the validation it
    exists to catch. Three occurrences is a structural problem, not a slip, so
    the writer now refuses such rows with its own exception type rather than
    reporting them as a generic dangling/empty-id error.

    Kept as a distinct subclass so a caller can catch "you forgot to resolve"
    specifically, and re-raise with the caller's dataset name attached.
    """


def assert_resolved(rows: Sequence[dict[str, Any]], who: str = "rows") -> None:
    """Refuse rows that still carry private pre-resolution keys, or empty gold.

    Called by every write path. `resolved=True` is not an opt-out: a row that
    still carries `_paragraphs`/`_supporting_local` is pre-resolution BY
    DEFINITION, whatever the caller believes, because
    :func:`resolve_supporting_ids` and its per-question twin both pop those keys
    before returning. So there is no flag to get wrong.
    """
    for i, r in enumerate(rows):
        present = [k for k in PRE_RESOLUTION_KEYS if k in r]
        if present:
            raise UnresolvedRowsError(
                f"{who}[{i}] ({r.get('id')!r}) still carries pre-resolution "
                f"key(s) {present}. Pass the output of resolve_supporting_ids() "
                f"/ resolve_supporting_ids_per_question(), not the raw sample. "
                f"(This is the defect that bit Notebook 0, B6 and B7.)")
        if not (r.get("supporting_ids") or []):
            raise UnresolvedRowsError(
                f"{who}[{i}] ({r.get('id')!r}) has empty supporting_ids. That is "
                f"the signature of pre-resolution rows: gold ids are filled in "
                f"by resolve_supporting_ids(), so an empty list means resolve "
                f"never ran on this row.")


def _finalize_suite_rows(rows: Sequence[dict[str, Any]],
                         header: dict[str, Any],
                         pool: Sequence[dict[str, Any]] | None,
                         who: str) -> list[dict[str, Any]]:
    """Shared gate for both write paths: assert resolved, then validate."""
    assert_resolved(rows, who)
    validate_suite_records(rows, header, pool)
    return list(rows)



# PLACEHOLDER_POOL


def validate_suite_records(rows: Sequence[dict[str, Any]],
                           header: dict[str, Any],
                           pool: Sequence[dict[str, Any]] | None = None
                           ) -> dict[str, Any]:
    """Hard validation of suite records against SUITE_RECORDS_CONTRACT.md.

    Every failure below is a BUILD ERROR, raised as ValueError, never a warning.
    The contract is the interface the suite runner codes to; a file that is
    subtly wrong will not fail loudly there — it will produce a run whose rows
    silently reference evidence that does not exist. So the check happens here,
    at the only point where the whole file is visible.

    Checks: no private keys; `supporting_ids` non-empty on every row; no dangling
    ids; id scheme matches the pool mode (contiguous p0..pN per question);
    `split` is calibrate/report and the halves are disjoint; the header carries
    every required field; shared mode leaves `paragraphs` empty on every row and
    per_question mode supplies them inline.
    """
    pool_mode = header.get("pool_mode")
    if pool_mode not in (POOL_MODE_SHARED, POOL_MODE_PER_QUESTION):
        raise ValueError(
            f"header pool_mode must be 'shared' or 'per_question', got "
            f"{pool_mode!r}")

    for key in ("dataset", "pool_mode", "source", "seed", "split_seed", "n",
                "n_calibrate", "n_report", "sample_rule", "strata_counts",
                "strata_taken", "metric_definitions"):
        if key not in header:
            raise ValueError(f"header is missing required key {key!r}")

    src = header.get("source") or {}
    for key in ("repo", "file", "bytes", "sha256", "license"):
        if key not in src:
            raise ValueError(f"header source is missing required key {key!r}")

    if not rows:
        raise ValueError("no question rows")

    shared_pool_ids = {p["id"] for p in (pool or [])}
    cal_ids: set[str] = set()
    rep_ids: set[str] = set()
    n_cal = n_rep = 0

    for i, r in enumerate(rows):
        rid = r.get("id")
        where = f"row {i} ({rid!r})"

        for key in ("id", "question", "answer", "aliases", "type", "hops",
                    "split", "supporting_ids"):
            if key not in r:
                raise ValueError(f"{where}: missing required key {key!r}")

        for private in ("_paragraphs", "_supporting_local"):
            if private in r:
                raise ValueError(
                    f"{where}: private key {private!r} leaked into the suite "
                    f"records; the contract forbids it")

        golds = r.get("supporting_ids") or []
        if not golds:
            raise ValueError(
                f"{where}: empty supporting_ids. The contract says this is a "
                f"build error, not a warning — a question with no gold "
                f"evidence cannot be scored.")

        split = r.get("split")
        if split not in (SPLIT_CALIBRATE, SPLIT_REPORT):
            raise ValueError(
                f"{where}: split must be 'calibrate' or 'report', got {split!r}")

        if pool_mode == POOL_MODE_PER_QUESTION:
            paras = r.get("paragraphs") or []
            if not paras:
                raise ValueError(
                    f"{where}: per_question mode requires inline 'paragraphs'")
            expected = [f"p{j}" for j in range(len(paras))]
            if [p["id"] for p in paras] != expected:
                raise ValueError(
                    f"{where}: per_question paragraph ids must be contiguous "
                    f"p0..p{len(paras) - 1}, got "
                    f"{[p['id'] for p in paras][:5]}...")
            dangling = [g for g in golds if g not in set(expected)]
            if dangling:
                raise ValueError(
                    f"{where}: dangling supporting_ids {dangling} not in this "
                    f"question's own paragraphs {expected[:5]}...")
        else:
            if r.get("paragraphs"):
                raise ValueError(
                    f"{where}: shared mode must leave 'paragraphs' empty on "
                    f"every row; the pool lives in a separate pool_file")
            if not shared_pool_ids:
                raise ValueError(
                    "shared mode: the shared pool must be supplied for "
                    "dangling-id validation")
            dangling = [g for g in golds if g not in shared_pool_ids]
            if dangling:
                raise ValueError(
                    f"{where}: dangling supporting_ids {dangling} not in the "
                    f"shared pool")

        if split == SPLIT_CALIBRATE:
            n_cal += 1
            cal_ids.add(str(rid))
        else:
            n_rep += 1
            rep_ids.add(str(rid))

    overlap = cal_ids & rep_ids
    if overlap:
        raise ValueError(
            f"calibrate/report halves are not disjoint: {len(overlap)} shared "
            f"id(s), e.g. {sorted(overlap)[:3]}")

    if header.get("n_calibrate") != n_cal or header.get("n_report") != n_rep:
        raise ValueError(
            f"header split sizes ({header.get('n_calibrate')}/"
            f"{header.get('n_report')}) disagree with the rows ({n_cal}/"
            f"{n_rep})")
    if header.get("n") != len(rows):
        raise ValueError(
            f"header n={header.get('n')} disagrees with {len(rows)} rows")

    dup = len(rows) - len({str(r.get("id")) for r in rows})
    if dup:
        raise ValueError(f"{dup} duplicate question id(s) in the suite records")

    return {"rows": len(rows), "n_calibrate": n_cal, "n_report": n_rep,
            "pool_mode": pool_mode, "shared_pool_size": len(shared_pool_ids)}


def write_suite_records(path: str | Path,
                        rows: Sequence[dict[str, Any]],
                        header: dict[str, Any],
                        pool: Sequence[dict[str, Any]] | None = None
                        ) -> dict[str, Any]:
    """Validate, then write a suite-records file per the contract.

    Validation runs BEFORE anything touches disk, so a rejected build leaves no
    half-written artifact for the runner to pick up. In shared mode the pool is
    written alongside as `<dataset>_pool.jsonl` and the header gains `pool_file`.

    Pre-resolution rows are refused with :class:`UnresolvedRowsError` before the
    general validation runs, so the failure names the actual mistake instead of
    surfacing as a generic empty/dangling-id error.
    """
    who = f"{header.get('dataset', 'rows')!r} rows"
    assert_resolved(rows, who)
    summary = validate_suite_records(rows, header, pool)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if pool is not None:
        pool_path = p.with_name(f"{header['dataset']}_pool.jsonl")
        header = dict(header)
        header["pool_file"] = pool_path.name
        write_jsonl_with_header(
            pool_path, pool,
            {"dataset": header["dataset"], "pool_mode": POOL_MODE_SHARED,
             "n_paragraphs": len(pool)})
        write_jsonl_with_header(p, rows, header)
        summary["pool_file"] = pool_path.name
        summary["pool_bytes"] = pool_path.stat().st_size
        summary["pool_paragraphs"] = len(pool)
    else:
        write_jsonl_with_header(p, rows, header)

    summary["records_bytes"] = p.stat().st_size
    summary["records_path"] = p.name
    return summary
def build_suite_records(path: str | Path,
                        sample: Sequence[dict[str, Any]],
                        split_rows: Sequence[dict[str, Any]],
                        header: dict[str, Any],
                        pool_mode: str = POOL_MODE_SHARED,
                        pool: Sequence[dict[str, Any]] | None = None
                        ) -> dict[str, Any]:
    """Resolve + write in ONE call, so the caller cannot skip the resolve step.

    This is the structural close on the defect that bit this project three times
    (Notebook 0, B6, B7): every occurrence was the same shape — a build script
    that resolved its records correctly, then assembled the output rows from the
    raw `sample`/`cal+rep` list instead, so every row shipped with empty
    `supporting_ids`. Offering a path where the caller never touches the
    unresolved list removes the opportunity, and :func:`assert_resolved` now
    refuses the unresolved list if someone reaches for it anyway.

    `split_rows` must be the output of ``split_calibrate_report`` (it supplies
    the per-row `split` label). The gold ids come from resolving `sample`.
    """
    if pool_mode == POOL_MODE_PER_QUESTION:
        pools, _ = build_per_question_pools(sample)
        resolved = resolve_supporting_ids_per_question(sample, pools)
    else:
        if pool is None:
            pool, _ = build_pool(sample)
        resolved = resolve_supporting_ids(sample, pool)

    split_of = {str(r.get("id")): r.get("split") for r in split_rows}
    rows = []
    for r in resolved:
        rid = str(r.get("id"))
        if rid not in split_of:
            raise ValueError(
                f"id {rid!r} has no split label; pass the output of "
                f"split_calibrate_report() as split_rows")
        rows.append({
            "id": r["id"], "question": r["question"], "answer": r["answer"],
            "aliases": r.get("aliases") or [], "type": r.get("type") or "",
            "hops": r.get("hops"), "split": split_of[rid],
            "supporting_ids": r.get("supporting_ids") or [],
            "supporting_unresolved": r.get("supporting_unresolved", 0),
            "paragraphs": (pools.get(r["id"], [])
                           if pool_mode == POOL_MODE_PER_QUESTION else []),
        })
    for extra in ("level", "frozen_split"):
        if any(extra in r for r in resolved):
            for row, r in zip(rows, resolved):
                row[extra] = r.get(extra, "")

    header = dict(header)
    header["pool_mode"] = pool_mode
    if pool_mode == POOL_MODE_SHARED:
        return write_suite_records(path, rows, header, pool=pool)
    return write_suite_records(path, rows, header)





def read_suite_records(path: str | Path
                       ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a suite-records file back. Round-trips with write_suite_records."""
    return read_jsonl_with_header(path)


def read_pool_file(path: str | Path) -> list[dict[str, Any]]:
    """Read a `<dataset>_pool.jsonl` written by :func:`write_suite_records`."""
    _, rows = read_jsonl_with_header(path)
    return rows
