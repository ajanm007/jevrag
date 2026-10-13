"""Offline tests for jevrag.benchmarks.closed_pool — tiny hand-built fixtures.

No network, no dataset downloads, no files over a few hundred bytes. The point
of these is the invariants the 500-question sample depends on: a sample you can
regenerate, a split that doesn't leak, pool ids that don't move between runs,
and gold evidence that survives deduplication.
"""

import json

import pytest

from jevrag.benchmarks.closed_pool import (
    DEFAULT_SEED,
    SPLIT_CALIBRATE,
    SPLIT_REPORT,
    build_index,
    build_pool,
    make_pool_retrieve_fn,
    make_question,
    pool_sanity_report,
    read_jsonl_with_header,
    resolve_supporting_ids,
    sample_questions,
    score_with_aliases,
    split_calibrate_report,
    write_jsonl_with_header,
)


def make_row(qid, hops=2, n_para=4, gold=(0,), answer="Paris"):
    """One question with its own paragraph pool; gold paragraphs at `gold`."""
    paras = []
    for i in range(n_para):
        paras.append({
            "idx": i,
            "title": f"doc{i}",
            "text": (f"Fact {i} about topic {qid}. " * 8),
            "is_supporting": i in gold,
        })
    return make_question(id=qid, question=f"What is fact for {qid}?",
                         answer=answer, aliases=[], type=f"{hops}hop",
                         hops=hops), paras


def rows(n=40, hops=2, **kw):
    """n questions at the given hop count, with UNIQUE ids across hop values.

    The hop count is in the id on purpose: two strata must never share ids, or
    a set-based disjointness check silently passes on a collision.
    """
    out = []
    for i in range(n):
        r, paras = make_row(f"{hops}hop__{i:04d}_x", hops=hops, **kw)
        r["_paragraphs"] = paras
        r["_supporting_local"] = [p["idx"] for p in paras if p["is_supporting"]]
        out.append(r)
    return out


# --- sampler determinism ---

def test_same_seed_same_sample():
    data = rows(40)
    a, _ = sample_questions(data, n=10, seed=7)
    b, _ = sample_questions(data, n=10, seed=7)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_different_seed_different_sample():
    data = rows(40)
    a, _ = sample_questions(data, n=10, seed=7)
    b, _ = sample_questions(data, n=10, seed=8)
    assert [r["id"] for r in a] != [r["id"] for r in b]


def test_sample_size_and_no_duplicates():
    data = rows(40)
    sample, header = sample_questions(data, n=12, seed=DEFAULT_SEED)
    assert len(sample) == 12
    assert len({r["id"] for r in sample}) == 12
    assert header["sampled_n"] == 12 and header["seed"] == DEFAULT_SEED


def test_sampler_refuses_more_than_available():
    with pytest.raises(ValueError, match="dataset has"):
        sample_questions(rows(5), n=10)


def test_sampler_records_its_rule_in_the_header():
    _, header = sample_questions(rows(40), n=10, seed=DEFAULT_SEED)
    assert "stratify_rule" in header and "seed" in header
    assert header["strata_counts"]


def test_stratification_follows_hops():
    data = rows(20, hops=2) + rows(20, hops=4)
    sample, header = sample_questions(data, n=20, seed=DEFAULT_SEED)
    got = header["strata_taken"]
    # 40 rows, 20 per hop bucket, n=20 -> 10 each, not 20 from one bucket.
    assert got["hops=2"] == 10 and got["hops=4"] == 10
    hops = [r["hops"] for r in sample]
    assert hops.count(2) == 10 and hops.count(4) == 10


# --- split: disjoint, sized, stratified ---

def test_split_is_250_250_by_default():
    data = rows(500)
    sample, _ = sample_questions(data, n=500, seed=DEFAULT_SEED)
    cal, rep, header = split_calibrate_report(sample)
    assert len(cal) == 250 and len(rep) == 250
    assert header["disjoint"] is True


def test_split_is_disjoint():
    data = rows(60)
    sample, _ = sample_questions(data, n=60, seed=DEFAULT_SEED)
    cal, rep, _ = split_calibrate_report(sample, n_calibrate=30, n_report=30)
    cal_ids = {r["id"] for r in cal}
    rep_ids = {r["id"] for r in rep}
    assert not (cal_ids & rep_ids)
    assert cal_ids | rep_ids == {r["id"] for r in sample}


def test_split_labels_are_set():
    data = rows(20)
    sample, _ = sample_questions(data, n=20, seed=DEFAULT_SEED)
    cal, rep, _ = split_calibrate_report(sample, n_calibrate=10, n_report=10)
    assert all(r["split"] == SPLIT_CALIBRATE for r in cal)
    assert all(r["split"] == SPLIT_REPORT for r in rep)


def test_split_is_deterministic():
    data = rows(40)
    sample, _ = sample_questions(data, n=40, seed=DEFAULT_SEED)
    a, _, _ = split_calibrate_report(sample, n_calibrate=20, n_report=20)
    b, _, _ = split_calibrate_report(sample, n_calibrate=20, n_report=20)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_split_is_stratified_across_hops():
    data = rows(30, hops=2) + rows(30, hops=4)
    sample, _ = sample_questions(data, n=60, seed=DEFAULT_SEED)
    cal, rep, _ = split_calibrate_report(sample, n_calibrate=30, n_report=30)
    cal_hops = sorted(r["hops"] for r in cal)
    rep_hops = sorted(r["hops"] for r in rep)
    assert cal_hops.count(2) == 15 and cal_hops.count(4) == 15
    assert rep_hops.count(2) == 15 and rep_hops.count(4) == 15


def test_split_refuses_to_exceed_sample():
    data = rows(10)
    sample, _ = sample_questions(data, n=10, seed=DEFAULT_SEED)
    with pytest.raises(ValueError, match="exceeds sample size"):
        split_calibrate_report(sample, n_calibrate=8, n_report=8)


def test_split_does_not_mutate_input():
    data = rows(20)
    sample, _ = sample_questions(data, n=20, seed=DEFAULT_SEED)
    before = [dict(r) for r in sample]
    split_calibrate_report(sample, n_calibrate=10, n_report=10)
    assert sample == before


# --- pool: dedup, stable ids, gold resolution ---

def test_pool_deduplicates_identical_paragraphs():
    data = rows(3, n_para=2)
    # Make every question share the exact same two paragraphs.
    shared = data[0]["_paragraphs"]
    for r in data:
        r["_paragraphs"] = [dict(p) for p in shared]
    pool, header = build_pool(data)
    assert len(pool) == 2
    assert header["paragraphs_seen"] == 6
    assert header["duplicates_removed"] == 4


def test_pool_dedup_ignores_whitespace_differences():
    r, paras = make_row("2hop__a_x", n_para=2)
    r["_paragraphs"] = paras
    dup = [dict(paras[0]), dict(paras[0])]
    dup[1]["text"] = "  " + paras[0]["text"].replace(" ", "  ") + "\n"
    r["_paragraphs"] = [paras[0], dup[0], dup[1]]
    pool, header = build_pool([r])
    assert len(pool) == 1, "whitespace-only difference must not survive dedup"
    assert header["duplicates_removed"] == 2


def test_pool_ids_are_stable_regardless_of_input_order():
    data = rows(6, n_para=3)
    a, _ = build_pool(data)
    b, _ = build_pool(list(reversed(data)))
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert [p["text"] for p in a] == [p["text"] for p in b]


def test_pool_ids_are_stable_across_runs():
    data = rows(6, n_para=3)
    first, _ = build_pool(data)
    second, _ = build_pool(data)
    assert [p["id"] for p in first] == [p["id"] for p in second]


def test_pool_entries_carry_id_title_text():
    data = rows(2, n_para=2)
    pool, _ = build_pool(data)
    assert set(pool[0]) == {"id", "title", "text"}
    assert pool[0]["id"] == "p0"


def test_empty_paragraphs_are_dropped_not_indexed():
    r, paras = make_row("2hop__e_x", n_para=3)
    paras[1]["text"] = "   "
    r["_paragraphs"] = paras
    pool, header = build_pool([r])
    assert len(pool) == 2
    assert all(p["text"].strip() for p in pool)

# --- alias-aware EM/F1 through the vendored evaluator ---

def test_em_matches_the_canonical_answer():
    rec = make_question(id="q", question="Q?", answer="Paris")
    assert score_with_aliases("paris", rec)["em"] is True


def test_em_matches_an_alias():
    rec = make_question(id="q", question="Q?", answer="Paris",
                        aliases=["City of Light", "Paris, France"])
    out = score_with_aliases("City of Light", rec)
    assert out["em"] is True
    assert out["matched_alias"] == "City of Light"


def test_em_rejects_a_non_answer():
    rec = make_question(id="q", question="Q?", answer="Paris",
                        aliases=["City of Light"])
    assert score_with_aliases("Berlin", rec)["em"] is False


def test_empty_prediction_is_not_em():
    rec = make_question(id="q", question="Q?", answer="Paris")
    assert score_with_aliases("", rec)["em"] is False


def test_f1_takes_the_best_alias_not_the_first():
    # Best-alias selection: "Paris" must outscore a prediction that only
    # partially matches the primary gold. Normalization strips punctuation, so
    # the alias must differ in CONTENT, not formatting.
    rec = make_question(id="q", question="Q?", answer="Paris",
                        aliases=["New York City"])
    best = score_with_aliases("Paris", rec)["f1"]
    partial = score_with_aliases("New York", rec)["f1"]
    assert best == 1.0
    assert partial < best


def test_scoring_raises_on_a_record_with_no_gold():
    rec = make_question(id="q", question="Q?", answer="")
    with pytest.raises(ValueError, match="no gold answer"):
        score_with_aliases("anything", rec)


# --- BM25 retrieval over the shared pool ---

def test_retrieve_fn_returns_pool_ids():
    data = rows(5, n_para=3)
    pool, _ = build_pool(data)
    retrieve = make_pool_retrieve_fn(build_index(pool), pool, top_k=5)
    hits = retrieve("Fact 0 about topic 2hop__0002_x.", 3)
    assert hits and all("id" in h for h in hits)
    assert all(h["id"] in {p["id"] for p in pool} for h in hits)


def test_bm25_finds_the_supporting_paragraph_for_an_easy_fixture():
    """The signal the pool-sanity report depends on: a distinctive question
    retrieves the paragraph that actually contains its supporting fact."""
    rec, paras = make_row("2hop__marker_x", n_para=6, gold=(4,))
    rec["_paragraphs"] = paras
    rec["_supporting_local"] = [4]
    paras[4]["text"] = ("The Zylophone Committee met in Utrecht to discuss "
                        "the quicksilver protocol. " * 6)
    for i in (0, 1, 2, 3, 5):
        paras[i]["text"] = f"Unrelated filler paragraph number {i}. " * 6

    pool, _ = build_pool([rec])
    resolved = resolve_supporting_ids([rec], pool)
    assert resolved[0]["supporting_ids"], "gold must be in the pool"

    retrieve = make_pool_retrieve_fn(build_index(pool), pool, top_k=10)
    got = [c["id"] for c in retrieve("Zylophone Committee quicksilver protocol", 10)]
    assert resolved[0]["supporting_ids"][0] in got


# --- the pool-sanity report ---

def test_sanity_reports_gold_in_pool_and_recall():
    data = rows(10, n_para=4, gold=(0,))
    pool, _ = build_pool(data)
    resolved = resolve_supporting_ids(data, pool)
    retrieve = make_pool_retrieve_fn(build_index(pool), pool, top_k=20)
    rep = pool_sanity_report(resolved, pool, retrieve, ks=(5, 10, 15, 20))

    assert rep["n_questions"] == 10
    assert rep["n_questions_with_gold"] == 10
    assert rep["gold_in_pool_rate"] == 1.0
    for k in ("5", "10", "15", "20"):
        assert 0.0 <= rep["paragraph_recall_at_k"][k] <= 1.0
        assert 0.0 <= rep["all_gold_found_at_k"][k] <= 1.0
    assert set(rep["paragraph_recall_at_k"]) == {"5", "10", "15", "20"}


def test_sanity_excludes_questions_without_gold_instead_of_scoring_zero():
    data = rows(4, n_para=3, gold=(0,))
    for r in data[2:]:
        r["_supporting_local"] = []
    pool, _ = build_pool(data)
    resolved = resolve_supporting_ids(data, pool)
    retrieve = make_pool_retrieve_fn(build_index(pool), pool, top_k=10)
    rep = pool_sanity_report(resolved, pool, retrieve)

    assert rep["n_questions_with_gold"] == 2
    assert rep["n_questions_without_gold"] == 2
    assert rep["gold_paragraphs_total"] == 2


def test_sanity_flags_missing_gold():
    r, paras = make_row("2hop__m_x", n_para=3, gold=(0,))
    paras[1]["text"] = ""
    r["_paragraphs"] = paras
    r["_supporting_local"] = [1]
    pool, _ = build_pool([r])
    resolved = resolve_supporting_ids([r], pool)
    retrieve = make_pool_retrieve_fn(build_index(pool), pool, top_k=10)
    rep = pool_sanity_report(resolved, pool, retrieve)
    assert rep["n_questions_with_missing_gold"] == 1
    assert rep["questions_with_missing_gold"][0]["id"] == "2hop__m_x"


def test_sanity_breakdown_by_hop_count():
    data = rows(6, n_para=3, hops=2, gold=(0,)) + \
        rows(6, n_para=3, hops=4, gold=(0,))
    pool, _ = build_pool(data)
    resolved = resolve_supporting_ids(data, pool)
    retrieve = make_pool_retrieve_fn(build_index(pool), pool, top_k=10)
    rep = pool_sanity_report(resolved, pool, retrieve)
    assert set(rep["per_hop"]) == {"2", "4"}
    assert rep["per_hop"]["2"]["questions"] == 6


# --- header round-trip ---

def test_jsonl_header_round_trip(tmp_path):
    p = tmp_path / "out.jsonl"
    header = {"seed": 42, "n": 2}
    rows_out = [{"id": "a"}, {"id": "b"}]
    write_jsonl_with_header(p, rows_out, header)
    got_header, got_rows = read_jsonl_with_header(p)
    assert got_header == header
    assert got_rows == rows_out


def test_jsonl_first_line_is_the_header(tmp_path):
    p = tmp_path / "out.jsonl"
    write_jsonl_with_header(p, [{"id": "a"}], {"seed": 1})
    first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert first["__header__"]["seed"] == 1

def test_supporting_ids_resolve_into_pool_space():
    data = rows(4, n_para=3, gold=(1,))
    pool, _ = build_pool(data)
    resolved = resolve_supporting_ids(data, pool)
    for rec in resolved:
        assert rec["supporting_ids"], "every gold paragraph must resolve"
        assert all(pid in {p["id"] for p in pool} for pid in rec["supporting_ids"])
        assert rec["supporting_unresolved"] == 0


def test_resolve_strips_private_keys():
    data = rows(3, n_para=2)
    pool, _ = build_pool(data)
    resolved = resolve_supporting_ids(data, pool)
    assert "_paragraphs" not in resolved[0]
    assert "_supporting_local" not in resolved[0]


def test_unresolvable_gold_is_counted_not_hidden():
    r, paras = make_row("2hop__x_x", n_para=2, gold=(0,))
    paras[1]["text"] = ""     # dropped from the pool by build_pool
    r["_paragraphs"] = paras
    r["_supporting_local"] = [1]
    pool, _ = build_pool([r])
    resolved = resolve_supporting_ids([r], pool)
    assert resolved[0]["supporting_unresolved"] == 1


def test_rare_stratum_is_not_rounded_away_again():
    # 1 four-hop row among 39 two-hop rows: largest-remainder must keep it.
    data = rows(39, hops=2)
    rare, paras = make_row("4hop__0001_x", hops=4, n_para=3)
    rare["_paragraphs"] = paras
    rare["_supporting_local"] = [0]
    data.append(rare)
    _, header = sample_questions(data, n=40, seed=DEFAULT_SEED)
    assert header["strata_taken"]["hops=4"] == 1


def test_sampler_input_not_mutated():
    data = rows(20)
    before = [dict(r) for r in data]
    sample_questions(data, n=5, seed=1)
    assert data == before
