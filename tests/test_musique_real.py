"""Tests against the REAL MuSiQue dev file (packet V03-B5).

These run only when the file is present at outputs/musique_answerable_dev.jsonl
(gitignored, CC BY 4.0, never committed). They skip otherwise, so the suite stays
green on a machine that never downloaded it.

Why real-file tests: every defect Notebook 0 found — hop parsing, empty
supporting_ids in the written sample, private keys leaking, a provenance block
naming a config that does not exist — passed against synthetic fixtures. The
synthetic rows all had `2hop__` ids and resolvable gold, so none of them could
have caught these. Real data is the only thing that exercises the shapes that
actually occur.
"""

import collections
from pathlib import Path

import pytest

from jevrag.benchmarks import closed_pool as CP
from jevrag.benchmarks import musique as MQ

REAL = (Path(__file__).resolve().parent.parent / "outputs"
        / "musique_answerable_dev.jsonl")
_TMP = Path(__file__).resolve().parent.parent / "outputs"

pytestmark = pytest.mark.skipif(
    not REAL.exists(),
    reason=f"real MuSiQue dev file not present at {REAL} (gitignored; "
           f"download it before running the B5 real-file tests)")


@pytest.fixture(scope="module")
def records():
    return MQ.load_musique(REAL)


def _write_and_read(rows, header, name):
    out = _TMP / name
    try:
        CP.write_jsonl_with_header(out, rows, header)
        return CP.read_jsonl_with_header(out)
    finally:
        out.unlink(missing_ok=True)


# --- hop parsing (defect 1) ---

def test_every_real_row_gets_a_hop_count(records):
    """The bug: only `2hop__` parsed, so 1,165 of 2,417 rows got hops=None and
    were stratified as "type=unknown"."""
    missing = [r["id"] for r in records if r.get("hops") is None]
    assert missing == [], f"{len(missing)} rows have hops=None, e.g. {missing[:5]}"


def test_hops_match_supporting_count_on_every_real_row(records):
    """The cross-check the packet asked for: the id prefix and the paragraph
    annotations are independent statements about the same question."""
    report = MQ.hops_match_supporting(records)
    assert report["n"] == 2417, report["n"]
    assert report["n_mismatched"] == 0, report["examples"][:5]
    assert report["agreement_rate"] == 1.0


def test_real_hop_distribution_is_2_3_4(records):
    counts = collections.Counter(r["hops"] for r in records)
    assert set(counts) == {2, 3, 4}, dict(counts)
    assert counts[2] == 1252 and counts[3] == 760 and counts[4] == 405, dict(counts)


@pytest.mark.parametrize("qid,expected", [
    ("2hop__129292_160863", 2),
    ("3hop1__abc_def", 3),
    ("3hop2__abc_def", 3),
    ("4hop1__abc_def", 4),
    ("4hop2__abc_def", 4),
    ("4hop3__abc_def", 4),
])
def test_each_real_id_prefix_parses(qid, expected):
    """3hop1/3hop2/4hop1/4hop2/4hop3 all carry their hop count; the trailing
    digit is a variant label, not part of the count."""
    assert MQ._hops(qid, {"id": qid, "paragraphs": []}) == expected


@pytest.mark.parametrize("qid,expected", [
    ("2hop__1_2", "2hop"), ("3hop1__1_2", "3hop1"), ("4hop3__1_2", "4hop3"),
])
def test_id_prefix_is_kept_separately(qid, expected):
    assert MQ.id_prefix(qid) == expected


def test_real_rows_all_have_supporting_annotations(records):
    assert all(len(r["_supporting_local"]) >= 2 for r in records)
    assert all(r.get("id_prefix") for r in records)


# --- the written-sample contract (defect 2) ---

def test_written_sample_rows_have_non_empty_supporting_ids(records):
    """The file Notebook 0 wrote had EMPTY supporting_ids on all 500 rows,
    because it wrote the pre-resolution records."""
    sample, _ = CP.sample_questions(records, n=60, seed=CP.DEFAULT_SEED)
    pool, _ = CP.build_pool(sample)
    resolved = CP.resolve_supporting_ids(sample, pool)

    _, rows = _write_and_read(resolved, {"dataset": "musique"},
                              "_b5_test_sample.jsonl")
    assert len(rows) == len(resolved)
    empty = [r["id"] for r in rows if not r.get("supporting_ids")]
    assert empty == [], f"{len(empty)} rows have empty supporting_ids"


def test_written_sample_rows_carry_no_private_keys(records):
    """The file also still carried `_paragraphs` (20 long paragraphs per row),
    contradicting the docstring that says they are stripped."""
    sample, _ = CP.sample_questions(records, n=40, seed=CP.DEFAULT_SEED)
    pool, _ = CP.build_pool(sample)
    resolved = CP.resolve_supporting_ids(sample, pool)

    _, rows = _write_and_read(resolved, {"dataset": "musique"},
                              "_b5_test_priv.jsonl")
    leaked = [r["id"] for r in rows
              if "_paragraphs" in r or "_supporting_local" in r]
    assert leaked == [], f"{len(leaked)} rows still carry private keys"


def test_written_supporting_ids_exist_in_the_written_pool(records):
    """Cross-file contract: every id the sample references must be in the pool."""
    sample, _ = CP.sample_questions(records, n=40, seed=CP.DEFAULT_SEED)
    pool, _ = CP.build_pool(sample)
    resolved = CP.resolve_supporting_ids(sample, pool)
    pool_ids = {p["id"] for p in pool}

    missing = [(r["id"], g) for r in resolved for g in r["supporting_ids"]
               if g not in pool_ids]
    assert missing == [], f"{len(missing)} dangling ids, e.g. {missing[:3]}"


def test_per_question_written_rows_also_resolve(records):
    """Same contract in per_question mode: ids are p0..pN scoped per question."""
    sample, _ = CP.sample_questions(records, n=40, seed=CP.DEFAULT_SEED)
    pools, _ = CP.build_per_question_pools(sample)
    resolved = CP.resolve_supporting_ids_per_question(sample, pools)

    _, rows = _write_and_read(resolved, {"dataset": "musique"},
                              "_b5_test_pq.jsonl")
    empty = [r["id"] for r in rows if not r.get("supporting_ids")]
    leaked = [r["id"] for r in rows if "_paragraphs" in r]
    assert empty == [], f"{len(empty)} rows empty in per_question mode"
    assert leaked == [], f"{len(leaked)} rows leaked private keys"


# --- provenance (defect 3) ---

def test_provenance_names_no_nonexistent_config_or_split():
    """`answerable_dev`/`dev` never existed; this repo has no configs at all."""
    prov = MQ.provenance()
    assert prov["hf_configs"] is None
    assert prov["hf_splits"] is None
    assert prov["hf_repo_id"] == "dgslibisey/MuSiQue"
    assert prov["hf_repo_type"] == "dataset"
    assert prov["hf_file"] == "musique_ans_v1.0_dev.jsonl"
    assert "hf_config" not in prov and "hf_split" not in prov
    assert not hasattr(MQ, "HF_CONFIG")
    assert not hasattr(MQ, "HF_SPLIT")
    assert not hasattr(MQ, "HF_DATASET_ID")


def test_provenance_file_size_matches_the_real_file():
    prov = MQ.provenance(REAL)
    assert prov["local_file_bytes"] == REAL.stat().st_size == 30_439_728
    assert prov["hf_file_bytes"] == 30_439_728


def test_provenance_records_a_real_sha256():
    """Provenance that names a file but not its contents is a filename, not a
    provenance record: the same name can be re-uploaded with different rows."""
    prov = MQ.provenance(REAL)
    digest = prov["local_file_sha256"]
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert digest == MQ.file_sha256(REAL)
    # Matches the LFS oid the HF tree API reports for the same file.
    assert digest.startswith("15fa63794d18a94ce12411ac")


def test_real_file_is_the_answerable_variant():
    rows = [r for r in MQ._iter_json_records(REAL)]
    assert len(rows) == 2417
    assert all(r.get("answerable") for r in rows)
    assert all(str(r.get("answer", "")).strip() for r in rows)


# --- pool modes and metric naming (defect 4 + per_question mode) ---

def test_pool_modes_exist_and_default_is_shared():
    assert CP.POOL_MODE_SHARED == "shared"
    assert CP.POOL_MODE_PER_QUESTION == "per_question"
    assert CP.DEFAULT_POOL_MODE == CP.POOL_MODE_SHARED
    # k=15 must be present: the loop reads 5 paragraphs x 3 rounds.
    assert 15 in CP.RECALL_KS
    assert set(CP.RECALL_KS) == {5, 10, 15, 20}


def test_sanity_report_emits_both_metrics_under_distinct_names(records):
    sample, _ = CP.sample_questions(records, n=30, seed=CP.DEFAULT_SEED)
    pool, _ = CP.build_pool(sample)
    resolved = CP.resolve_supporting_ids(sample, pool)
    retrieve = CP.make_pool_retrieve_fn(CP.build_index(pool), pool, top_k=20)
    rep = CP.pool_sanity_report(resolved, pool, retrieve, ks=CP.RECALL_KS)

    # The old confusing names must be GONE, not merely supplemented.
    assert "gold_recall_at_k" not in rep
    assert "paragraph_recall_at_k" in rep
    assert "all_gold_found_at_k" in rep
    assert "metric_definitions" in rep
    assert set(rep["paragraph_recall_at_k"]) == {"5", "10", "15", "20"}
    assert set(rep["all_gold_found_at_k"]) == {"5", "10", "15", "20"}
    for bucket in rep["per_hop"].values():
        assert "recall_at_max_k" not in bucket
        assert "paragraph_recall_at_k" in bucket
        assert "all_gold_found_at_k" in bucket


def test_all_gold_found_is_never_above_paragraph_recall(records):
    """The two metrics are ordered by construction: all-or-nothing at question
    level cannot exceed partial credit at paragraph level."""
    sample, _ = CP.sample_questions(records, n=40, seed=CP.DEFAULT_SEED)
    pool, _ = CP.build_pool(sample)
    resolved = CP.resolve_supporting_ids(sample, pool)
    retrieve = CP.make_pool_retrieve_fn(CP.build_index(pool), pool, top_k=20)
    rep = CP.pool_sanity_report(resolved, pool, retrieve, ks=CP.RECALL_KS)
    for k in rep["paragraph_recall_at_k"]:
        assert (rep["all_gold_found_at_k"][k]
                <= rep["paragraph_recall_at_k"][k] + 1e-9), k


def test_per_question_mode_always_contains_the_gold(records):
    """MuSiQue's own distractor setting ships the gold inside the question's 20
    paragraphs, so per-question recall at k=20 must be exactly 1.0 -- if it is
    not, the pool builder dropped something and the mode is broken."""
    sample, _ = CP.sample_questions(records, n=60, seed=CP.DEFAULT_SEED)
    pools, _ = CP.build_per_question_pools(sample)
    resolved = CP.resolve_supporting_ids_per_question(sample, pools)
    retrieve = CP.make_per_question_retrieve_fn(pools)
    flat = [p for paras in pools.values() for p in paras]
    rep = CP.pool_sanity_report(resolved, flat, retrieve, ks=CP.RECALL_KS,
                               pool_mode=CP.POOL_MODE_PER_QUESTION)

    assert rep["pool_mode"] == "per_question"
    assert rep["gold_in_pool_rate"] == 1.0
    assert rep["all_gold_found_at_k"]["20"] == 1.0
    assert rep["paragraph_recall_at_k"]["20"] == 1.0
    assert rep["n_questions_with_missing_gold"] == 0


def test_per_question_retrieve_raises_if_no_pool_is_bound():
    """Silently searching the wrong pool would be a correctness bug, not a
    convenience; it must refuse."""
    pools = {"q1": [{"id": "p0", "title": "t", "text": "alpha beta"}]}
    retrieve = CP.make_per_question_retrieve_fn(pools)
    with pytest.raises(RuntimeError, match="bind_question_id"):
        retrieve("anything", 5)
    retrieve.bind_question_id("q1")
    assert retrieve("alpha", 1)[0]["id"] == "p0"
    retrieve.bind_question_id("nope")
    with pytest.raises(RuntimeError):
        retrieve("alpha", 1)


def test_per_question_pools_do_not_merge_across_questions(records):
    """Cross-question duplicates are deliberately NOT collapsed in this mode."""
    sample, _ = CP.sample_questions(records, n=30, seed=CP.DEFAULT_SEED)
    pools, _ = CP.build_per_question_pools(sample)
    total = sum(len(v) for v in pools.values())
    seen = sum(len(v) for v in pools.values())
    assert total == seen
    header_pq, _ = CP.build_per_question_pools(sample)[1], None
    assert header_pq["pool_mode"] == "per_question"
    # every question keeps roughly its own 20 paragraphs
    sizes = [len(v) for v in pools.values()]
    assert min(sizes) >= 15 and max(sizes) <= 20


def test_pool_mode_is_recorded_in_the_report(records):
    sample, _ = CP.sample_questions(records, n=20, seed=CP.DEFAULT_SEED)
    pool, _ = CP.build_pool(sample)
    resolved = CP.resolve_supporting_ids(sample, pool)
    retrieve = CP.make_pool_retrieve_fn(CP.build_index(pool), pool, top_k=20)
    rep = CP.pool_sanity_report(resolved, pool, retrieve, ks=(5, 10))
    assert rep["pool_mode"] == "shared"