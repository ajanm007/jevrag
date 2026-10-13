"""Tests for the suite-records writer (B6 item 1) and the 2Wiki adapter.

The writer's job is to REFUSE bad records before they reach the suite runner, so
most of these tests assert that a specific defect raises rather than that a good
file passes. A validation that has never rejected anything is decoration.
"""

import json

import pytest

from jevrag.benchmarks import closed_pool as CP
from jevrag.benchmarks import twowiki as TW
from jevrag.benchmarks.closed_pool import make_question
from jevrag.benchmarks.closed_pool import make_question


def good_header(pool_mode=CP.POOL_MODE_PER_QUESTION, n=4, n_cal=2, n_rep=2):
    return {
        "dataset": "toy",
        "pool_mode": pool_mode,
        "source": {"repo": "r/x", "file": "f.jsonl", "bytes": 10,
                   "sha256": "0" * 64, "license": "CC-BY-4.0"},
        "seed": 1, "split_seed": 2, "n": n,
        "n_calibrate": n_cal, "n_report": n_rep,
        "sample_rule": "toy", "strata_counts": {}, "strata_taken": {},
        "metric_definitions": {"paragraph_recall_at_k": "p",
                               "all_gold_found_at_k": "a"},
    }


def good_row(i, pool_mode=CP.POOL_MODE_PER_QUESTION, split=None,
             resolved=True):
    """A RESOLVED row, as a real build produces after resolve_supporting_ids.

    `resolved=False` builds the pre-resolution shape on purpose: raw sample
    rows carry the private keys and an empty supporting_ids list, which the
    writer now refuses.
    """
    paras = [{"id": f"p{j}", "title": f"t{j}", "text": f"body {j}"}
             for j in range(3)]
    row = {
        "id": f"q{i}", "question": f"Q{i}?", "answer": "A", "aliases": [],
        "type": "2hop", "hops": 2,
        "split": split or (CP.SPLIT_CALIBRATE if i % 2 == 0
                           else CP.SPLIT_REPORT),
        "supporting_ids": ["p0"] if pool_mode == CP.POOL_MODE_PER_QUESTION
                          else [f"p{i}"],
        "paragraphs": paras if pool_mode == CP.POOL_MODE_PER_QUESTION else [],
    }
    if not resolved:
        row["supporting_ids"] = []
        row["_paragraphs"] = [{"title": f"t{j}", "text": f"body {j}"}
                              for j in range(3)]
        row["_supporting_local"] = [0]
    return row


SHARED_POOL = [{"id": f"p{i}", "title": f"t{i}", "text": f"body {i}"}
               for i in range(4)]


# --- the happy path ---

def test_accepts_a_good_per_question_file():
    rows = [good_row(i) for i in range(4)]
    out = CP.validate_suite_records(rows, good_header(), None)
    assert out["rows"] == 4 and out["pool_mode"] == "per_question"


def test_accepts_a_good_shared_file():
    rows = [good_row(i, CP.POOL_MODE_SHARED) for i in range(4)]
    out = CP.validate_suite_records(rows, good_header(CP.POOL_MODE_SHARED),
                                    SHARED_POOL)
    assert out["shared_pool_size"] == 4


def test_round_trip_write_then_read(tmp_path):
    rows = [good_row(i) for i in range(4)]
    p = tmp_path / "toy_suite_records.jsonl"
    summary = CP.write_suite_records(p, rows, good_header())
    back_header, back_rows = CP.read_suite_records(p)
    assert summary["records_bytes"] > 0
    assert len(back_rows) == 4
    assert back_header["pool_mode"] == "per_question"
    assert back_rows[0]["supporting_ids"] == ["p0"]
    # re-validating what came off disk must also pass
    CP.validate_suite_records(back_rows, back_header)


def test_shared_mode_writes_a_pool_file_and_names_it_in_the_header(tmp_path):
    rows = [good_row(i, CP.POOL_MODE_SHARED) for i in range(4)]
    p = tmp_path / "toy_suite_records.jsonl"
    summary = CP.write_suite_records(p, rows,
                                     good_header(CP.POOL_MODE_SHARED),
                                     pool=SHARED_POOL)
    assert summary["pool_file"] == "toy_pool.jsonl"
    header, _ = CP.read_suite_records(p)
    assert header["pool_file"] == "toy_pool.jsonl"
    assert len(CP.read_pool_file(tmp_path / "toy_pool.jsonl")) == 4


# --- the rejections: one per contract failure ---

def test_rejects_empty_supporting_ids():
    """The exact defect Notebook 0 shipped, on all 500 rows."""
    rows = [good_row(i) for i in range(4)]
    rows[2]["supporting_ids"] = []
    with pytest.raises(ValueError, match="empty supporting_ids"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_dangling_id_per_question():
    rows = [good_row(i) for i in range(4)]
    rows[1]["supporting_ids"] = ["p99"]
    with pytest.raises(ValueError, match="dangling supporting_ids"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_dangling_id_shared():
    rows = [good_row(i, CP.POOL_MODE_SHARED) for i in range(4)]
    rows[0]["supporting_ids"] = ["p999"]
    with pytest.raises(ValueError, match="dangling supporting_ids"):
        CP.validate_suite_records(rows, good_header(CP.POOL_MODE_SHARED),
                                  SHARED_POOL)


def test_rejects_leaked_private_keys():
    rows = [good_row(i) for i in range(4)]
    rows[0]["_paragraphs"] = [{"title": "t", "text": "x"}]
    with pytest.raises(ValueError, match="private key"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_leaked_supporting_local():
    rows = [good_row(i) for i in range(4)]
    rows[1]["_supporting_local"] = [0]
    with pytest.raises(ValueError, match="private key"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_non_disjoint_splits():
    """A genuine overlap: the same id in both halves. Setting one row's split
    to 'calibrate' would only change the 2/2 counts and trip the count check
    instead, so this duplicates an id ACROSS the halves."""
    rows = [good_row(i) for i in range(4)]
    rows[3]["id"] = rows[0]["id"]        # q0 now appears in both halves
    rows[3]["split"] = CP.SPLIT_REPORT
    with pytest.raises(ValueError, match="disjoint"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_a_split_count_that_disagrees_with_the_rows():
    rows = [good_row(i) for i in range(4)]
    rows[1]["split"] = CP.SPLIT_CALIBRATE  # 3 calibrate / 1 report
    with pytest.raises(ValueError, match="disagree with the rows"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_bad_split_label():
    rows = [good_row(i) for i in range(4)]
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="split must be"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_non_contiguous_paragraph_ids():
    """A gap in p0..pN means supporting_ids point somewhere untrustworthy."""
    rows = [good_row(i) for i in range(4)]
    rows[0]["paragraphs"][1]["id"] = "p7"
    with pytest.raises(ValueError, match="contiguous"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_shared_mode_with_inline_paragraphs():
    rows = [good_row(i, CP.POOL_MODE_SHARED) for i in range(4)]
    rows[0]["paragraphs"] = [{"id": "p0", "title": "t", "text": "x"}]
    with pytest.raises(ValueError, match="shared mode must leave"):
        CP.validate_suite_records(rows, good_header(CP.POOL_MODE_SHARED),
                                  SHARED_POOL)


def test_rejects_per_question_mode_without_paragraphs():
    rows = [good_row(i) for i in range(4)]
    rows[3]["paragraphs"] = []
    with pytest.raises(ValueError, match="requires inline"):
        CP.validate_suite_records(rows, good_header())


def test_rejects_header_count_mismatch():
    rows = [good_row(i) for i in range(4)]
    with pytest.raises(ValueError, match="disagree with the rows"):
        CP.validate_suite_records(rows, good_header(n_cal=3, n_rep=3))


def test_rejects_missing_source_fields():
    rows = [good_row(i) for i in range(4)]
    header = good_header()
    del header["source"]["sha256"]
    with pytest.raises(ValueError, match="source is missing"):
        CP.validate_suite_records(rows, header)


# --- the 2Wiki adapter ---

def _tw_row(context, facts, qid="x1", qtype="comparison"):
    return {"_id": qid, "type": qtype, "question": "Q?",
            "answer": "A", "context": json.dumps(context),
            "supporting_facts": json.dumps(facts),
            "evidences": json.dumps([])}


CTX = [["Alpha", ["a1", "a2"]], ["Beta", ["b1"]], ["Gamma", ["g1"]]]


def test_twowiki_loads_a_row(tmp_path):
    """The real file stores context and supporting_facts as JSON STRINGS; an
    adapter that reads them as lists gets nothing."""
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(_tw_row(CTX, [["Alpha", 0]])) + "\n", encoding="utf-8")
    recs, health = TW.load_twowiki(p)
    assert len(recs) == 1
    r = recs[0]
    assert r["question"] == "Q?" and r["answer"] == "A"
    assert [x["title"] for x in r["_paragraphs"]] == ["Alpha", "Beta", "Gamma"]
    assert r["_paragraphs"][0]["text"] == "a1 a2"
    assert r["_supporting_titles"] == ["Alpha"]
    assert r["_supporting_local"] == [0]
    assert health["n_kept"] == 1 and health["n_unmapped_title"] == 0


def test_twowiki_dedupes_repeated_supporting_titles(tmp_path):
    """One paragraph can supply two of the supporting sentences."""
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(_tw_row(CTX, [["Alpha", 0], ["Alpha", 1]])) + "\n",
                 encoding="utf-8")
    recs, _ = TW.load_twowiki(p)
    assert recs[0]["_supporting_titles"] == ["Alpha"]
    assert recs[0]["_supporting_local"] == [0]


def test_twowiki_counts_a_supporting_title_missing_from_context(tmp_path):
    """A build error, reported rather than silently skipped."""
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(_tw_row(CTX, [["Alpha", 0], ["Zeta", 1]])) + "\n",
                 encoding="utf-8")
    _, health = TW.load_twowiki(p)
    assert health["n_unmapped_title"] == 1
    assert health["unmapped_examples"][0]["titles"] == ["Zeta"]


def test_twowiki_drops_rows_with_no_answer(tmp_path):
    p = tmp_path / "d.jsonl"
    row = _tw_row(CTX, [["Alpha", 0]])
    row["answer"] = ""
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    recs, health = TW.load_twowiki(p)
    assert recs == []
    assert health["n_dropped_no_answer"] == 1


def test_twowiki_raises_on_a_malformed_context(tmp_path):
    """A JSON string that parses to a non-list is caught with a clear message."""
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(_tw_row("not json", [["Alpha", 0]])) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="context should be a list"):
        TW.load_twowiki(p)


def test_twowiki_raises_on_an_unparseable_json_field(tmp_path):
    """context as a bare non-JSON string, which is what a mis-mapped column
    looks like in practice."""
    p = tmp_path / "d.jsonl"
    row = _tw_row(CTX, [["Alpha", 0]])
    row["context"] = "{not valid json"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        TW.load_twowiki(p)


def test_twowiki_missing_file_points_at_the_notebook(tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        TW.load_twowiki(tmp_path / "absent.parquet")


def test_twowiki_license_states_what_was_and_was_not_verified():
    prov = TW.provenance()
    assert prov["license"] == "Apache-2.0"
    assert prov["license_verified_in_hf_repo"] is False
    assert "NO LICENSE file" in prov["license_source"]
    assert prov["hf_file"] == "dev.parquet"
    assert prov["hf_file_bytes"] == 30_056_098
    assert "test labels are withheld" in prov["split_note"]
    rows = [good_row(i) for i in range(4)]
    header = good_header()
    header["pool_mode"] = "hybrid"
    with pytest.raises(ValueError, match="pool_mode must be"):
        CP.validate_suite_records(rows, header)


def test_rejects_duplicate_ids():
    """A duplicate WITHIN one half: same id twice, both calibrate. Disjointness
    is unaffected, so this reaches the duplicate check."""
    rows = [good_row(i) for i in range(4)]
# --- item 7: the pre-resolution-rows defect, closed structurally ---
# This exact mistake shipped a 500-row file with empty gold ids in Notebook 0,
# and was re-made by the B6 and B7 build scripts. Three times is structural,
# not accidental, so the writer now refuses the shape by construction.

def test_writer_hard_fails_on_pre_resolution_rows(tmp_path):
    """The regression test for the recurring defect, written the way it
    actually happened: the caller resolved nothing and passed raw rows."""
    rows = [good_row(i, resolved=False) for i in range(4)]
    p = tmp_path / "toy_suite_records.jsonl"
    with pytest.raises(CP.UnresolvedRowsError, match="pre-resolution"):
        CP.write_suite_records(p, rows, good_header())
    assert not p.exists(), "nothing may be written for a rejected build"


def test_pre_resolution_error_names_the_actual_mistake():
    """It must not surface as a generic empty/dangling-id error, or the fix is
    indistinguishable from the validation that was already there."""
    rows = [good_row(0, resolved=True), good_row(1, resolved=False)]
    with pytest.raises(CP.UnresolvedRowsError) as exc:
        CP.assert_resolved(rows, "'toy' rows")
    msg = str(exc.value)
    assert "_paragraphs" in msg and "_supporting_local" in msg
    assert "Notebook 0, B6 and B7" in msg, "the regression history is the point"
    assert "q1" in msg, "the offending row id must be named"


def test_empty_gold_alone_is_also_refused():
    """Empty supporting_ids is the other fingerprint of pre-resolution rows,
    even with the private keys already stripped."""
    rows = [good_row(i) for i in range(4)]
    rows[2]["supporting_ids"] = []
    with pytest.raises(CP.UnresolvedRowsError, match="never ran"):
        CP.assert_resolved(rows)


def test_unresolved_rows_error_is_a_valueerror():
    """Callers that already catch ValueError keep working."""
    assert issubclass(CP.UnresolvedRowsError, ValueError)


def test_resolved_rows_pass_the_gate():
    rows = [good_row(i) for i in range(4)]
    assert CP.assert_resolved(rows) is None


def test_build_suite_records_resolves_internally(tmp_path):
    """The structural close: the caller hands over the RAW sample plus the
    split labels, and never touches a resolved list at all — so the mistake
    has no code path to live in."""
    sample = []
    for i in range(4):
        rec, paras = _toy_raw(i)
        sample.append(rec)
    split_rows = [dict(r, split=(CP.SPLIT_CALIBRATE if i % 2 == 0
                                 else CP.SPLIT_REPORT))
                  for i, r in enumerate(sample)]
    p = tmp_path / "toy_suite_records.jsonl"
    summary = CP.build_suite_records(p, sample, split_rows, good_header(),
                                     pool_mode=CP.POOL_MODE_PER_QUESTION)
    header, rows = CP.read_suite_records(p)
    assert summary["rows"] == 4
    assert all(r["supporting_ids"] for r in rows), "resolve must have happened"
    assert not any("_paragraphs" in r for r in rows)


def test_build_suite_records_shared_mode_writes_a_pool(tmp_path):
    sample = [_toy_raw(i)[0] for i in range(4)]
    split_rows = [dict(r, split=(CP.SPLIT_CALIBRATE if i % 2 == 0
                                 else CP.SPLIT_REPORT))
                  for i, r in enumerate(sample)]
    p = tmp_path / "toy_suite_records.jsonl"
    summary = CP.build_suite_records(p, sample, split_rows, good_header(),
                                     pool_mode=CP.POOL_MODE_SHARED)
    assert summary["pool_file"] == "toy_pool.jsonl"
    header, rows = CP.read_suite_records(p)
    assert all(r["supporting_ids"] for r in rows)
    assert all(not r["paragraphs"] for r in rows)


def test_build_suite_records_rejects_an_unlabelled_id(tmp_path):
    sample = [_toy_raw(i)[0] for i in range(4)]
    split_rows = [dict(sample[i], split=CP.SPLIT_CALIBRATE) for i in (0, 1)]
    with pytest.raises(ValueError, match="no split label"):
        CP.build_suite_records(tmp_path / "x.jsonl", sample, split_rows,
                               good_header(),
                               pool_mode=CP.POOL_MODE_PER_QUESTION)


def _toy_raw(i):
    """A RAW (pre-resolution) toy record with its own paragraph pool."""
    paras = [{"title": f"t{j}", "text": f"body {j} for q{i}"} for j in range(3)]
    rec = make_question(id=f"q{i}", question=f"Q{i}?", answer="A",
                        aliases=[], type="2hop", hops=2)
    rec["_paragraphs"] = paras
    rec["_supporting_local"] = [0]
    return rec, paras


# PLACEHOLDER_TWOWIKI
    rows[2]["id"] = rows[0]["id"]   # q0 twice, both calibrate
    rows[2]["split"] = CP.SPLIT_CALIBRATE
    rows[3]["split"] = CP.SPLIT_CALIBRATE
    header = good_header(n_cal=3, n_rep=1)
    with pytest.raises(ValueError, match="duplicate"):
        CP.validate_suite_records(rows, header)


def test_rejects_missing_row_field():
    rows = [good_row(i) for i in range(4)]
    del rows[1]["answer"]
    with pytest.raises(ValueError, match="missing required key"):
        CP.validate_suite_records(rows, good_header())


def test_failed_validation_writes_nothing(tmp_path):
    """A rejected build must leave no half-written artifact for the runner."""
    rows = [good_row(i) for i in range(4)]
    rows[0]["supporting_ids"] = []
    p = tmp_path / "toy_suite_records.jsonl"
    with pytest.raises(ValueError):
        CP.write_suite_records(p, rows, good_header())
    assert not p.exists()


# PLACEHOLDER_TWOWIKI
# PLACEHOLDER_REJECTS
