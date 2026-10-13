"""Offline tests for jevrag.benchmarks.musique — tiny hand-built fixtures.

No network and no dataset download: every fixture is written to tmp_path as a
few-hundred-byte JSON or JSONL file. These cover the adapter's contract with the
common record shape, and specifically the places where a wrong mapping would be
invisible downstream — gold-paragraph index mapping and alias handling.
"""

import json

import pytest

from jevrag.benchmarks.closed_pool import (
    build_pool,
    resolve_supporting_ids,
    score_with_aliases,
)
from jevrag.benchmarks.musique import (
    HF_FILE,
    HF_REPO_ID,
    LICENSE,
    load_musique,
    provenance,
)

RAW_ROW = {
    "id": "2hop__0001_0002",
    "question": "Who founded the Zylophone Committee?",
    "answer": "Utrecht",
    "answer_aliases": ["Utrecht, Netherlands"],
    "paragraphs": [
        {"idx": 0, "title": "Unrelated", "paragraph_text": "Nothing here.",
         "is_supporting": False},
        {"idx": 1, "title": "Committee", "paragraph_text": "The committee met.",
         "is_supporting": True},
        {"idx": 2, "title": "City", "paragraph_text": "Utrecht hosted it.",
         "is_supporting": True},
    ],
    "answerable": True,
}


@pytest.fixture
def jsonl_file(tmp_path):
    p = tmp_path / "musique.jsonl"
    p.write_text(json.dumps(RAW_ROW) + "\n", encoding="utf-8")
    return p


# --- provenance / license disclosure ---

def test_license_is_cc_by_4_and_cited():
    assert LICENSE == "CC-BY-4.0"
    prov = provenance()
    assert prov["license"] == "CC-BY-4.0"
    assert "stonybrooknlp/musique" in prov["license_source"]


def test_provenance_names_the_exact_source_used():
    """This repo has NO configs and NO splits -- the earlier assumed
    "answerable_dev"/"dev" names never existed and were written into a manifest
    that a reviewer then had to correct. Real-file tests in
    test_musique_real.py check the same thing against the actual download."""
    prov = provenance()
    assert prov["hf_repo_id"] == HF_REPO_ID == "dgslibisey/MuSiQue"
    assert prov["hf_repo_type"] == "dataset"
    assert prov["hf_file"] == HF_FILE == "musique_ans_v1.0_dev.jsonl"
    assert prov["hf_configs"] is None
    assert prov["hf_splits"] is None
    assert "answerable" in prov["variant_rationale"].lower()


def test_hop_parsing_covers_every_real_prefix():
    """The regression: only `2hop__` parsed, so 3hop*/4hop* fell to None."""
    for qid, hops in (("2hop__a_b", 2), ("3hop1__a_b", 3), ("3hop2__a_b", 3),
                      ("4hop1__a_b", 4), ("4hop2__a_b", 4), ("4hop3__a_b", 4)):
        from jevrag.benchmarks.musique import _hops, id_prefix

        assert _hops(qid, {"id": qid, "paragraphs": []}) == hops
        assert id_prefix(qid) == qid.split("__")[0]


# --- loading, both file shapes ---

def test_loads_jsonl(jsonl_file):
    rows = load_musique(jsonl_file)
    assert len(rows) == 1
    assert rows[0]["id"] == "2hop__0001_0002"
    assert rows[0]["question"] == RAW_ROW["question"]


def test_loads_json_array(tmp_path):
    p = tmp_path / "musique.json"
    p.write_text(json.dumps([RAW_ROW]), encoding="utf-8")
    assert len(load_musique(p)) == 1


def test_missing_file_raises_pointing_at_the_notebook(tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        load_musique(tmp_path / "absent.jsonl")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_musique(p)


def test_malformed_line_raises_rather_than_skipping(tmp_path):
    """A silently dropped row in a 500-question sample is an invisible
    denominator change, so this must be loud."""
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(RAW_ROW) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_musique(p)


# --- the common record shape ---

def test_emits_the_common_record_fields(jsonl_file):
    r = load_musique(jsonl_file)[0]
    for key in ("id", "question", "answer", "aliases", "type", "hops",
                "split", "supporting_ids"):
        assert key in r


def test_hops_parsed_from_the_id(jsonl_file):
    assert load_musique(jsonl_file)[0]["hops"] == 2
    assert load_musique(jsonl_file)[0]["type"] == "2hop"


def test_hops_none_when_the_id_carries_none(tmp_path):
    row = dict(RAW_ROW, id="plain-1")
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    r = load_musique(p)[0]
    assert r["hops"] is None
    # `type` is the dataset's own id prefix now, which falls back to "unknown"
    # when the id carries none. hops stays None rather than being invented as 0.
    assert r["type"] == "unknown"
    assert r["id_prefix"] == "unknown"


def test_aliases_exclude_the_canonical_answer(tmp_path):
    row = dict(RAW_ROW, answer_aliases=["Utrecht, Netherlands", "Utrecht"])
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert "Utrecht" not in load_musique(p)[0]["aliases"]


def test_rows_without_a_gold_answer_are_dropped(tmp_path):
    row = dict(RAW_ROW, answer="", answer_aliases=[])
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_musique(p) == []


def test_answerable_only_false_keeps_empty_gold_rows(tmp_path):
    row = dict(RAW_ROW, answer="")
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(load_musique(p, answerable_only=False)) == 1


def test_rows_without_a_question_are_dropped(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(dict(RAW_ROW, question="")) + "\n", encoding="utf-8")
    assert load_musique(p) == []


# --- gold-paragraph mapping: the part that must not silently drift ---

def test_supporting_local_indices_point_at_the_is_supporting_rows(jsonl_file):
    r = load_musique(jsonl_file)[0]
    paras = r["_paragraphs"]
    assert r["_supporting_local"] == [1, 2]
    for i in r["_supporting_local"]:
        assert paras[i]["is_supporting"] is True


def test_paragraph_order_is_preserved(jsonl_file):
    """Local indices address position in this list; reordering would mis-map
    the gold evidence while still looking plausible."""
    paras = load_musique(jsonl_file)[0]["_paragraphs"]
    assert [p["idx"] for p in paras] == [0, 1, 2]
    assert paras[1]["title"] == "Committee"


def test_paragraph_text_is_read_from_paragraph_text(jsonl_file):
    paras = load_musique(jsonl_file)[0]["_paragraphs"]
    assert paras[0]["text"] == "Nothing here."


def test_falls_back_to_the_decomposition_field(tmp_path):
    """A file with no is_supporting flags must still recover gold positions."""
    row = {"id": "2hop__9_9", "question": "Q?", "answer": "A",
           "paragraphs": [{"idx": 0, "title": "t0", "paragraph_text": "zero"},
                          {"idx": 1, "title": "t1", "paragraph_text": "one"}],
           "paragraph_support_idx": [1]}
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_musique(p)[0]["_supporting_local"] == [1]


def test_supports_the_hotpot_style_title_sentences_shape(tmp_path):
    row = {"id": "2hop__1_1", "question": "Q?", "answer": "A",
           "paragraphs": [["Title A", ["sent one.", "sent two."]]],
           "paragraph_support_idx": [0]}
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    paras = load_musique(p)[0]["_paragraphs"]
    assert paras[0]["title"] == "Title A"
    assert "sent one." in paras[0]["text"]


def test_gold_resolves_through_the_closed_pool(jsonl_file):
    rows = load_musique(jsonl_file)
    pool, _ = build_pool(rows)
    resolved = resolve_supporting_ids(rows, pool)
    assert len(resolved[0]["supporting_ids"]) == 2
    assert resolved[0]["supporting_unresolved"] == 0


def test_duplicate_paragraphs_across_questions_collapse_in_the_pool(tmp_path):
    shared = {"idx": 0, "title": "Shared", "paragraph_text": "Same text."}
    rows = []
    for i in range(3):
        rows.append({"id": f"2hop__{i}_1", "question": f"Q{i}?", "answer": "A",
                     "paragraphs": [dict(shared)],
                     "paragraph_support_idx": [0]})
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded = load_musique(p)
    pool, header = build_pool(loaded)
    assert len(pool) == 1
    assert header["duplicates_removed"] == 2
    # All three questions must still point at that one pool entry.
    for rec in resolve_supporting_ids(loaded, pool):
        assert rec["supporting_ids"] == ["p0"]


# --- alias EM through the shared scorer ---

def test_alias_em_accepts_the_alias_answer(jsonl_file):
    r = load_musique(jsonl_file)[0]
    assert score_with_aliases("Utrecht, Netherlands", r)["em"] is True


def test_alias_em_rejects_a_wrong_answer(jsonl_file):
    r = load_musique(jsonl_file)[0]
    assert score_with_aliases("Amsterdam", r)["em"] is False


def test_answerable_flag_is_carried(jsonl_file):
    assert load_musique(jsonl_file)[0]["answerable"] is True


def test_missing_id_gets_a_synthesized_stable_one(tmp_path):
    row = dict(RAW_ROW)
    row.pop("id")
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_musique(p)[0]["id"] == "musique:0"