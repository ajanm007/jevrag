"""Tests for the HotpotQA suite adapter (packet B7).

Offline throughout: the real-file tests read Anmol's RAG-Gate checkout IN PLACE
and are skipped when it is absent, so the suite stays green on a machine that
does not have it and CI never copies the data.
"""

import json

import pytest

from jevrag.benchmarks import hotpotqa_suite as HS

RAG_GATE = HS.rag_gate_root()
QPATH = RAG_GATE / HS.QUESTIONS_REL
CPATH = RAG_GATE / HS.CORPUS_REL

real_only = pytest.mark.skipif(
    not QPATH.exists(),
    reason=f"RAG-Gate questions file not present at {QPATH}")


def _row(titles=("Alpha", "Beta", "Gamma"), sents=None,
         gold=("Alpha",), sents_ids=(0,), answer="A"):
    sents = sents or [["a1", "a2"], ["b1"], ["g1"]][:len(titles)]
    return {
        "id": "hq1", "question": "Q?", "answer": answer, "type": "bridge",
        "level": "hard", "split": "test",
        "context": {"titles": list(titles), "sentences": sents},
        "supporting_facts": {"titles": list(gold), "sent_ids": list(sents_ids)},
    }


# --- shape handling: the local file is a THIRD shape, not 2Wiki's or MuSiQue's ---

def test_parses_native_dict_context_and_supporting():
    """Local HotpotQA stores context/supporting_facts as dicts with
    titles/sentences, not as 2Wiki's JSON strings or MuSiQue's booleans."""
    paras = HS._paragraphs({"titles": ["Alpha", "Beta"],
                            "sentences": [["a1", "a2"], ["b1"]]})
    assert [p["title"] for p in paras] == ["Alpha", "Beta"]
    assert paras[0]["text"] == "a1 a2"
    assert HS._gold_titles({"titles": ["Alpha", "Beta"], "sent_ids": [0, 1]}) == \
        ["Alpha", "Beta"]


def test_also_accepts_the_json_string_shape():
    """2Wiki-shaped input still parses, so one code path covers both."""
    ctx = json.dumps([["Alpha", ["a1"]], ["Beta", ["b1"]]])
    sf = json.dumps([["Alpha", 0]])
    assert [p["title"] for p in HS._paragraphs(ctx)] == ["Alpha", "Beta"]
    assert HS._gold_titles(sf) == ["Alpha"]


def test_gold_titles_are_deduplicated_preserving_order():
    """One paragraph can supply two supporting sentences; the pool id is
    per-paragraph, so two sentences in one paragraph are one gold paragraph."""
    assert HS._gold_titles({"titles": ["B", "A", "B"], "sent_ids": [0, 0, 1]}) \
        == ["B", "A"]


def test_context_title_sentence_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        HS._paragraphs({"titles": ["A", "B"], "sentences": [["only one"]]})


def test_paragraph_order_is_preserved():
    paras = HS._paragraphs({"titles": ["Z", "A", "M"],
                            "sentences": [["z"], ["a"], ["m"]]})
    assert [p["idx"] for p in paras] == [0, 1, 2]
    assert [p["title"] for p in paras] == ["Z", "A", "M"]


# --- missing file ---

def test_missing_file_says_it_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        HS.load_hotpotqa(tmp_path)


# --- provenance / license disclosure ---

def test_provenance_states_license_is_card_sourced_only():
    """The local RAG-Gate checkout carries no LICENSE file, so the claim must
    be marked unverified rather than asserted."""
    prov = HS.provenance(HS.rag_gate_root())
    assert prov["license"] == "cc-by-sa-4.0"
    assert prov["license_verified_in_repo"] is False
    assert "NOT verified" in prov["license_source"]


def test_provenance_names_both_files_and_the_split_note():
    prov = HS.provenance(HS.rag_gate_root())
    assert prov["questions_file"] == "hotpotqa_1000.json"
    assert prov["corpus_file"] == "hotpotqa_corpus.json"
    assert "its OWN 500" in prov["split_note"]


# --- the real file ---

@real_only
def test_real_file_loads_1000_rows():
    recs, health = HS.load_hotpotqa()
    assert health["n_rows"] == 1000
    assert health["n_kept"] == 1000
    assert len(recs) == 1000


@real_only
def test_real_file_has_no_unmapped_gold_titles():
    """A supporting title with no context paragraph is a build error; verified 0
    on the real file, and the build script refuses if it is ever non-zero."""
    _, health = HS.load_hotpotqa()
    assert health["n_unmapped_title"] == 0
    assert health["unmapped_examples"] == []


@real_only
def test_real_file_preserves_the_frozen_split_and_suite_split_is_separate():
    """The suite draws its own 500/250-250; the original val/test label is kept
    per row as frozen_split for traceability and is NOT the suite split."""
    recs, health = HS.load_hotpotqa()
    assert health["frozen_splits"] == {"test": 700, "val": 300}
    assert health["split_seed"] == 20260715
    assert {r["frozen_split"] for r in recs} == {"val", "test"}
    # no record carries a suite split yet -- that is added by the splitter
    assert not any("split" in r and r["split"] for r in recs)


@real_only
def test_every_real_record_has_exactly_two_gold_titles():
    recs, _ = HS.load_hotpotqa()
    assert all(len(r["_supporting_titles"]) == 2 for r in recs)
    assert all(len(r["_supporting_local"]) == 2 for r in recs)


@real_only
def test_real_file_shape():
    recs, health = HS.load_hotpotqa()
    assert health["types"] == {"bridge": 807, "comparison": 193}
    assert health["levels"] == {"hard": 1000}
    # 989 of 1000 have the standard 10 distractor paragraphs
    assert health["paragraph_histogram"]["10"] == 989


@real_only
def test_real_corpus_loads_9769_documents():
    pytest.importorskip("pyarrow")
    docs = HS.load_corpus()
    assert len(docs) == 9769
    assert set(docs[0]) == {"title", "text"}


@real_only
def test_real_file_sha256_is_recorded():
    prov = HS.provenance(HS.rag_gate_root())
    assert len(prov["questions_sha256"]) == 64
    assert prov["questions_bytes"] == QPATH.stat().st_size
    assert prov["corpus_bytes"] == CPATH.stat().st_size