"""HotpotQA suite adapter (v0.3 I-1 slice 3) — the continuity dataset.

HotpotQA is the dataset every earlier result in this project was measured on, so
this adapter's job is as much "do not change the measurement" as "emit contract
records". Two deliberate choices:

1. **It reads the local RAG-Gate files by path and never copies, uploads or
   commits them.** They belong to Anmol's private research checkout. Paths
   resolve through the existing ``jevrag._rag_gate.rag_gate_root()`` pattern,
   overridable by argument or ``RAG_GATE_PATH``.
2. **It does not edit ``hotpotqa.py``.** That module is the V1 headline path
   (EM/F1, the frozen val/test split, the prebuilt index) and is protected by
   continuity, so this is a separate module.

⚠️  SHAPE NOTE — HotpotQA's local file is a THIRD shape, different from both
MuSiQue and 2Wiki:
    context          -> {"titles": [...], "sentences": [[...], ...]}  (a dict)
    supporting_facts -> {"titles": [...], "sent_ids": [...]}          (a dict)
2Wiki stores the same information as JSON *strings*; this file stores native
dicts. An adapter written against either will not read the other.

GOLD MAPPING: by title, not by index. `supporting_facts.titles` may contain the
same title twice (two sentences from one paragraph), so titles are de-duplicated
while preserving order. A gold title absent from `context.titles` is a build
error, not a warning — verified 0 on all 1,000 rows.

SPLIT RELATION: the file carries a frozen val 300 / test 700 split (split_seed
20260715, val_fraction 0.3). The suite draws its OWN 500 from the 1,000 and its
own stratified 250/250 calibrate/report split. The original label is preserved
per row as `frozen_split` for traceability, and both labels are stated in the
header so nobody mistakes the suite split for the frozen one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .closed_pool import make_question

#: Relative locations inside the RAG-Gate checkout.
QUESTIONS_REL = Path("data") / "hotpotqa_1000.json"
CORPUS_REL = Path("data") / "hotpotqa_corpus.json"

#: License, as stated on the dataset card. Not verified in-repo: the local files
#: carry no LICENSE and the RAG-Gate checkout is not the upstream distribution.
#: Recorded as card-sourced rather than claimed as verified.
LICENSE = "cc-by-sa-4.0"
LICENSE_SOURCE = (
    "HotpotQA dataset card (cc-by-sa-4.0); NOT verified against a LICENSE file "
    "in the local RAG-Gate checkout — the local files carry none")


def rag_gate_root() -> Path:
    """Root of the RAG-Gate checkout, via the project's existing accessor."""
    from .._rag_gate import rag_gate_root as _root

    return _root()


def _paths(root: Path | None = None) -> tuple[Path, Path]:
    base = Path(root) if root is not None else rag_gate_root()
    return base / QUESTIONS_REL, base / CORPUS_REL


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _paragraphs(context: Any) -> list[dict[str, Any]]:
    """context -> [{title, text}] in file order.

    Accepts the local file's native dict ({"titles", "sentences"}) and also the
    JSON-string / list-of-pairs forms 2Wiki uses, so one code path covers both.
    Order is preserved: gold mapping indexes into this list.
    """
    ctx = context
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    if isinstance(ctx, dict):
        titles = list(ctx.get("titles") or [])
        sents = list(ctx.get("sentences") or [])
    elif isinstance(ctx, list):
        titles, sents = [], []
        for entry in ctx:
            titles.append(str(entry[0]))
            s = entry[1] if len(entry) > 1 else []
            sents.append(s if isinstance(s, list) else [str(s)])
    else:
        raise ValueError(f"context is neither dict nor list: {type(ctx).__name__}")
    if len(titles) != len(sents):
        raise ValueError(
            f"context titles/sentences length mismatch: {len(titles)} vs "
            f"{len(sents)}")
    return [{"idx": i, "title": str(t),
             "text": " ".join(str(s) for s in (sents[i] or []))}
            for i, t in enumerate(titles)]


def _gold_titles(supporting_facts: Any) -> list[str]:
    """supporting_facts -> ordered, de-duplicated gold titles.

    De-duplicated because one paragraph can supply two supporting sentences; the
    pool id is per-paragraph, so two sentences in one paragraph are one gold
    paragraph.
    """
    sf = supporting_facts
    if isinstance(sf, str):
        sf = json.loads(sf)
    if isinstance(sf, dict):
        raw = list(sf.get("titles") or [])
    elif isinstance(sf, list):
        raw = [str(f[0]) for f in sf if isinstance(f, (list, tuple)) and f]
    else:
        raise ValueError(f"supporting_facts is neither dict nor list: "
                         f"{type(sf).__name__}")
    out: list[str] = []
    for t in raw:
        t = str(t)
        if t not in out:
            out.append(t)
    return out

def load_hotpotqa(root: str | Path | None = None
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the 1,000-question local file as common records + a health report.

    Health mirrors the 2Wiki adapter: the failure modes here (a gold title with
    no matching context paragraph, a row with no usable question or answer) are
    both silent unless counted, and both would ship a sample whose rows cannot be
    scored against evidence.
    """
    qpath, _ = _paths(Path(root) if root is not None else None)
    if not qpath.exists():
        raise FileNotFoundError(
            f"{qpath} not found. This adapter never downloads; point RAG_GATE_"
            f"PATH at the local RAG-Gate checkout.")

    data = json.loads(qpath.read_text(encoding="utf-8"))
    health: dict[str, Any] = {
        "file": str(qpath),
        "split_seed": data.get("split_seed"),
        "val_fraction": data.get("val_fraction"),
        "n_rows": 0, "n_kept": 0,
        "frozen_splits": {}, "types": {}, "levels": {},
        "paragraph_histogram": {},
        "n_dropped_no_answer": 0, "n_unmapped_title": 0,
        "unmapped_examples": [], "dropped_examples": [],
    }

    records: list[dict[str, Any]] = []
    for lineno, row in enumerate(data.get("questions") or []):
        health["n_rows"] += 1
        qid = str(row.get("id") or f"hotpot:{lineno}")
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            health["n_dropped_no_answer"] += 1
            if len(health["dropped_examples"]) < 10:
                health["dropped_examples"].append(qid)
            continue

        paras = _paragraphs(row.get("context"))
        titles = _gold_titles(row.get("supporting_facts"))
        present = {p["title"] for p in paras}
        unmapped = [t for t in titles if t not in present]

        health["n_kept"] += 1
        frozen = str(row.get("split") or "unknown")
        qtype = str(row.get("type") or "unknown")
        level = str(row.get("level") or "unknown")
        health["frozen_splits"][frozen] = health["frozen_splits"].get(frozen, 0) + 1
        health["types"][qtype] = health["types"].get(qtype, 0) + 1
        health["levels"][level] = health["levels"].get(level, 0) + 1
        key = str(len(paras))
        health["paragraph_histogram"][key] = \
            health["paragraph_histogram"].get(key, 0) + 1
        if unmapped:
            health["n_unmapped_title"] += 1
            if len(health["unmapped_examples"]) < 10:
                health["unmapped_examples"].append(
                    {"id": qid, "titles": unmapped[:3]})

        rec = make_question(id=qid, question=question, answer=answer,
                            aliases=[], type=qtype, hops=None)
        rec["_paragraphs"] = paras
        rec["_supporting_titles"] = titles
        rec["_supporting_local"] = [
            p["idx"] for p in paras if p["title"] in set(titles)]
        rec["supporting_sent_ids"] = list(
            (row.get("supporting_facts") or {}).get("sent_ids") or [])
        rec["level"] = level
        rec["frozen_split"] = frozen   # traceability only; not the suite split
        records.append(rec)

    health["unmapped_rate"] = (
        health["n_unmapped_title"] / health["n_kept"]) if health["n_kept"] else None
    return records, health


def load_corpus(root: str | Path | None = None) -> list[dict[str, Any]]:
    """The 9,769-document corpus the earlier smokes indexed, as {title, text}.

    Used only for the "how does the suite pool relate to the full corpus"
    comparison; the suite itself builds its pool from the sampled questions.
    """
    _, cpath = _paths(Path(root) if root is not None else None)
    if not cpath.exists():
        raise FileNotFoundError(f"{cpath} not found.")
    data = json.loads(cpath.read_text(encoding="utf-8"))
    docs = data.get("documents") or []
    return [{"title": str(d.get("title", "")), "text": str(d.get("text", ""))}
            for d in docs]


def provenance(root: str | Path | None = None) -> dict[str, Any]:
    """Source identity for the contract header, per FILE (two files, not one)."""
    qpath, cpath = _paths(Path(root) if root is not None else None)
    out: dict[str, Any] = {
        "dataset": "hotpotqa",
        "source_kind": "local RAG-Gate copies (not an upstream download)",
        "questions_file": qpath.name,
        "corpus_file": cpath.name,
        "license": LICENSE,
        "license_source": LICENSE_SOURCE,
        "license_verified_in_repo": False,
        "split_note": (
            "the local file carries a frozen val 300 / test 700 split; the suite "
            "draws its OWN 500 and its own stratified 250/250 calibrate/report "
            "split. The original label is preserved per row as frozen_split for "
            "traceability and is NOT the suite split."),
    }
    for key, path in (("questions", qpath), ("corpus", cpath)):
        if path.exists():
            out[f"{key}_bytes"] = path.stat().st_size
            out[f"{key}_sha256"] = file_sha256(path)
        else:
            out[f"{key}_error"] = f"{path} not found"
    return out

# PLACEHOLDER_LOAD