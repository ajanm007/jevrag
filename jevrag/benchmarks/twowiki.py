"""2WikiMultihopQA adapter — second dataset on the v0.3 closed-pool suite.

2WikiMultihopQA (Ho et al., COLING 2020; upstream https://github.com/Alab-NII/2wikimultihop)
is a multi-hop set built for reasoning-step evaluation. It follows the HotpotQA
file format, which is why the suite can treat it almost identically to MuSiQue
— but with two real differences this adapter has to absorb:

1. ``context`` and ``supporting_facts`` are JSON **strings**, not native nested
   lists. Verified on the real dev.parquet: they parse cleanly, but an adapter
   that reads them as lists gets nothing. (The local HotpotQA file has the same
   quirk.)
2. Gold evidence is given as ``(title, sentence_index)`` pairs, not as a boolean
   per paragraph. Mapping is therefore **by title**, and a supporting title that
   does not appear in ``context`` is a build error rather than a silent skip.

⚠️  LICENSE STATEMENT — what I verified and what I did not:
  - VERIFIED: the upstream GitHub repo `Alab-NII/2wikimultihop` carries an
    Apache-2.0 LICENSE file, and the HF card for `xanhho/2WikiMultihopQA`
    declares the dataset "Apache-2.0".
  - NOT VERIFIED: there is **no LICENSE file inside the Hugging Face repo**
    itself (HF tree listing 2026-10-01: .gitattributes, .gitignore, the loading
    script, README.md, two convert scripts, three parquet files). The license
    here is inherited from upstream via the mirror, not from an in-repo license
    file. Flagged rather than asserted.

SOURCE USED: repo `xanhho/2WikiMultihopQA`, file `dev.parquet`
(30,056,098 bytes). **test labels are withheld** by the dataset authors, so
`dev` is the labelled split and the suite must use it.

Like `musique.py`, this module NEVER downloads: it reads a local path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .closed_pool import make_question

HF_REPO_ID = "xanhho/2WikiMultihopQA"
HF_FILE = "dev.parquet"
HF_FILE_BYTES = 30_056_098
UPSTREAM_REPO = "https://github.com/Alab-NII/2wikimultihop"
LICENSE = "Apache-2.0"
LICENSE_SOURCE = (
    f"{UPSTREAM_REPO} LICENSE file (Apache-2.0) + HF card; NO LICENSE file "
    f"inside the HF repo itself — inherited from upstream, not verified in-repo")

#: The four question types this dataset ships.
TYPES = ("comparison", "inference", "compositional", "bridge_comparison")

#: Paragraphs per question in dev.parquet (verified 10 on 2,000 rows).
PARAGRAPHS_PER_QUESTION = 10


def _iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dicts from a .parquet or .jsonl file.

    Parquet needs pyarrow; jsonl is supported so the same adapter reads a
    converted file without a second code path. Malformed input raises — a
    silently dropped row in a 500-question sample is an invisible denominator
    change.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. This adapter never downloads; the notebook writes "
            f"the file first."
        )
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(str(p)).to_pylist()
        return
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{p} is empty")
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno} is not valid JSON: {exc}") from exc


def _loads(value: Any, what: str) -> Any:
    """Parse a field that may be a JSON string or already a native object."""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{what} is a string but not valid JSON: {exc}")
    return value


def _paragraphs(context: Any) -> list[dict[str, Any]]:
    """context -> [{title, text}] in file order.

    Order is preserved because gold mapping indexes into this list, and a
    reordering here would silently mis-map evidence.
    """
    ctx = _loads(context, "context")
    if not isinstance(ctx, list):
        raise ValueError(f"context should be a list, got {type(ctx).__name__}")
    out = []
    for i, entry in enumerate(ctx):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(f"context[{i}] is not a [title, sentences] pair")
        sents = entry[1] if isinstance(entry[1], list) else [str(entry[1])]
        out.append({"idx": i, "title": str(entry[0]),
                    "text": " ".join(str(s) for s in sents)})
    return out


def _supporting_titles(supporting_facts: Any) -> list[str]:
    """supporting_facts -> ordered, de-duplicated list of gold TITLES.

    2Wiki gives ``[title, sent_id]`` pairs. Only the title identifies the
    paragraph; the sentence index narrows *within* it and is kept in the record
    for traceability but does not change which paragraph is gold. De-duplicated
    because one question can list two sentences from one paragraph.
    """
    facts = _loads(supporting_facts, "supporting_facts")
    if not isinstance(facts, list):
        raise ValueError(
            f"supporting_facts should be a list, got {type(facts).__name__}")
    titles: list[str] = []
    for fact in facts:
        if not isinstance(fact, (list, tuple)) or not fact:
            raise ValueError(
                f"supporting_facts entry is not [title, sent_id]: {fact!r}")
        title = str(fact[0])
        if title not in titles:
            titles.append(title)
    return titles


def load_twowiki(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load 2Wiki dev rows as common records, plus a build-health report.

    Returns ``(records, health)``. The report exists because this adapter's
    failure modes are both silent if unchecked: a supporting title absent from
    ``context`` (gold that maps to no paragraph) and a row with no usable
    question or answer. Those are counted so the caller can treat a non-zero
    count as a build error instead of shipping unscoreable questions.
    """
    records: list[dict[str, Any]] = []
    health: dict[str, Any] = {
        "n_rows": 0, "n_kept": 0, "types": {}, "paragraph_histogram": {},
        "n_dropped_no_answer": 0, "n_unmapped_title": 0,
        "n_supporting_titles": 0, "unmapped_examples": [],
        "dropped_examples": [],
    }

    for lineno, row in enumerate(_iter_rows(path)):
        health["n_rows"] += 1
        qid = str(row.get("_id") or f"2wiki:{lineno}")
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        qtype = str(row.get("type") or "unknown")
        if not question or not answer:
            health["n_dropped_no_answer"] += 1
            if len(health["dropped_examples"]) < 10:
                health["dropped_examples"].append(qid)
            continue

        paras = _paragraphs(row.get("context"))
        titles = _supporting_titles(row.get("supporting_facts"))
        present = {p["title"] for p in paras}
        unmapped = [t for t in titles if t not in present]

        health["n_kept"] += 1
        health["types"][qtype] = health["types"].get(qtype, 0) + 1
        key = str(len(paras))
        health["paragraph_histogram"][key] = \
            health["paragraph_histogram"].get(key, 0) + 1
        health["n_supporting_titles"] += len(titles)
        if unmapped:
            health["n_unmapped_title"] += 1
            if len(health["unmapped_examples"]) < 10:
                health["unmapped_examples"].append(
                    {"id": qid, "titles": unmapped[:3]})

        rec = make_question(id=qid, question=question, answer=answer,
                            aliases=[], type=qtype, hops=None)
        rec["_paragraphs"] = paras
        # Titles, not indices: the pool builder dedups and re-numbers, so the
        # only stable handle across dedup is the title.
        rec["_supporting_titles"] = titles
        rec["_supporting_local"] = [
            p["idx"] for p in paras if p["title"] in set(titles)]
        rec["supporting_facts_sent_ids"] = [
            list(f) for f in _loads(row.get("supporting_facts"),
                                    "supporting_facts")]
        records.append(rec)

    health["unmapped_rate"] = (
        health["n_unmapped_title"] / health["n_kept"]) if health["n_kept"] else None
    return records, health


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """sha256 of the file actually read."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance(path: str | Path | None = None) -> dict[str, Any]:
    """Dataset provenance for the suite-records header."""
    out: dict[str, Any] = {
        "dataset": "twowiki",
        "hf_repo_id": HF_REPO_ID,
        "hf_repo_type": "dataset",
        "hf_file": HF_FILE,
        "hf_file_bytes": HF_FILE_BYTES,
        "upstream_repo": UPSTREAM_REPO,
        "license": LICENSE,
        "license_source": LICENSE_SOURCE,
        "license_verified_in_hf_repo": False,
        "split_note": ("test labels are withheld by the dataset authors; dev "
                       "is the labelled split"),
    }
    if path is not None:
        p = Path(path)
        if p.exists():
            out["local_file"] = p.name
            out["local_file_bytes"] = p.stat().st_size
            out["local_file_sha256"] = file_sha256(p)
        else:
            out["local_file_error"] = f"{p} not found"
    return out