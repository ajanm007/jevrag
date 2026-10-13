"""MuSiQue adapter — the first dataset on the v0.3 closed-pool suite.

MuSiQue (Trivedi et al., TACL 2022; repo https://github.com/stonybrooknlp/musique)
is the cleanest of the four D1 datasets: every question ships its own paragraph
pool with an explicit ``is_supporting`` flag per paragraph, so there is no
sentence-id bookkeeping and no distractor heuristic.

⚠️  LICENSE DISCLOSURE: MuSiQue is distributed under **CC BY 4.0** — stated in the
upstream repository README ("MuSiQue is distributed under a CC BY 4.0 License")
and its LICENSE file. Re-verified 2026-10-01. This is an attribution-share
license: redistributing the data (as opposed to evaluating against it) carries
CC BY 4.0 obligations. The Hugging Face mirrors carry no license field of their
own, so the upstream repo is the authority cited here.

SOURCE USED (per the packet, and per the B3 recon):
  HF dataset id: ``dgslibisey/MuSiQue``
  config:        ``answerable_dev`` (the ANSWERABLE variant, dev split)
  local file:    written by the notebook (Notebook 0) — this module never
                 downloads anything. It reads a local path.

Why the answerable dev variant: unanswerable questions have no gold paragraphs,
which makes "recall of gold supporting paragraphs" undefined for them and would
silently deflate every pool-sanity number. The ablation subsets (2hop/3hop/4hop)
are also available upstream; ``answerable_dev`` is the superset and its rows
carry the hop count, which is what the stratified sampler stratifies on.

Deliberately NOT implemented here: bulk download, HF ``load_dataset`` calls, or
any network access. The adapter is a pure file reader so the same code runs on
Kaggle, on Colab and in the tests without a network.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

from .closed_pool import make_question

#: Provenance constants, emitted into run manifests so a result states which
#: dataset and variant produced it.
HF_REPO_ID = "dgslibisey/MuSiQue"
#: The repo has NO configs and NO splits -- verified from the HF tree API on
#: 2026-10-01. It holds exactly two files; the earlier assumed config name
#: "answerable_dev" did not exist and was never a real handle.
#:   musique_ans_v1.0_dev.jsonl   (30,439,728 bytes)  <- the answerable dev file
#:   musique_ans_v1.0_train.jsonl
HF_FILE = "musique_ans_v1.0_dev.jsonl"
HF_FILE_BYTES = 30_439_728
UPSTREAM_REPO = "https://github.com/stonybrooknlp/musique"
LICENSE = "CC-BY-4.0"
LICENSE_SOURCE = f"{UPSTREAM_REPO} (README + LICENSE, re-verified 2026-10-01)"


def _iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a ``.jsonl`` file, or from a ``.json`` list/array.

    The HF auto-converted export is JSON-lines; the raw upstream download is a
    single JSON array. Supporting both means the notebook can write whichever it
    has without a conversion step, and means the tests can use a tiny fixture in
    either shape. Malformed lines raise rather than being skipped — a silently
    dropped row in a 500-question sample is an invisible denominator change.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. This adapter never downloads; the notebook writes "
            f"the file first (see Notebook 0)."
        )
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{p} is empty")

    if p.suffix == ".jsonl" or text[0] not in "[{":
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
        return

    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("data", "questions", "rows"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{p}: expected a JSON list of records, got "
                         f"{type(data).__name__}")
    yield from data


def _hops(row_id: str, row: dict[str, Any]) -> int | None:
    """Hop count for stratification, or ``None`` when the id doesn't say.

    MuSiQue ids are NOT all ``2hop__``-shaped. The real dev file (2,417 rows)
    uses six prefixes: 2hop, 3hop1, 3hop2, 4hop1, 4hop2, 4hop3. The earlier
    parser matched only a literal ``2hop``, so 1,165 rows silently got
    ``hops=None`` and were stratified as "type=unknown" -- a third of the sample
    in the wrong bucket.

    The hop count is the leading integer of the id prefix; the trailing digit on
    3hop/4hop is a *variant* label (1, 2, 3), not part of the count. Verified on
    every row of the real file: parsed hops == number of ``is_supporting``
    paragraphs, 0 mismatches in 2,417.
    """
    m = re.match(r"^(\d)hop", str(row_id))
    if m:
        return int(m.group(1))
    for field in ("num_hops", "hops", "hop_count"):
        if row.get(field) is not None:
            try:
                return int(row[field])
            except (TypeError, ValueError):
                return None
    return None


def id_prefix(row_id: str) -> str:
    """The dataset's own variant label, e.g. ``2hop``, ``3hop2``, ``4hop1``.

    Kept separate from :func:`_hops` because the two answer different questions:
    "how many hops" vs "which construction variant". Stratify on hops; report
    breakdowns by prefix so a 3hop1-vs-3hop2 difference would be visible.
    """
    m = re.match(r"^(\d+hop\d*)", str(row_id))
    return m.group(1) if m else "unknown"


def _paragraphs(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-question paragraphs as dicts, in the file's original order.

    Order matters: the caller records gold paragraphs by their LOCAL index into
    this list, and those indices resolve to pool ids later. Reordering here would
    silently mis-map the gold evidence.
    """
    out = []
    for i, para in enumerate(row.get("paragraphs") or []):
        if isinstance(para, dict):
            title = str(para.get("title", ""))
            text = str(para.get("paragraph_text", para.get("text", "")))
            is_sup = bool(para.get("is_supporting", False))
        else:
            # Tolerate the [title, sentences] shape the HotpotQA family uses.
            title = str(para[0]) if len(para) > 0 else ""
            text = " ".join(str(s) for s in para[1]) if len(para) > 1 else ""
            is_sup = False
        out.append({"idx": i, "title": title, "text": text,
                    "is_supporting": is_sup})
    return out


def load_musique(path: str | Path, *, answerable_only: bool = True
                 ) -> list[dict[str, Any]]:
    """Load MuSiQue rows as common question records with attached paragraphs.

    Each returned record is the common shape (id, question, answer, aliases,
    type, hops, split, supporting_ids) PLUS two private keys consumed by
    :mod:`jevrag.benchmarks.closed_pool`: ``_paragraphs`` (this question's own
    pool) and ``_supporting_local`` (gold positions within it). Both are
    stripped before the sample is written out.

    ``answerable_only`` keeps rows with a non-empty gold answer. The
    ``answerable_dev`` config is already answerable, but the filter is kept so
    the same adapter can read an unfiltered file without producing empty-gold
    rows that would deflate recall.
    """
    records: list[dict[str, Any]] = []
    for lineno, row in enumerate(_iter_json_records(path)):
        if not isinstance(row, dict):
            raise ValueError(f"row {lineno} is {type(row).__name__}, not a dict")

        answer = str(row.get("answer", "") or "").strip()
        if answerable_only and not answer:
            continue

        raw_id = row.get("id")
        qid = str(raw_id) if raw_id is not None else f"musique:{lineno}"
        question = str(row.get("question", "") or "").strip()
        if not question:
            continue

        aliases = [str(a) for a in (row.get("answer_aliases") or [])
                   if str(a).strip()]
        # MuSiQue's alias list sometimes repeats the canonical answer; drop that
        # so EM isn't effectively scored twice against the same string.
        aliases = [a for a in aliases if a.strip() != answer]

        paras = _paragraphs(row)
        supporting_local = [p["idx"] for p in paras if p["is_supporting"]]
        # If the file carries no is_supporting flags, fall back to MuSiQue's
        # decomposition field, which lists gold paragraph indices.
        if not supporting_local:
            for field in ("paragraph_support_idx", "supporting_paras"):
                val = row.get(field)
                if isinstance(val, list) and all(isinstance(i, int) for i in val):
                    supporting_local = list(val)
                    break

        hops = _hops(qid, row)
        rec = make_question(
            id=qid,
            question=question,
            answer=answer,
            aliases=aliases,
            type=id_prefix(qid),
            hops=hops,
        )
        rec["_paragraphs"] = paras
        rec["_supporting_local"] = supporting_local
        rec["id_prefix"] = id_prefix(qid)
        rec["answerable"] = bool(row.get("answerable", True))
        records.append(rec)
    return records


def hops_match_supporting(records: Sequence[dict[str, Any]]
                          ) -> dict[str, Any]:
    """Cross-check: parsed hop count == number of gold supporting paragraphs.

    The id prefix and the paragraph annotations are two independent statements
    about the same question. When they disagree, one of them is wrong and the
    run should say so rather than silently stratify on a bad number. Returns
    the agreement rate and up to 20 disagreeing ids.
    """
    total = mismatched = 0
    bad: list[dict[str, Any]] = []
    for r in records:
        sup = len(r.get("_supporting_local") or [])
        total += 1
        if r.get("hops") != sup:
            mismatched += 1
            if len(bad) < 20:
                bad.append({"id": r.get("id"), "hops": r.get("hops"),
                            "n_supporting": sup})
    return {"n": total, "n_mismatched": mismatched,
            "agreement_rate": (1.0 - mismatched / total) if total else None,
            "examples": bad}


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """sha256 of the dataset file actually read.

    Provenance that names a file but not its contents is a filename, not a
    provenance record: the same name can be re-uploaded with different rows.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance(path: str | Path | None = None) -> dict[str, Any]:
    """Dataset provenance block for run manifests.

    Records the REAL file identity: this HF repo has no configs and no splits,
    so there is no config/split to name. The file's own size and (when a path is
    supplied) sha256 travel with the manifest, so a number can later be traced
    to the exact bytes that produced it.
    """
    out = {
        "dataset": "musique",
        "hf_repo_id": HF_REPO_ID,
        "hf_repo_type": "dataset",
        "hf_configs": None,
        "hf_splits": None,
        "hf_file": HF_FILE,
        "hf_file_bytes": HF_FILE_BYTES,
        "upstream_repo": UPSTREAM_REPO,
        "license": LICENSE,
        "license_source": LICENSE_SOURCE,
        "variant_rationale": (
            "the answerable dev file: unanswerable questions have no gold "
            "paragraphs, which makes supporting-paragraph recall undefined "
            "for them and would silently deflate pool-sanity numbers"
        ),
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