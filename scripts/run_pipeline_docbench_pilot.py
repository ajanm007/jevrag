"""DocBench pilot runner (packet V03-A3; originally V03-A2, tracker I-4 part 1).

Full pipeline (jevrag/pipeline.py, memoization on for both caches, default
thresholds) over every DocBench question with a PDF on disk: docbench-sample/0
(7 q) + docbench-scale/{103: 3, 110: 4, 218: 3, 59: 6} = 23 questions over 5
documents. docbench-qa-scan/ has QA files but no PDFs — verified on disk.

A3 changes vs the A2 version: (1) overwrite guard — existing non-empty outputs
are refused unless --overwrite (the A2 overwrite cost 47% of its spend);
(2) real generator (openai/gpt-4o-mini via OpenRouter, key verified live this
packet), fail-closed on the first generator error, generator tag per row;
(3) persisted per-doc chunk caches (outputs/v03_a3_chunkcache_<doc>.jsonl) so
ingestion is paid once; (4) cache-trust lookups accounted as their own line
item (this closes the A2 one-call/one-token gap); (5) chunk-length stats per
doc; (6) --probe mode for selector diagnosis. Judge-free (EM/F1 only).

Writes (prefix set by --out-prefix, default outputs/v03_a3_pilot):
  <prefix>.jsonl + <prefix>.trace.json (one row per question, one trace/run)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._load_run_keys import load_jev_key, load_openrouter_key  # noqa: E402

from jevrag._rag_gate import jevrag_kaggle_root, rag_gate_root  # noqa: E402
from jevrag.benchmarks.docbench import (  # noqa: E402
    LICENSE_NOTE,
    TEXT_ONLY,
    extract_pdf_pages,
    load_docbench_questions,
)
from jevrag.benchmarks.hotpotqa import exact_match, f1_score  # noqa: E402
from jevrag.decision import JevDecision, estimate_cost_usd  # noqa: E402
from jevrag.pipeline import (  # noqa: E402
    PipelineConfig,
    build_pipeline,
    normalize_query,
    run_pipeline,
)

KAGGLE = jevrag_kaggle_root()
PROMPT_TEMPLATE_FILE = rag_gate_root() / "data" / "prompt_template.txt"
GENERATOR_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: (doc_id, qa_dir, pdf_name, qa_name, seed_cache_qid | None)
DOCS: tuple[tuple[str, str, str, str, str | None], ...] = (
    ("doc0", "docbench-sample", "P19-1598.pdf", "0_qa.jsonl", "doc0:4"),
    ("doc103", "docbench-scale/103",
     "USCOURTS-nmd-1_24-cr-00032-8.pdf", "103_qa.jsonl", None),
    ("doc110", "docbench-scale/110",
     "Country_Reports_on_Terrorism_2020.pdf", "110_qa.jsonl", None),
    ("doc218", "docbench-scale/218", "20220311.pdf", "218_qa.jsonl", None),
    ("doc59", "docbench-scale/59", "NYSE_PEP_2020.pdf", "59_qa.jsonl", None),
)

QUESTION_TYPES = (TEXT_ONLY, "multimodal-t", "meta-data", "unanswerable")
OUT_PREFIX = "outputs/v03_a3_pilot"
CHUNK_CACHE_PREFIX = "v03_a3_chunkcache"


class OverwriteRefused(Exception):
    """Raised when outputs already exist and --overwrite was not passed."""


def checked_out_paths(out_prefix: str, overwrite: bool = False,
                      extra_suffixes: tuple[str, ...] = ()) -> dict[str, Path]:
    """Resolve <prefix>.jsonl/.trace.json (+ extras), fail-closed on clobber.

    Any listed path that exists AND is non-empty refuses unless ``overwrite``
    is true. Missing or empty files are fine (a previous crash may leave an
    empty file behind). Pure function of the filesystem; unit-tested.
    """
    paths = {"rows": REPO_ROOT / f"{out_prefix}.jsonl",
             "trace": REPO_ROOT / f"{out_prefix}.trace.json"}
    for i, suffix in enumerate(extra_suffixes):
        paths[f"extra{i}"] = REPO_ROOT / f"{out_prefix}{suffix}"
    if not overwrite:
        clashes = sorted(str(p) for p in paths.values()
                         if p.exists() and p.stat().st_size > 0)
        if clashes:
            raise OverwriteRefused(
                "refusing to overwrite non-empty output(s): "
                + ", ".join(clashes) + " — pass --overwrite to allow it.")
    return paths


class GeneratorHalted(Exception):
    """Fail-closed stop: the generator errored; no fallback, no retry."""


def fail_closed_answer_fn(inner, tag: str):
    """Wrap a generator: first error of ANY kind halts the run loudly.

    Packet rule: stop on the first 402 (or any non-retryable generator
    error); never fall back silently to extractive. The 402 status is
    reported explicitly when present (openai.APIStatusError.status_code).
    """
    def answer_fn(question: str, evidence: list[dict]):
        try:
            return inner(question, evidence)
        except Exception as exc:  # noqa: BLE001 — halt, never swallow
            code = getattr(exc, "status_code", None)
            raise GeneratorHalted(
                f"generator halted ({tag}): "
                f"{'HTTP ' + str(code) + ' ' if code else ''}"
                f"{type(exc).__name__}: {exc}") from exc
    return answer_fn


def score_question(out: dict, q: dict) -> None:
    """EM + F1 onto an outcome row. Judge-free by packet decision (A3 §4);
    judge_verdict is None with the reason recorded, never zero."""
    out["judge_verdict"] = None
    out["judge_reason"] = "judge-free pilot: EM/F1 against gold only"
    out["em"] = bool(exact_match(out.get("prediction", ""), q["answer"]))
    out["f1"] = f1_score(out.get("prediction", ""), q["answer"])


def stage_split(out: dict) -> dict:
    """Per-question Jev cost split: selection / sufficiency / abstain."""
    sel_live = int(out.get("selection_calls", 0))
    sel_hits = int((out.get("selection") or {}).get("cache_hits", 0))
    sel_toks = sum(int(r.get("jev_input_tokens", 0))
                   for r in (out.get("selection") or {}).get("rounds", []))
    if out.get("sufficiency"):
        srec, strc = out["sufficiency"]["record"], out["sufficiency"]["trace"]
        suff_calls, suff_toks = len(strc), sum(
            int(t.get("input_tokens", 0)) for t in strc)
        gen_toks = int(srec.get("input_tokens", 0)) - suff_toks
    else:
        suff_calls, suff_toks, gen_toks = 0, 0, 0
    if out.get("abstain"):
        arec = out["abstain"]["record"]
        abst_calls, abst_toks = 1, int(arec.get("input_tokens", 0))
    else:
        abst_calls, abst_toks = 0, 0
    return {
        "sel_live": sel_live, "sel_hits": sel_hits, "sel_toks": sel_toks,
        "suff_calls": suff_calls, "suff_toks": suff_toks,
        "abst_calls": abst_calls, "abst_toks": abst_toks,
        "gen_toks": max(gen_toks, 0),
    }


def chunk_length_stats(chunks: list[dict]) -> dict:
    """Min/median/max chunk chars + share over the controller cap."""
    from jevrag.pipeline import DEFAULT_MAX_CHUNK_CHARS  # noqa: E402
    lens = sorted(len(str(c.get("text", ""))) for c in chunks)
    if not lens:
        return {"n": 0}
    mid = lens[len(lens) // 2]
    return {"n": len(lens), "min": lens[0], "median": mid, "max": lens[-1],
            "over_cap": sum(1 for x in lens if x > DEFAULT_MAX_CHUNK_CHARS),
            "cap": DEFAULT_MAX_CHUNK_CHARS}


def run_doc(doc_id: str, qa_dir: str, pdf_name: str, qa_name: str,
            seed_qid: str | None, backends: dict, answer_fn,
            generator_tag: str, chunk_cache_path: str,
            crc_specs: dict | None = None) -> tuple[list[dict], dict]:
    """One document end to end: ingest -> all questions -> scored rows."""
    pages = extract_pdf_pages(KAGGLE / qa_dir / pdf_name)
    questions = load_docbench_questions(doc_id, KAGGLE / qa_dir / qa_name,
                                        types=QUESTION_TYPES)
    print(f"\ndoc {doc_id}: {pdf_name}: {len(pages)} pages, "
          f"{len(questions)} questions", flush=True)

    cache_entries: dict[str, dict[str, str]] = {}
    if seed_qid is not None:
        seed_q = next(q for q in questions if q["id"] == seed_qid)
        cache_entries[normalize_query(seed_q["question"])] = {
            "answer": seed_q["answer"], "gold": seed_q["answer"]}
        print(f"  toy cache: 1 seeded entry ({seed_qid})", flush=True)

    config = PipelineConfig(
        chunk_decision=backends["chunk"],
        selection_decision=backends["selection"],
        sufficiency_decision=backends["sufficiency"],
        abstain_decision=backends["abstain"],
        cache_decision=backends["cache"],
        answer_fn=answer_fn,
        crc_specs=crc_specs or {},
        cache_entries=cache_entries,
        max_chunk_chars=9000,
        memoize_selection=True,
        memoize_chunk=True,
        chunk_cache_path=chunk_cache_path,
    )
    print(f"  chunk cache file: {chunk_cache_path or '(in-memory only)'}",
          flush=True)
    ctx = build_pipeline(config, doc_id, pages)
    ing = ctx.ingestion_trace
    print(f"  ingestion: {ing['n_paragraphs']} paras -> "
          f"{ing['n_candidates']} CB candidates -> {ing['n_splits']} splits "
          f"(splits={ing.get('splits')}, +{ing['force_splits']} force) -> "
          f"{ing['n_chunks']} chunks; jev_calls={ing['jev_calls']} "
          f"cache_hits={ing.get('n_cache_hits')} "
          f"per_round_k={ctx.per_round_k}", flush=True)
    if ing["n_candidates"] > 0 and ing["n_splits"] == 0:
        print("  FLAG: 0 splits over "
              f"{ing['n_candidates']} candidates — all-merge ingestion",
              flush=True)

    # Warm re-ingest through the same store: must cost zero Jev calls.
    ctx2 = build_pipeline(config, doc_id, pages)
    ing2 = ctx2.ingestion_trace
    print(f"  re-ingest: jev_calls={ing2['jev_calls']} "
          f"hits={ing2.get('n_cache_hits')} splits={ing2.get('splits')}",
          flush=True)

    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            out = run_pipeline(ctx, [q])[0]
        except GeneratorHalted:
            raise
        except Exception as exc:  # noqa: BLE001 — crash rows, not aborts
            out = {"question_id": q["id"], "question": q["question"],
                   "gold": q.get("answer", ""), "outcome": "error",
                   "answer": "", "prediction": "", "source": "error",
                   "generator": generator_tag,
                   "error": f"{type(exc).__name__}: {exc}",
                   "error_trace": traceback.format_exc(limit=5),
                   "sufficiency": None, "abstain": None,
                   "selection_calls": 0, "jev_calls": 0,
                   "jev_input_tokens": 0, "thresholds": {},
                   "wall_ms": (time.perf_counter() - t0) * 1000.0}
            print(f"  [{i}/{len(questions)}] {q['id']} ERROR: {out['error']}",
                  flush=True)
            rows.append(out)
            continue
        out["wall_ms_ask"] = (time.perf_counter() - t0) * 1000.0
        out["generator"] = generator_tag
        score_question(out, q)
        sp = stage_split(out)
        out["stage"] = sp
        sel = out.get("selection") or {}
        nfb = sum(1 for r in sel.get("rounds", []) if r.get("fallback_top1"))
        if nfb:
            print(f"  FLAG {q['id']}: fallback_top1 on {nfb} round(s)",
                  flush=True)
        print(f"  [{i}/{len(questions)}] {q['id']} [{q.get('type')}] "
              f"src={out['source']} "
              f"rounds={out['sufficiency']['record']['rounds_used'] if out['sufficiency'] else '-'} "
              f"sel={sp['sel_live']}+{sp['sel_hits']}h "
              f"suff={sp['suff_calls']} abst={sp['abst_calls']} "
              f"-> {out['outcome']} "
              f"em={out['em']} f1={out['f1']:.3f}", flush=True)
        rows.append(out)

    doc_trace = {
        "doc_id": doc_id, "pdf": pdf_name,
        "crc_specs": sorted(crc_specs or {}),
        "generator": generator_tag,
        "chunk_cache_path": chunk_cache_path,
        "ingestion": ing, "reingest": ing2,
        "chunk_lengths": chunk_length_stats(ctx.chunks),
        "per_round_k": ctx.per_round_k,
        "n_questions": len(rows),
        "n_answered": sum(1 for r in rows if r["outcome"] == "answer"),
        "n_no_answer": sum(1 for r in rows if r["outcome"] == "no_answer"),
        "n_cache_served": sum(1 for r in rows if r.get("source") == "cache"),
        "n_errors": sum(1 for r in rows if r["outcome"] == "error"),
    }
    return rows, doc_trace


def run_probe(question_ids: list[str], backends: dict,
              chunk_cache_dir: str) -> dict:
    """Selector diagnosis probe (packet §5), no pipeline.py changes.

    For each qid: warm re-ingest its doc from the PERSISTED chunk cache (0
    Jev calls if the store works cross-process), rebuild BM25, reconstruct
    the round-1 pool deterministically, and judge that pool live with
    decide_query. These are FRESH probe judgments (backend jitter applies),
    labeled as such — they diagnose selector behavior, they are not the
    run's own verdicts. Also reports chunk-length stats for doc110/doc59.
    """
    from jevrag.benchmarks.docbench import (  # noqa: E402
        bm25_retrieve, build_bm25_index, default_k_for_doc)
    from jevrag.pipeline import (  # noqa: E402
        ingest_document, load_memo_cache_jsonl)
    from jevrag.primitives.context_selection import (  # noqa: E402
        DEFAULT_INCLUDE_THRESHOLD, decide_query)

    by_doc: dict[str, list[str]] = {}
    for qid in question_ids:
        doc, _, _ = qid.partition(":")
        by_doc.setdefault(doc, []).append(qid)
    doc_meta = {d[0]: d for d in DOCS}
    probe_out: dict[str, Any] = {"probes": [], "chunk_stats": {}}
    for doc_id, qids in by_doc.items():
        qa_dir, pdf_name, qa_name = doc_meta[doc_id][1:4]
        pages = extract_pdf_pages(KAGGLE / qa_dir / pdf_name)
        questions = {q["id"]: q for q in load_docbench_questions(
            doc_id, KAGGLE / qa_dir / qa_name, types=QUESTION_TYPES)}
        cache_path = (f"{chunk_cache_dir}/{CHUNK_CACHE_PREFIX}_{doc_id}.jsonl"
                      if chunk_cache_dir else "")
        cache = (load_memo_cache_jsonl(cache_path) if cache_path else {})
        chunks, ing = ingest_document(
            pages, doc_id=doc_id, chunk_decision=backends["chunk"],
            max_chunk_chars=9000, repeats=3,
            chunk_cache=cache, memoize_chunk=True)
        print(f"probe {doc_id}: re-ingest jev_calls={ing['jev_calls']} "
              f"hits={ing.get('n_cache_hits')} chunks={len(chunks)}",
              flush=True)
        probe_out["chunk_stats"][doc_id] = chunk_length_stats(chunks)
        index = build_bm25_index(chunks)
        k = default_k_for_doc(len(chunks))
        pool_n = min(max(k * 2, 1), 20)
        for qid in qids:
            q = questions[qid]
            pool = bm25_retrieve(
                q["question"], index, chunks, top_k=pool_n)["chunks"]
            cands = [{"query": q["question"], "query_id": "",
                      "passage": d.get("text", ""),
                      "passage_id": str(d.get("title", "")),
                      "passage_title": str(d.get("title", ""))}
                     for d in pool]
            recs, _ = decide_query(cands, backends["selection"],
                                   threshold=DEFAULT_INCLUDE_THRESHOLD)
            ordered = sorted(
                ({"passage_id": r["passage_id"],
                  "confidence": r["confidence"],
                  "selected": r["selected"],
                  "text": next(d.get("text", "") for d in pool
                               if str(d.get("title", "")) == r["passage_id"])}
                 for r in recs),
                key=lambda r: r["confidence"], reverse=True)
            print(f"probe {qid}: pool={len(pool)} "
                  f"selected={sum(1 for r in ordered if r['selected'])} "
                  f"top_conf={[round(r['confidence'], 3) for r in ordered[:5]]}",
                  flush=True)
            probe_out["probes"].append({
                "question_id": qid, "question": q["question"],
                "gold": q.get("answer", ""), "pool_n": pool_n,
                "fresh_probe": True, "verdicts": ordered})
    return probe_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=[d[0] for d in DOCS],
                    help="subset of doc ids to run")
    ap.add_argument("--generator", choices=("openrouter", "extractive"),
                    default="openrouter",
                    help="openrouter: 4o-mini via OpenRouter, fail-closed on "
                    "the first generator error (no silent fallback); "
                    "extractive: explicit zero-cost fallback for offline "
                    "testing only")
    ap.add_argument("--chunk-cache-dir", default="outputs",
                    help="persisted per-doc chunk caches live here as "
                    "v03_a3_chunkcache_<doc>.jsonl; empty string disables")
    ap.add_argument("--probe", nargs="*", default=[],
                    help="question ids for selector diagnosis "
                    "(e.g. doc110:3 doc218:0); runs instead of the pilot")
    ap.add_argument("--probe-out",
                    default="outputs/v03_a3_probe.json")
    ap.add_argument("--out-prefix", default=OUT_PREFIX)
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing to existing non-empty outputs "
                    "(refused by default — the A2 lesson)")
    args = ap.parse_args()
    print(f"license: {LICENSE_NOTE}", flush=True)

    from openai import OpenAI
    from scripts.produce_records import make_answer_fn

    or_client = OpenAI(base_url=OPENROUTER_BASE_URL,
                       api_key=load_openrouter_key())
    template = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    if args.generator == "openrouter":
        generator_tag = f"openrouter:{GENERATOR_MODEL}"
        answer_fn = fail_closed_answer_fn(
            make_answer_fn(or_client, GENERATOR_MODEL, template),
            generator_tag)
    else:
        generator_tag = "extractive:fallback-explicit"
        answer_fn = None
    print(f"generator: {generator_tag}", flush=True)
    jev_key = load_jev_key()
    backends = {j: JevDecision(api_key=jev_key)
                for j in ("chunk", "selection", "sufficiency", "abstain",
                          "cache")}

    if args.probe:
        paths = checked_out_paths(
            args.probe_out[:-len(".json")] if args.probe_out.endswith(".json")
            else args.probe_out, args.overwrite)
        probe_out = run_probe(args.probe, backends, args.chunk_cache_dir)
        with open(paths["rows"], "w", encoding="utf-8") as f:
            json.dump(probe_out, f, indent=2)
        print(f"wrote {paths['rows']}", flush=True)
        return 0

    paths = checked_out_paths(args.out_prefix, args.overwrite)
    out_path, trace_path = paths["rows"], paths["trace"]

    all_rows: list[dict] = []
    doc_traces: list[dict] = []
    t0 = time.perf_counter()
    try:
        for doc_id, qa_dir, pdf_name, qa_name, seed_qid in DOCS:
            if doc_id not in args.docs:
                continue
            cache_path = (
                f"{args.chunk_cache_dir}/{CHUNK_CACHE_PREFIX}_{doc_id}.jsonl"
                if args.chunk_cache_dir else "")
            rows, dt = run_doc(doc_id, qa_dir, pdf_name, qa_name, seed_qid,
                               backends, answer_fn, generator_tag,
                               cache_path)
            all_rows.extend(rows)
            doc_traces.append(dt)
    except GeneratorHalted as exc:
        # Fail closed: partial rows are still written (with a flag) so the
        # spend is auditable, but the exit is non-zero and no fallback ran.
        partial = REPO_ROOT / f"{args.out_prefix}.partial.jsonl"
        with open(partial, "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nHALTED: {exc}\nwrote {len(all_rows)} partial rows to "
              f"{partial}; no fallback ran, no totals written.", flush=True)
        return 1

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok_rows = [r for r in all_rows if r["outcome"] != "error"]
    pipe_rows = [r for r in ok_rows if r.get("source") == "pipeline"]
    ing_calls = sum(d["ingestion"]["jev_calls"] for d in doc_traces)
    ing_toks = sum(d["ingestion"]["jev_input_tokens"] for d in doc_traces)
    q_calls = sum(int(r.get("jev_calls", 0)) for r in pipe_rows)
    q_toks = sum(int(r.get("jev_input_tokens", 0)) for r in pipe_rows)
    # Cache-trust lookups are real Jev calls with real tokens (A2 counted the
    # call but dropped the tokens — the one-token/one-call gap). Own line.
    ct_rows = [r for r in ok_rows if r.get("cache", {}).get("hit")]
    ct_calls = len(ct_rows)
    ct_toks = sum(int(r["cache"].get("input_tokens") or 0) for r in ct_rows)
    sel_live = sum(int(r.get("selection_calls", 0)) for r in pipe_rows)
    sel_hits = sum(int((r.get("selection") or {}).get("cache_hits", 0))
                   for r in pipe_rows)
    total_toks = ing_toks + q_toks + ct_toks
    answered = [r for r in ok_rows if r["outcome"] == "answer"]
    trace_doc = {
        "packet": "V03-A3", "mode": "default",
        "generator": generator_tag,
        "judge": None,
        "docs": doc_traces,
        "n_questions": len(all_rows),
        "n_answered": len(answered),
        "em_answered": sum(1 for r in answered if r.get("em")) / len(answered)
        if answered else 0.0,
        "em_overall": sum(1 for r in ok_rows if r.get("em")) / len(ok_rows)
        if ok_rows else 0.0,
        "f1_mean_answered": sum(float(r.get("f1", 0)) for r in answered)
        / len(answered) if answered else 0.0,
        "f1_mean_overall": sum(float(r.get("f1", 0)) for r in ok_rows)
        / len(ok_rows) if ok_rows else 0.0,
        "jev_calls_ingestion": ing_calls,
        "jev_calls_questions": q_calls,
        "jev_calls_cache_trust": ct_calls,
        "jev_calls_selection_live": sel_live,
        "jev_calls_selection_hits": sel_hits,
        "jev_calls_total": ing_calls + q_calls + ct_calls,
        "jev_input_tokens_ingestion": ing_toks,
        "jev_input_tokens_questions": q_toks,
        "jev_input_tokens_cache_trust": ct_toks,
        "jev_input_tokens": total_toks,
        "jev_cost_usd": estimate_cost_usd(total_toks),
        "outcomes_path": str(out_path),
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_doc, f, indent=2)
    print(f"\nTOTAL: {len(all_rows)} q, {len(answered)} answered; "
          f"Jev {trace_doc['jev_calls_total']} calls "
          f"(ing {ing_calls} + q {q_calls} + cache-trust {ct_calls}), "
          f"{total_toks} tokens "
          f"(est. ${estimate_cost_usd(total_toks):.4f}); "
          f"wall {(time.perf_counter() - t0):.0f}s; wrote {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
