# Suite records contract (v0.3)

The interface between the dataset loaders and the suite runner. The loaders
and the writer (`jevrag/benchmarks/closed_pool.py`) produce these files;
`scripts/run_suite.py` consumes them; `scripts/suite_table.py` and
`scripts/shift_test.py` read what the runner writes. The writer-side checks
live in `closed_pool.validate_suite_records`; the runner re-checks what it
needs in `run_suite.load_suite_records` and `run_suite.load_pool`.

## Input: `<dataset>_suite_records.jsonl`

UTF-8, one JSON object per line. Line 1 is a header, every other line is one
question.

### Header line

```json
{"__header__": {
  "dataset": "musique",
  "pool_mode": "per_question",
  "source": {"repo": "...", "file": "...", "bytes": 0, "sha256": "...", "license": "..."},
  "seed": 20261001, "split_seed": 20261002,
  "n": 500, "n_calibrate": 250, "n_report": 250,
  "sample_rule": "...", "strata_counts": {}, "strata_taken": {},
  "metric_definitions": {"paragraph_recall_at_k": "...", "all_gold_found_at_k": "..."}
}}
```

`pool_mode` is `per_question` or `shared`. In `shared` mode the header also
carries `pool_file`, the name of the pool file (see below). The writer refuses
a header that is missing any of these keys, or whose `n`, `n_calibrate` and
`n_report` disagree with the rows.

### Question row

```json
{"id": "2hop__...", "question": "...", "answer": "...", "aliases": ["..."],
 "type": "2hop", "hops": 2, "split": "calibrate",
 "supporting_ids": ["p3", "p7"],
 "paragraphs": [{"id": "p0", "title": "...", "text": "..."}]}
```

- `split` is `calibrate` or `report`. The halves are disjoint and stratified by
  hop count where the dataset has one, else by question type.
- `per_question` mode: `paragraphs` is that question's own pool, with ids
  `p0..pN` scoped to the question, and `supporting_ids` point into it.
- `shared` mode: `paragraphs` is empty on every row, and `supporting_ids`
  point into the shared pool file.
- `aliases` may be empty. Exact match accepts `answer` or any alias; F1 takes
  the best-scoring one.
- `hops` may be `null`. The writer may add `supporting_unresolved`, and the
  HotpotQA adapter adds `level` and `frozen_split` for traceability.
- Never present: the private keys `_paragraphs` and `_supporting_local`, an
  empty `supporting_ids`, or an id that is not in the question's pool. Each is
  a build error, not a warning; the writer raises before anything touches disk.

### Shared pool file: `<dataset>_pool.jsonl`

Rows are `{"id": "p0", "title": "...", "text": "..."}`. **Known mismatch in
v0.3.0:** the writer puts a `{"__header__": {...}}` object on line 1 of this
file, and `run_suite.load_pool` rejects it; the writer also stores the header's
`pool_file` as a bare file name, which the runner resolves against the working
directory. Until both are fixed, strip line 1 and pass the file explicitly with
`--pool-file` (see "Reproducing the v0.3 suite" in the README).

## Pool-mode rule

Each dataset uses the pool mode in which the pipeline has a fair chance: if the
shared pool's `all_gold_found_at_20` (from `closed_pool.pool_sanity_report`) is
below 0.5, the dataset uses `per_question`. For MuSiQue it measured 0.196
shared against 1.000 per-question, so MuSiQue is `per_question`; 2Wiki was
decided the same way and is `per_question` too. HotpotQA stays `shared`, to
match the 9,769-document index every earlier result was measured on. The mode travels in every header, because the same recall
number means something different in each.

## Output of `scripts/run_suite.py`

Given `--out-prefix P`, the runner writes `P.jsonl` and `P.trace.json`.

- **`P.jsonl`**: a header, then one row per (question, arm). The header
  records `dataset`, `pool_mode`, `generator`, `alpha`, `per_round_k`,
  `candidate_pool_cap`, `grid_source`, `oversample`, `rounds`, `records_file`,
  `records_sha256` and `pipeline_sha256`. Each row carries the pipeline's own
  row keys plus `arm`, `split`, `dataset`, `hops`, `em`, `f1`,
  `matched_alias` and a per-stage Jev call and token split under `stage`.
  `gated_crc` rows also carry `crc_spec_sha`.
- **`P.trace.json`**: `abst_thresholds` per arm and `metrics` per
  `arm/split`. It is written only when a run finishes normally, so a halted run
  leaves a stale trace; `suite_table.py` and `shift_test.py` rebuild the
  `gated_crc` threshold from the calibrate rows in that case and say so.
- **Arms**: `gated` and `fixed` on both halves, `gated_crc` on the report half
  only. A `--limit` run without `--with-crc` is a checkpoint and runs `gated`
  and `fixed` only.
- **Resume and overwrite**: an existing non-empty output is refused unless its
  header has the same run identity (dataset, pool mode, generator, alpha, grid,
  oversample, rounds, records hash), in which case the runner appends only the
  missing (question, arm) pairs. Anything else needs `--overwrite`.
- **Failure**: the first generator error, including an HTTP 402 when the
  generator account is out of credit, halts the run. Rows already written are
  kept and nothing falls back silently.
