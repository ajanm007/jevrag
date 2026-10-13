# v0.3 suite results: provenance

These files back the v0.3 rows in the top-level README ("Results at a glance").
They are **generated outputs**, committed so the README's numbers can be
checked without re-running anything paid. The per-question result files they
were generated from are **not** committed: the HotpotQA rows derive from a
private research checkout, and the MuSiQue and 2Wiki rows carry licensed
dataset text. Their identity is recorded below by SHA-256.

## What is here

| file | what it is |
|---|---|
| `suite_tables.md` | `scripts/suite_table.py` over the three result files: per arm and split, n, EM, coverage, both risk readings, AURC, ECE, rounds, Jev dollars |
| `shift_hotpotqa_source.md` | `scripts/shift_test.py` with HotpotQA as the calibration source and MuSiQue and 2Wiki as targets |
| `shift_twowiki_source.md` | the reverse direction: 2Wiki as the source, HotpotQA and MuSiQue as targets |
| `answer_abstain_n300_summary.json` | aggregate metrics of `scripts/eval_answer_abstain_backends.py --n 300` (Jev vs logprob, HotpotQA validation split, first 300 questions in file order); aggregates only, no question text |

## Source result files (not committed)

| file | question/arm rows | SHA-256 |
|---|---|---|
| `hotpotqa_halted_n98.jsonl` | 594 (150 calibrate + 98 report questions) | `6bcb0e0bd11ea712fad50efa08fbf563b83c254bb61a19ed295ff0e31d3df42d` |
| `musique_halted_n111.jsonl` | 633 (150 calibrate + 111 report questions) | `c351f9835609936ab664a7f84a3eb91212362c57bafb3f1788557c875a24da07` |
| `twowiki_n150.jsonl` | 750 (150 calibrate + 150 report questions) | `a22bc524ca8b57ac8e56443ec0812e45340aaebbeb6a44aa7b83f05c25e6b067` |

All three came from `scripts/run_suite.py --with-crc --limit 150` (generator
`openai/gpt-4o-mini` via OpenRouter, CRC alpha 0.3), run on Kaggle in October
2026. The HotpotQA and MuSiQue runs were halted when the generator account ran
out of credit; the runner fails closed, so every row kept is complete, but the
report halves stop at 98 and 111 questions. Their `.trace.json` files predate
the halt (they are checkpoint traces), which is why `suite_tables.md` rebuilds
the CRC threshold from the calibrate rows and says so in its notes column.

## Regenerating

```bash
python scripts/suite_table.py hotpotqa_halted_n98.jsonl musique_halted_n111.jsonl twowiki_n150.jsonl
python scripts/shift_test.py --source hotpotqa_halted_n98.jsonl \
    --targets musique_halted_n111.jsonl twowiki_n150.jsonl
python scripts/shift_test.py --source twowiki_n150.jsonl \
    --targets hotpotqa_halted_n98.jsonl musique_halted_n111.jsonl
```

See the top-level README for what these numbers do and do not show, and
`docs/suite_records_contract.md` for the file formats.
