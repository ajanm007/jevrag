# JevRAG — Decision Path ↔ Measurement Path Interface

The decision path (retrieval + Jev + generation) and the measurement path
(the `jevrag eval` CLI) communicate through JSONL record files, never
directly. The measurement path never calls Jev or a retriever itself —
this separation is deliberate, so the same evaluation command works
identically on a run from five minutes ago or five days ago.

## 1. The record shape (sufficiency)

The decision path emits one JSON object per question (JSONL — one record
per line):

```json
{
  "question_id": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "gold": "yes",
  "prediction": "yes",
  "confidence": 0.93,
  "rounds_used": 2,
  "latency_ms": 1180.4,
  "input_tokens": 6412,
  "type": "comparison"
}
```

Field semantics:

| field | meaning | notes |
|---|---|---|
| `question_id` | HotpotQA `id` | joined against the dataset to recover the frozen val/test `split`. Records whose id isn't in the dataset are rejected loudly. |
| `gold` | gold answer string | dataset field is `answer`; the decision path copies it into `gold` |
| `prediction` | final generated answer | empty string if the policy abstains |
| `confidence` | Jev's Boolean probability **at the stopping point** — P(evidence sufficient) | must be in [0, 1]. This is the gate signal; it goes straight into the calibration harness. `null` only for records from a gate-free policy (e.g. the fixed-iteration baseline) |
| `rounds_used` | retrieval rounds actually run | ≥ 1 |
| `latency_ms` | wall-clock for the whole question — retrieval + Jev calls + generation | kept as the sum if per-call latency is logged separately elsewhere |
| `input_tokens` | total input tokens across all calls for the question (Jev + generator) | cost table comes from this; Jev output tokens are free |
| `type` | `bridge` \| `comparison` | from the dataset, for the free per-type breakdown |

Optional extra fields are fine — the measurement path ignores what it
doesn't know — but include `policy` (see §3) from the start.

Each other primitive (chunk-boundary, context-selection, answer-abstain,
cache-trust) has its own record shape, documented in that primitive's
module docstring and read directly by its own `jevrag eval <primitive>`
report function in `jevrag/__main__.py`.

## 2. Files on disk

Convention: `records/<policy>_<dataset>.jsonl` relative to repo root,
e.g. `records/sufficiency_hotpotqa.jsonl`,
`records/fixed_iter3_hotpotqa.jsonl`. The CLI accepts explicit paths
(`--records`, `--baseline-records`), so this is a convention, not a
discovery mechanism.

## 3. Baseline records — same machinery, different stopping rule

`fixed_iteration` records must come from the same retrieval and
generation stack as the gated run, with only the stopping policy
changed (gate disabled, always exactly N rounds). If the baseline uses a
different retriever or generator, the comparison measures the wrong
thing — the CLI's generator-parity guard refuses to report over a
mismatch rather than produce a meaningless delta.

Baseline records have `confidence: null` and `policy: "fixed_iteration_N"`.
Gated records have `policy: "sufficiency_jev"`.
