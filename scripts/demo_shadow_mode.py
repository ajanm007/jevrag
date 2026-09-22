"""Demo: shadow/observe mode — a challenger alongside production, zero live spend.

Replays one primitive's existing records (default: answer-abstain's 4o-mini
val set, n=30) through a ShadowDecision-wrapped LogprobDecision challenger
while production — the recorded Jev decisions in the file — is left
untouched (the input file is opened read-only; nothing is written back to
it). Each replayed decision point appends one row to the ledger; the
companion script reports on it:

    py scripts/demo_shadow_mode.py
    py scripts/report_shadow_ledger.py --ledger outputs/shadow_logprob_demo.jsonl \\
        --records outputs/answer_abstain_4omini_val.jsonl

No API keys, no network: both backends involved (the recorded Jev numbers
and the live LogprobDecision challenger) cost nothing at demo time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevrag.backends.logprob_decision import LogprobDecision  # noqa: E402
from jevrag.backends.shadow import ShadowDecision  # noqa: E402
from jevrag.primitives.answer_abstain import grounding_question  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records",
                    default="outputs/answer_abstain_4omini_val.jsonl",
                    help="existing records file to replay (read-only)")
    ap.add_argument("--ledger", default="outputs/shadow_logprob_demo.jsonl",
                    help="ledger file to write (truncated unless --append)")
    ap.add_argument("--append", action="store_true",
                    help="append to an existing ledger instead of truncating")
    ap.add_argument("--limit", type=int, default=None,
                    help="replay only the first N rows")
    args = ap.parse_args()

    records_path = REPO_ROOT / args.records
    with open(records_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        print(f"error: no records in {records_path}", file=sys.stderr)
        return 2

    ledger_path = REPO_ROOT / args.ledger
    if ledger_path.exists() and not args.append:
        print(f"note: truncating existing demo ledger {ledger_path}")
        ledger_path.unlink()

    challenger = ShadowDecision(LogprobDecision(), ledger_path)
    question = grounding_question()
    for i, r in enumerate(rows, 1):
        state = {
            "question": r["question"],
            "prediction": r["prediction"],
            "mean_logprob": r["mean_logprob"],
            "n_logprob_tokens": r["n_logprob_tokens"],
            "generator": r.get("generator", "unknown"),
        }
        res = challenger.ask(state, [question])
        print(f"[{i}/{len(rows)}] challenger_conf={float(res.confidence):.4f} "
              f"production_conf={r['confidence']:.2f} "
              f"production_action={r['action']}",
              flush=True)

    print(f"\nreplayed {len(rows)} decision points from {records_path} "
          "(file untouched)")
    print(f"ledger: {ledger_path} "
          f"({challenger.logged} rows, {challenger.dropped} dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
