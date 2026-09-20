"""jevrag.eval.cost — token and latency accounting for decision-path records.

The cost/latency table in the success bar comes from here. Records carry
per-question ``latency_ms`` (wall clock, whole question) and ``input_tokens``
(total across all calls for that question), per INTERFACE.md.

Jev pricing: $0.042 per million input tokens, output free.
The generator model is priced separately if its price is known; we never invent
a price — unknown prices are reported as token counts only.
"""

from __future__ import annotations

import numpy as np

JEV_USD_PER_MTOK_INPUT = 0.042  # TypeSafe list price, brief §4. Output: free.


def _values(records: list[dict], key: str) -> np.ndarray:
    vals = [float(r[key]) for r in records if r.get(key) is not None]
    return np.asarray(vals, dtype=float)


def _percentiles(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def summarize_cost(records: list[dict]) -> dict:
    """Aggregate cost/latency over decision-path records.

    Missing fields are tolerated (a baseline record may not have input_tokens);
    every aggregate reports how many records it covered so nothing is silently
    computed over a subset.
    """
    latency = _values(records, "latency_ms")
    tokens = _values(records, "input_tokens")
    rounds = _values(records, "rounds_used")

    total_tokens = int(tokens.sum()) if tokens.size else 0
    return {
        "n_records": len(records),
        "n_with_latency": int(latency.size),
        "n_with_tokens": int(tokens.size),
        "latency_ms": _percentiles(latency),
        "input_tokens": {
            "total": total_tokens,
            "per_question": _percentiles(tokens),
        },
        "rounds_used": _percentiles(rounds),
        "jev_cost_usd": {
            "price_per_mtok_input": JEV_USD_PER_MTOK_INPUT,
            "total": total_tokens / 1e6 * JEV_USD_PER_MTOK_INPUT,
            "note": "Jev-side estimate only, if input_tokens covers all calls; "
                    "output tokens are free. Generator cost not priced.",
        },
    }


def format_cost_table(summary: dict) -> str:
    """Render summarize_cost() output as an ASCII table for the CLI."""
    def f(v, nd=2):
        return "—" if v is None else f"{v:.{nd}f}"

    lat, tok, rnd = summary["latency_ms"], summary["input_tokens"]["per_question"], summary["rounds_used"]
    lines = [
        f"  records: {summary['n_records']}  (latency on {summary['n_with_latency']}, tokens on {summary['n_with_tokens']})",
        f"  {'metric':<22}{'mean':>12}{'p50':>12}{'p95':>12}{'max':>12}",
        f"  {'-'*22}{'-'*12}{'-'*12}{'-'*12}{'-'*12}",
        f"  {'latency_ms':<22}{f(lat['mean']):>12}{f(lat['p50']):>12}{f(lat['p95']):>12}{f(lat['max']):>12}",
        f"  {'input_tokens':<22}{f(tok['mean'], 0):>12}{f(tok['p50'], 0):>12}{f(tok['p95'], 0):>12}{f(tok['max'], 0):>12}",
        f"  {'rounds_used':<22}{f(rnd['mean']):>12}{f(rnd['p50']):>12}{f(rnd['p95']):>12}{f(rnd['max'], 0):>12}",
        f"  total input tokens: {summary['input_tokens']['total']:,}",
        f"  est. Jev cost: ${summary['jev_cost_usd']['total']:.4f} "
        f"(@ ${summary['jev_cost_usd']['price_per_mtok_input']}/Mtok input, output free)",
    ]
    return "\n".join(lines)
