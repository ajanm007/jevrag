"""Clean-room Kaggle-bundle tests (packet V03-A8 §§2-3). Offline only.

Copies the exact runtime closure of scripts/run_suite.py (table in the
report) plus jevrag/ and pyproject.toml into a temp dir — no .env, no
RAG_GATE_PATH, no JEVRAG_KAGGLE_PATH, scrubbed PYTHONPATH — and proves:
(1) `run_suite.py --help` exits 0 there (import-time safety);
(2) a stub end-to-end (A6 fixture records+pool, --limit 2) proves key
loading from environment variables, prompt-template lookup through a temp
RAG_GATE_PATH, and result writing, with ZERO network calls (socket.socket
raises in the subprocess bootstrap).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Exact runtime closure of scripts/run_suite.py beyond the jevrag/ package
#: (transitively; keep in sync — the --help test below enforces it).
CLOSURE = [
    "scripts/run_suite.py",
    "scripts/run_pipeline_docbench_pilot.py",
    "scripts/_load_run_keys.py",
    "scripts/produce_records.py",
]


def stage_clean_room(tmp_path: Path) -> Path:
    """Build the scrubbed layout; return its root."""
    room = tmp_path / "cleanroom"
    shutil.copytree(REPO_ROOT / "jevrag", room / "jevrag",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO_ROOT / "pyproject.toml", room / "pyproject.toml")
    for rel in CLOSURE:
        dest = room / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, dest)
    # Fixture inputs (A6 smoke records + pool), header pool_file rewritten.
    recs = [json.loads(l) for l in
            open(REPO_ROOT / "outputs" / "v03_a6_hotpotqa_records.jsonl",
                 encoding="utf-8") if l.strip()]
    header, rows = recs[0]["__header__"], recs[1:]
    # The fixture header's pool_file is a machine-specific absolute path
    # (written on WSL). Resolve the pool from this checkout's outputs/ by
    # basename instead, so the test is portable; skip clearly if absent.
    pool_src = REPO_ROOT / "outputs" / Path(header["pool_file"]).name
    if not pool_src.exists():
        pytest.skip(f"fixture pool not present: {pool_src.name}")
    pool_dst = room / "pool.jsonl"
    shutil.copy(pool_src, pool_dst)
    header = dict(header, pool_file=str(pool_dst))
    with open(room / "records.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"__header__": header}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rgdata = room / "rgdata" / "data"
    rgdata.mkdir(parents=True)
    (rgdata / "prompt_template.txt").write_text(
        "Q: {question}\nC: {context}\n", encoding="utf-8")
    return room


def scrubbed_env(room: Path) -> dict:
    """Environment with secrets/paths controlled and nothing leaking in."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(room)
    for var in ("RAG_GATE_PATH", "JEVRAG_KAGGLE_PATH", "JEV_API_KEY",
                "OPENROUTER_API_KEY"):
        env.pop(var, None)
    return env


def test_clean_room_help(tmp_path):
    room = stage_clean_room(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(room / "scripts" / "run_suite.py"), "--help"],
        capture_output=True, text=True, cwd=str(room),
        env=scrubbed_env(room), timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "usage:" in proc.stdout


def test_clean_room_stub_run_zero_network(tmp_path):
    room = stage_clean_room(tmp_path)
    driver = (
        "import socket as _s, sys\n"
        "_real_socket = _s.socket\n"
        "class _Blocked(_real_socket):\n"
        "    def __init__(self, *a, **k):\n"
        "        raise RuntimeError('network blocked by clean-room test')\n"
        "_s.socket = _Blocked\n"
        "import ssl as _ssl\n"  # must still import (subclasses socket)
        f"sys.path.insert(0, {str(room)!r})\n"
        "import scripts.run_suite as rs\n"
        "import scripts.produce_records as pr\n"
        "from jevrag.decision import StubDecision\n"
        "rs.JevDecision = lambda **kw: StubDecision(confidences={\n"
        "    'sufficient': 0.95, 'grounded': 0.8, 'relevant': 0.9,\n"
        "    'split': 0.1, 'serve_from_cache': 0.1,\n"
        "    'staleness_risk': 0.1})\n"
        "pr.make_answer_fn = lambda client, model, template: (\n"
        "    lambda q, ev: ('stub answer', {'input_tokens': 3,\n"
        "                                  'output_tokens': 2}))\n"
        "import sys as _s\n"
        f"_s.argv = ['run_suite.py', '--records', {str(room / 'records.jsonl')!r},\n"
        f"           '--out-prefix', {str(room / 'out')!r},\n"
        "           '--limit', '2', '--yes', '--allow-degenerate']\n"
        "rc = rs.main()\n"
        "print('MAIN_RC', rc)\n"
        "assert rc == 0, rc\n"
    )
    env = scrubbed_env(room)
    env["JEV_API_KEY"] = "stub-jev-from-env"
    env["OPENROUTER_API_KEY"] = "stub-or-from-env"
    env["RAG_GATE_PATH"] = str(room / "rgdata")
    proc = subprocess.run(
        [sys.executable, "-"], input=driver,
        capture_output=True, text=True, cwd=str(room),
        env=env, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "MAIN_RC 0" in proc.stdout
    out_lines = [l for l in open(room / "out.jsonl", encoding="utf-8")
                 if l.strip()]
    header = json.loads(out_lines[0])
    assert "__header__" in header  # config header written first
    rows = [json.loads(l) for l in out_lines[1:]]
    # --limit 2, no CRC on checkpoints: 2 cal x2 arms + 2 rep x2 arms.
    assert len(rows) == 8, len(rows)
    assert {r["arm"] for r in rows} == {"gated", "fixed"}
    # Keys demonstrably came from the environment (no .env exists here).
    assert not (room / ".env").exists()
