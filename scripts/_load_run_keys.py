"""Key loading for live runs. Used only by scripts/produce_records.py.

Rules:
- Both keys are read from D:\\JevRAG\\.env (the repo's own env file), which is
  gitignored. It is the single source of truth: rotate there.
- Keys are returned, never logged. Nothing in this module prints a key.

Superseded: an earlier version read OPEN_ROUTER_KEY by path from
D:\\Rag gate crate\\src\\.env. That was reversed — the key now lives here, and
reading the crate copy would silently use a revoked key after a rotation.
"""

from __future__ import annotations

from pathlib import Path

JEVRAG_ENV = Path(r"D:\JevRAG\.env")


def _read_key(env_path: Path, name: str) -> str:
    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} not found — cannot read {name}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            v = v.strip().strip('"').strip("'")
            if not v:
                break
            return v
    raise KeyError(f"{name} not found in {env_path}")


def load_jev_key() -> str:
    return _read_key(JEVRAG_ENV, "JEV_API_KEY")


def load_openrouter_key() -> str:
    return _read_key(JEVRAG_ENV, "OPENROUTER_API_KEY")
