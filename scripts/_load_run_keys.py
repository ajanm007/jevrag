"""Key loading for live runs.

Rules (packet V03-A8; Kaggle-safe):
- An environment variable of the same name wins when set and non-empty
  (after strip). On Kaggle the notebook sets keys in-process and must NOT
  write a .env (the working directory is saved with notebook output and
  could leak keys).
- Otherwise the repo's own .env file (gitignored, resolved relative to this
  script) is read exactly as before — the fallback behaviour is unchanged.
- Keys are returned, never logged. Nothing in this module prints a key, and
  no exception message or repr contains one.
"""

from __future__ import annotations

import os
from pathlib import Path

JEVRAG_ENV = Path(__file__).resolve().parent.parent / ".env"


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


def _load_key(name: str) -> str:
    """Environment variable first, .env fallback (both stripped)."""
    from_env = os.environ.get(name, "")
    if isinstance(from_env, str) and from_env.strip():
        return from_env.strip()
    return _read_key(JEVRAG_ENV, name)


def load_jev_key() -> str:
    return _load_key("JEV_API_KEY")


def load_openrouter_key() -> str:
    return _load_key("OPENROUTER_API_KEY")
