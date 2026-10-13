"""Key-loading tests (packet V03-A8 §1): env-first, .env fallback unchanged.

Offline: monkeypatched env, tmp .env files. The real repo .env is never
read here (JEVRAG_ENV is repointed), and no value asserted on is real.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts._load_run_keys as rk  # noqa: E402


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    """A tmp .env holding sentinel file-backed values; env cleared."""
    p = tmp_path / ".env"
    p.write_text('JEV_API_KEY="file-jev-key"\n'
                 'OPENROUTER_API_KEY="file-or-key"\n',
                 encoding="utf-8")
    monkeypatch.setattr(rk, "JEVRAG_ENV", p)
    for var in ("JEV_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return p


def test_env_wins_over_dotenv(env_file, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "  env-jev-key  ")
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-or-key")
    assert rk.load_jev_key() == "env-jev-key"  # stripped
    assert rk.load_openrouter_key() == "env-or-key"


def test_empty_env_falls_through_to_dotenv(env_file, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "   ")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    assert rk.load_jev_key() == "file-jev-key"
    assert rk.load_openrouter_key() == "file-or-key"


def test_dotenv_fallback_unchanged(env_file):
    assert rk.load_jev_key() == "file-jev-key"
    assert rk.load_openrouter_key() == "file-or-key"


def test_neither_present_raises_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "JEVRAG_ENV", tmp_path / "no-such.env")
    for var in ("JEV_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(FileNotFoundError, match="cannot read JEV_API_KEY"):
        rk.load_jev_key()
    with pytest.raises(FileNotFoundError, match="cannot read "
                                                "OPENROUTER_API_KEY"):
        rk.load_openrouter_key()


def test_missing_key_in_dotenv_raises_keyerror(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('OPENROUTER_API_KEY="x"\n', encoding="utf-8")
    monkeypatch.setattr(rk, "JEVRAG_ENV", p)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(KeyError, match="JEV_API_KEY"):
        rk.load_jev_key()


def test_no_key_in_messages_or_repr(tmp_path, monkeypatch):
    secret = "super-secret-marker-12345"
    monkeypatch.setattr(rk, "JEVRAG_ENV", tmp_path / "no-such.env")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    try:
        rk.load_jev_key()
    except FileNotFoundError as exc:
        assert secret not in str(exc)
    monkeypatch.setenv("JEV_API_KEY", secret)
    assert secret in rk.load_jev_key()  # returned, that is its job
    assert secret not in repr(rk)
