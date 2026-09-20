"""sha256 manifest of every source file — the acceptance-test proof.

Every new primitive raises the same question: did adding it require touching
``decision.py``, the calibration harness, the cost module, or any
controller-level code? The answer has to be mechanism, not assertion — so
snapshot before the first live call and re-check after:

    py -3.13 scripts/snapshot_protected_hashes.py --save outputs/hashes_before.json
    py -3.13 scripts/snapshot_protected_hashes.py --check outputs/hashes_before.json

The manifest covers every tracked ``.py``/``.md``/``.toml`` file under
``jevrag/``, ``tests/``, ``scripts/``, ``docs/prd/`` plus the root contract
docs — so "nothing else changed" is checkable across the whole repo, not just
for a hand-picked list. Working notes and run artifacts (``records/``,
``outputs/``, ``reports/``) are excluded since they're expected to change.

Files a given run expects to be NEW are excluded by name: they do not
exist in a legitimately-unchanged baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: New files from the context-selection primitive's own build — expected to
#: exist now, so they are not part of the "pre-existing code unchanged" claim.
NEW_FILES = {
    "jevrag/primitives/context_selection.py",
    "tests/test_context_selection.py",
    "scripts/eval_context_selection.py",
}

SCAN_ROOTS = ("jevrag", "tests", "scripts", "docs/prd")
ROOT_FILES = ("INTERFACE.md", "README.md", "pyproject.toml", "TEAM_BRIEF.md")
SUFFIXES = (".py", ".md", ".toml")


def manifest(root: Path = REPO_ROOT) -> dict[str, str]:
    """repo-relative path -> sha256, sorted, for every tracked source file."""
    rel: set[str] = set()
    for scan in SCAN_ROOTS:
        base = root / scan
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in SUFFIXES:
                    rel.add(p.relative_to(root).as_posix())
    for name in ROOT_FILES:
        if (root / name).is_file():
            rel.add(name)
    out: dict[str, str] = {}
    for r in sorted(rel - NEW_FILES):
        out[r] = hashlib.sha256((root / r).read_bytes()).hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save", type=Path, help="write the manifest here")
    mode.add_argument("--check", type=Path, help="compare against this manifest")
    args = ap.parse_args()

    current = manifest()
    if args.save:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "root": str(REPO_ROOT),
            "n_files": len(current),
            "hashes": current,
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8")
        print(f"saved {len(current)} file hashes to {args.save}")
        return 0

    before = json.loads(args.check.read_text(encoding="utf-8"))["hashes"]
    changed = sorted(k for k in before if k in current
                     and before[k] != current[k])
    missing = sorted(k for k in before if k not in current)
    added = sorted(k for k in current if k not in before)
    print(f"baseline: {len(before)} files ({args.check})")
    print(f"now:      {len(current)} files")
    print(f"  changed: {changed or 'none'}")
    print(f"  removed: {missing or 'none'}")
    print(f"  added:   {added or 'none'}")
    if changed or missing:
        print("\nFAIL — pre-existing files changed; the acceptance claim does "
              "not hold as stated.")
        return 1
    print("\nOK — every pre-existing file is byte-identical (sha256).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
