#!/usr/bin/env python
"""Deterministic quality gate. Cross-platform twin of .claude/hooks/check.sh.

Runs after any edit under src/ or tests/:
  1. Ground-truth leak guard  - only src/sim/ may mention injected_faults or bw_true.
  2. Layering guard           - nothing outside src/ui/ may import src.ui.
  3. Test suite               - pytest -q.

Exit 0 on pass, 1 on any violation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH = re.compile(r"injected_faults|bw_true")
UI_IMPORT = re.compile(r"from\s+src\.ui|import\s+src\.ui")

SKIP_DIRS = {"__pycache__", ".venv", ".git", ".pytest_cache"}


def python_exe() -> str:
    """Prefer the project venv interpreter over whatever is on PATH."""
    for candidate in (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def py_files(rel_dir: str) -> list[Path]:
    base = ROOT / rel_dir
    if not base.is_dir():
        return []
    return [
        p
        for p in base.rglob("*.py")
        if not SKIP_DIRS.intersection(p.relative_to(ROOT).parts)
    ]


def scan(pattern: re.Pattern[str], files: list[Path], exempt_prefix: str) -> list[str]:
    hits: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(exempt_prefix):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        hits.extend(
            f"{rel}:{n}: {line.strip()}"
            for n, line in enumerate(lines, 1)
            if pattern.search(line)
        )
    return hits


def main() -> int:
    failed = False
    src_files = py_files("src")

    leaks = scan(GROUND_TRUTH, src_files, exempt_prefix="src/sim/")
    if leaks:
        print("GROUND TRUTH LEAK - these fields are simulator-only:", file=sys.stderr)
        print("\n".join(leaks), file=sys.stderr)
        failed = True

    ui_deps = scan(UI_IMPORT, src_files, exempt_prefix="src/ui/")
    if ui_deps:
        print(
            "LAYERING VIOLATION - ui must not be imported by other modules:",
            file=sys.stderr,
        )
        print("\n".join(ui_deps), file=sys.stderr)
        failed = True

    if (ROOT / "tests").is_dir():
        result = subprocess.run(
            [python_exe(), "-m", "pytest", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        tail = (result.stdout + result.stderr).strip().splitlines()[-20:]
        print("\n".join(tail))
        # 0 = passed, 5 = nothing collected yet (early scaffolding).
        if result.returncode not in (0, 5):
            failed = True

    if failed:
        print("\nQUALITY GATE FAILED", file=sys.stderr)
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
