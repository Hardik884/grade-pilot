#!/usr/bin/env bash
# Runs after Claude edits anything under src/. Deterministic quality gate.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

fail=0

# Prefer the project venv interpreter over whatever is on PATH.
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python
fi

# 1. Guard: only the simulator may touch ground-truth fields.
leak=$(grep -rn --include=*.py -e 'injected_faults' -e 'bw_true' src/ 2>/dev/null \
       | grep -v '^src/sim/' || true)
if [ -n "$leak" ]; then
  echo "GROUND TRUTH LEAK - these fields are simulator-only:" >&2
  echo "$leak" >&2
  fail=1
fi

# 2. Guard: no module may import from the UI layer.
uidep=$(grep -rn --include=*.py 'from src.ui\|import src.ui' src/ 2>/dev/null \
        | grep -v '^src/ui/' || true)
if [ -n "$uidep" ]; then
  echo "LAYERING VIOLATION - ui must not be imported by other modules:" >&2
  echo "$uidep" >&2
  fail=1
fi

# 3. Tests.
if [ -d tests ]; then
  "$PY" -m pytest -q 2>&1 | tail -20
  rc=${PIPESTATUS[0]}
  # 0 = passed, 5 = nothing collected yet (early scaffolding). Anything else is a failure.
  [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ] || fail=1
fi

exit $fail
