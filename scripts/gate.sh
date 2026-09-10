#!/bin/zsh
# Validate an isolated committed tree without resetting an existing checkout.
set -eu -o pipefail
cd ${0:a:h}/..
ROOT=$PWD
WANT=$(git rev-parse "${1:-HEAD}^{commit}")
PY=${SCONE_PYTHON:-$ROOT/python/memory/.venv/bin/python}
[[ -x $PY ]] || { print "set SCONE_PYTHON to a test interpreter"; exit 2; }
(cd "$ROOT/python/memory" && PYTHONPATH=$ROOT/python/memory/src "$PY" -m pytest -q tests/test_private_paths.py)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/scone-framework-gate.XXXXXX")
git archive "$WANT" | tar -x -C "$WORK"
print "gating $WANT in $WORK"
(cd "$WORK/python/memory" && PYTHONPATH=$WORK/python/memory/src "$PY" -m pytest -q)
(cd "$WORK/python/scone-client" && PYTHONPATH=$WORK/python/scone-client "$PY" -m pytest -q)
if [[ -n ${SCONE_TEST_RUST_ROUNDTRIP:-} ]]; then
  (cd "$WORK" && SCONE_PYTHON=$PY scripts/episode-conformance.sh --ci)
fi
print "GATE PASSED: $WANT; isolated sources retained at $WORK"
