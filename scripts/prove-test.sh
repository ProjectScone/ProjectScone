#!/bin/zsh
# Prove a test can fail.
#
# A green test is not evidence until it has gone red for the right
# reason. This breaks the code the test claims to cover, runs the test,
# and requires it to fail; then it restores the file and requires the
# test to pass again. Anything else means the test is measuring nothing.
#
#   prove-test.sh <file> <needle> <replacement> <test runner args...>
#
# The runner defaults to `cargo test`; set PROVE_RUNNER to prove a test
# in another stack, e.g. PROVE_RUNNER="pytest -q" for python/scone-memory.
#
# Example:
#   prove-test.sh crates/scone-core/src/recall.rs \
#     'relative_window(query, &anchor)' 'None' \
#     -p scone-core --test relative_recall
set -o pipefail
RUNNER=(${=PROVE_RUNNER:-cargo test})
# Python caches bytecode by source mtime and size. A same-length mutation
# inside one second would leave the "restored" run executing the broken
# build, so the proof never writes bytecode.
export PYTHONDONTWRITEBYTECODE=1
FILE=$1; NEEDLE=$2; REPLACEMENT=$3; shift 3
[[ -f $FILE ]] || { print "no such file: $FILE"; exit 2 }

BACKUP=$(mktemp)
cp "$FILE" "$BACKUP"
restore() { cp "$BACKUP" "$FILE"; rm -f "$BACKUP" }
trap restore EXIT INT TERM

print "1. green before?"
"${RUNNER[@]}" "$@" >/dev/null 2>&1 || { print "   FAIL: the test does not pass to begin with"; exit 1 }
print "   yes"

python3 - "$FILE" "$NEEDLE" "$REPLACEMENT" <<'PY'
import sys, pathlib
path, needle, replacement = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path); s = p.read_text()
n = s.count(needle)
if n != 1:
    print(f"   FAIL: needle appears {n} times, need exactly 1"); sys.exit(1)
p.write_text(s.replace(needle, replacement))
PY
[[ $? -eq 0 ]] || exit 1

print "2. red once broken?"
if "${RUNNER[@]}" "$@" >/dev/null 2>&1; then
  print "   FAIL: the test still passes with the behaviour removed."
  print "   It is measuring something else, or nothing."
  exit 1
fi
print "   yes"

restore; trap - EXIT
print "3. green again after restore?"
"${RUNNER[@]}" "$@" >/dev/null 2>&1 || { print "   FAIL: restore left it broken"; exit 1 }
print "   yes"
print "PROVEN: the test fails when the behaviour is removed."
