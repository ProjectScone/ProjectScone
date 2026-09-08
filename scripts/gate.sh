#!/bin/zsh
# Run everything that must pass, against the committed tree, and say one
# word at the end about whether it did.
#
#   scripts/gate.sh [<commit>]      defaults to HEAD
#
# Three ways a gate has silently checked the wrong thing here, all of
# them fixed below rather than remembered:
#
#   The worktree kept a stale file, so `checkout` refused and every later
#   step ran against older content. It is reset hard, and the commit it
#   ends up on is checked afterwards rather than assumed.
#
#   The project venv holds an editable install rooted at the main
#   checkout, so pytest inside the worktree imported code that was never
#   committed. PYTHONPATH names the worktree first.
#
#   GIT_INDEX_FILE leaking from a private-index commit made `checkout`
#   consult that index, report success, and leave the files untouched.
#   It is cleared here.
#
# Read the last line. Everything above it is detail.
set -o pipefail
unset GIT_INDEX_FILE
cd ${0:a:h}/.. || exit 2

WORK=${SCONE_GATE_WORKTREE:-${TMPDIR:-/tmp}/scone-gate}
WANT=$(git rev-parse ${1:-HEAD}) || exit 2
VENV=$PWD/python/memory/.venv/bin/python
[[ -x $VENV ]] || { print "no interpreter at $VENV"; exit 2 }

if [[ ! -d $WORK ]]; then
  git worktree prune
  git worktree add -q --detach $WORK $WANT || exit 2
fi
git -C $WORK checkout -q --detach $WANT 2>/dev/null
# Hard reset rather than checkout alone: a file left behind in the
# worktree makes checkout refuse, and a refused checkout is how the whole
# gate ends up describing an older tree.
git -C $WORK reset -q --hard $WANT || exit 2
git -C $WORK clean -qfd

GOT=$(git -C $WORK rev-parse HEAD)
DIRT=$(git -C $WORK status --porcelain | wc -l | tr -d ' ')
if [[ $GOT != $WANT || $DIRT != 0 ]]; then
  print "GATE ABORTED: worktree is at ${GOT:0:7} with $DIRT change(s), wanted ${WANT:0:7}"
  exit 1
fi
print "gating ${WANT:0:7} in $WORK, load: $(uptime | sed 's/.*averages: //')"

FAILED=()

print "\n== python =="
(cd $WORK/python/memory && PYTHONPATH=$WORK/python/memory/src $VENV -m pytest -q -p no:warnings 2>&1 | tail -3) || FAILED+=(python)

print "\n== rust =="
(cd $WORK && cargo fmt --all -- --check) || FAILED+=(fmt)
(cd $WORK && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -2) || FAILED+=(clippy)
(cd $WORK && cargo test --workspace 2>&1 | tail -3) || FAILED+=(rust-tests)

print "\n== cross-runtime =="
(cd $WORK && SCONE_PYTHON=$VENV ./scripts/episode-conformance.sh --ci 2>&1 | tail -2) || FAILED+=(conformance)

print "\n== originality =="
$VENV -m scone_memory.testing.originality 2>&1 | tail -1

if (( ${#FAILED} )); then
  print "\nGATE FAILED on ${WANT:0:7}: ${(j:, :)FAILED}"
  exit 1
fi
print "\nGATE PASSED on ${WANT:0:7}"
