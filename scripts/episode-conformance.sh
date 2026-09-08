#!/bin/zsh
# Exchange real episode exports between the Rust and Python runtimes, in
# both directions, across every vector backend installed here.
#
# Why this exists. The cross-language tests skip themselves when the Rust
# probe is missing, so running the Python suite on its own reports success
# having checked nothing across the runtime boundary: 55 of 73 cases
# quietly do not run. This builds the probe first, so the cases that
# matter cannot be the ones that were skipped, and then says plainly which
# backends were exercised and which were not.
#
#   scripts/episode-conformance.sh          every backend installed here
#   scripts/episode-conformance.sh --ci     only what CI can run
#
# CI runs the second form: built-in stores plus the probe. The optional
# backends are deselected there because they are not installed, which is
# what "bounded" means in the name. Nothing is deselected for being
# awkward.
set -o pipefail
cd ${0:a:h}/.. || exit 2

BOUND=()
LABEL="every backend installed here"
if [[ ${1:-} == --ci ]]; then
  BOUND=(-k "memory or sqlite or probe")
  LABEL="the bounded set CI runs"
fi

# The project venv when there is one, and whatever python is otherwise:
# this has to run from a worktree, which has no venv of its own, and on a
# machine that installed the package rather than building a venv.
PY=$PWD/python/memory/.venv/bin/python
if [[ ! -x $PY ]]; then
  PY=${SCONE_PYTHON:-$(command -v python3)}
  [[ -n $PY ]] || { print "no python found; set SCONE_PYTHON"; exit 2 }
fi
# Test the tree this script lives in. Without this the project venv's
# editable install wins and a worktree silently checks the main checkout,
# which is how a gate passes on code that was never committed.
export PYTHONPATH=$PWD/python/memory/src${PYTHONPATH:+:$PYTHONPATH}
$PY -c "import scone_memory" 2>/dev/null || {
  print "scone_memory will not import for $PY; install its dependencies or set SCONE_PYTHON"
  exit 2
}
print "using $PY against $PWD/python/memory/src"

# The same profile CI builds: HashEmbedder in both runtimes, so the
# comparison is of the episode contract and not of two embedders.
print "building the Rust probe (no default features)"
cargo build --locked -p scone-core --no-default-features --example episode_roundtrip || exit 1
PROBE=$PWD/target/debug/examples/episode_roundtrip
[[ -f $PROBE ]] || { print "the probe did not appear at $PROBE"; exit 1 }

print "exchanging exports over $LABEL"
OUT=$(cd python/memory && SCONE_TEST_RUST_ROUNDTRIP=$PROBE $PY -m pytest -q -p no:warnings -rs \
  tests/test_cross_language.py $BOUND 2>&1)
STATUS=$?
print $OUT | tail -20

SKIPPED=$(print $OUT | grep -cE '^SKIPPED')
if (( STATUS != 0 )); then
  print "\nCONFORMANCE FAILED"
  exit 1
fi
if (( SKIPPED > 0 )); then
  print "\nPASSED, but $SKIPPED case(s) did not run. Above is why. A backend"
  print "that is not installed here is a gap in this machine's coverage, not"
  print "a fault in the contract; a skipped cross-runtime case means the"
  print "probe was not picked up and this run proved nothing."
else
  print "\nCONFORMANCE OK: nothing skipped"
fi
