#!/bin/sh
# Explicit launch only. Parse private configuration as data; never source it.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
python_path=${SCONE_PYTHON:-"$project_dir/packages/memory/.venv/bin/python"}
if [ ! -x "$python_path" ]; then
    printf '%s\n' 'The Python interpreter is missing; prepare packages/memory/.venv or set SCONE_PYTHON.' >&2
    exit 2
fi
if [ "${1:-}" = '--check' ]; then
    shift
    exec "$python_path" "$project_dir/scripts/local_env.py" --env-file "$project_dir/.env.local" --check "$@"
fi
exec "$python_path" "$project_dir/scripts/local_env.py" --env-file "$project_dir/.env.local" -- \
    "$python_path" -m scone_memory.runtime.cli serve "$@"
