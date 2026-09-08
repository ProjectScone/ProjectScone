#!/bin/sh
# Compatibility entrypoint; use serve-self-hosted.sh for new launchers.
set -eu
scripts_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$scripts_dir/serve-self-hosted.sh" "$@"
