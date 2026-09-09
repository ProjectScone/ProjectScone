#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${1:-scone-aws:local}"
platform="${SCONE_BUILD_PLATFORM:-linux/amd64}"
exec docker build --platform "$platform" --file "$repo_root/deploy/aws/Dockerfile" \
  --tag "$image" "$repo_root"
