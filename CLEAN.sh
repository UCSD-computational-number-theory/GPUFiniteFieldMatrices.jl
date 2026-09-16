#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

# Local review/agent artifacts. Keep source, tests, REVIEW_RESPONSES.md, and the
# benchmark technical specification tracked for reviewer and maintainer use.
rm -f -- \
  PR.md \
  plan.md \
  pr-20-comments.json

printf '%s\n' 'Removed local review plans and downloaded comments.'
