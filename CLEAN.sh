#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

# Local review/agent artifacts. Keep source, tests, REVIEW_RESPONSES.md, and the
# benchmark technical specification tracked for reviewer and maintainer use.
rm -f -- \
  PR.md \
  plan.md \
  pr-20-comments.json \
  src/CuModMatrix/inverse/HPDC.pdf \
  src/CuModMatrix/inverse/ICCS.pdf

printf '%s\n' 'Removed local review artifacts, downloaded comments, and paper copies.'
