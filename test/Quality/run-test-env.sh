#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
TEST_ENV="$(mktemp -d)"
trap 'rm -rf "$TEST_ENV"' EXIT HUP INT TERM
cp "$ROOT/test/Project.toml" "$TEST_ENV/Project.toml"

JULIA_CMD="${JULIA:-julia}"
GPUFFM_ROOT="$ROOT" "$JULIA_CMD" --startup-file=no --project="$TEST_ENV" \
    -e 'using Pkg; Pkg.develop(path=ENV["GPUFFM_ROOT"]; io=devnull); Pkg.instantiate(; io=devnull)'

if [ "${1:-}" = "--quality-suite" ]; then
    shift
    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 --quality-suite TARGET" >&2
        exit 2
    fi

    target=$1
    "$JULIA_CMD" --project="$TEST_ENV" test/Quality/aqua.jl "$target"
    "$JULIA_CMD" --project="$TEST_ENV" test/Quality/jet.jl "$target"
    "$JULIA_CMD" --project="$TEST_ENV" test/Quality/staticlint.jl "$target"
    "$JULIA_CMD" --project="$TEST_ENV" test/Quality/formatter.jl check "$target"
else
    "$JULIA_CMD" --project="$TEST_ENV" "$@"
fi
