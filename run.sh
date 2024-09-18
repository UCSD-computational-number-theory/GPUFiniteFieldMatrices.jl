#!/bin/bash

julia_exec=$(which julia)
if [ -z "$julia_exec" ]; then
  echo "Julia executable not found in PATH. Please ensure Julia is installed and available in your PATH."
  exit 1
fi

echo "Using Julia at: $julia_exec"

# $julia_exec --project -e 'using Pkg; Pkg.update()'

julia src/gpu_rref/main.jl

wait
