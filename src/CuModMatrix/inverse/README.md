# CuModMatrix Inverse Algorithms

This directory contains GPU implementations for PLUQ factorization and matrix
inversion over finite fields.

All algorithms in this directory assume that the modulus `N` is prime. Pass
`PLUQOptions(check_prime=true)` to public entry points to validate that
precondition on the host before GPU kernels launch. The current CUDA kernels pass
`N` as `Int32`, so inverse routines reject moduli larger than `typemax(Int32)`.

## Source Papers

- Jingen Xiang, Huangdong Meng, and Ashraf Aboulnaga, "Scalable Matrix
  Inversion Using MapReduce", HPDC 2014.
- Ahmad Abdelfattah, Azzam Haidar, Stanimire Tomov, and Jack Dongarra,
  "Factorization and Inversion of a Million Matrices using GPUs: Challenges and
  Countermeasures", ICCS 2017.

## Algorithms Implemented

- Blocked PLUQ factorization: adapts the HPDC block-recursive LU structure to
  finite-field PLUQ by factoring a panel, applying row and column permutations,
  solving the off-diagonal triangular systems, updating the Schur complement on
  the GPU, and recursing on the trailing block.
- Square inverse: defaults to `inverse_strategy = :pluq`, which routes through
  PLUQ factorization, triangular inverses, modular matrix multiplication, and
  permutation application. `inverse_strategy = :augmented` is kept as a
  reference baseline using GPU Gauss-Jordan elimination over `[A I]`.
- Rectangular one-sided inverses: use rank-revealing row and column pivoting over
  augmented systems to construct right inverses for full row-rank matrices, with
  left inverses handled through transposition. The square `inverse_strategy`
  option does not change these rectangular algorithms.
- Tiny batched PLUQ and inverse kernels: specialize the ICCS-style workload of
  many small matrices for fixed square sizes 4, 8, 16, and 32.
