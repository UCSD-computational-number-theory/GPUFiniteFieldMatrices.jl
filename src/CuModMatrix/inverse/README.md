# CuModMatrix Inverse Algorithms

This directory contains GPU implementations for PLUQ factorization and matrix
inversion over finite fields.

All algorithms in this directory assume that the modulus `N` is prime. Pass
`PLUQOptions(check_prime=true)` to public entry points to validate that
precondition on the host before GPU kernels launch. The current CUDA kernels pass
`N` as `Int32`, so inverse routines reject moduli larger than `typemax(Int32)`.

## Source Papers

- Jingen Xiang, Huangdong Meng, and Ashraf Aboulnaga, "Scalable Matrix
  Inversion Using MapReduce", HPDC 2014 ([local PDF](HPDC.pdf)).
- Ahmad Abdelfattah, Azzam Haidar, Stanimire Tomov, and Jack Dongarra,
  "Factorization and Inversion of a Million Matrices using GPUs: Challenges and
  Countermeasures", ICCS 2017 ([local PDF](ICCS.pdf)).

## Relationship to the Papers

- **HPDC block LU is the main large-matrix template.** Our recursive PLUQ path
  factors a diagonal panel, computes the off-diagonal blocks with left and
  right triangular solves, applies the Schur update `A22 -= L21*U12`, and
  recurses. It then forms the inverse from triangular inverses and
  permutations. We extend the paper's row-pivoted real LU to row-and-column
  pivoted PLUQ over a finite field. We do not implement its MapReduce/Hadoop
  pipeline, distributed storage, or I/O optimizations.
- **ICCS supplies the tiny-batch workload and fused-augmentation idea.** Our
  4/8/16/32 batch API assigns one matrix to each CUDA block, and the tiny
  inverse keeps `[A I]` in shared memory inside one kernel. The current kernel
  is a correctness baseline: one lane performs the elimination. It does not
  yet implement the paper's cooperative 1D register kernel, warp-shuffle pivot
  search, tunable multiple matrices per block, or delayed row swaps. The
  general augmented inverse also uses `[A I]`, but launches multiple kernels
  per pivot and therefore is not the paper's fused tiny-matrix design.

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
