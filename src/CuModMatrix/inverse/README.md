# CuModMatrix Inverse Algorithms

This directory contains GPU implementations for PLUQ factorization and matrix
inversion over finite fields.

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
- Square inverse: uses a GPU Gauss-Jordan path over an augmented matrix `[A I]`;
  `inverse_strategy = :pluq` can instead route through the PLUQ factorization and
  triangular inverses.
- Rectangular one-sided inverses: use rank-revealing row and column pivoting over
  augmented systems to construct right inverses for full row-rank matrices, with
  left inverses handled through transposition.
- Tiny batched PLUQ and inverse kernels: specialize the ICCS-style workload of
  many small matrices for fixed square sizes 4, 8, 16, and 32.
