# PLUQ Rewrite Plan for `CuModMatrix`

## Goal

Implement a new PLUQ decomposition and inverse workflow for `CuModMatrix` from scratch in `src/CuModMatrix/inverse`, combining:

- HPDC block-recursive LU structure
- ICCS very-small-matrix GPU kernel tactics
- finite-field PLUQ requirements over `mod N`

This implementation must ignore the current PLUQ and inverse implementation paths, while reusing existing matrix arithmetic kernels (`mul!`, subtraction/addition utilities) where useful.

## Mathematical Target

For square matrix `A` over `F_p`:

`P * A * Q = L * U`

with:

- `P`, `Q` permutation operators
- `L` unit lower triangular or lower trapezoidal
- `U` upper triangular or upper trapezoidal
- rank-revealing behavior for singular/rank-deficient input

## Architecture

Implement a blocked-recursive factorization:

1. Factor diagonal block `A11` with local PLUQ
2. Propagate local row/column permutations to coupled blocks
3. Compute `U12` with left lower-unit triangular solve
4. Compute `L21` with right upper triangular solve
5. Update Schur complement `A22 -= L21 * U12`
6. Recurse on `A22`

Use:

- permutation vectors, not dense permutation matrices
- packed LU storage in one matrix
- lazy handling of column permutations (`Q`) in hot paths to avoid strided column swaps in column-major GPU memory

## File Layout

### Source files

- `src/CuModMatrix/inverse/types.jl`
- `src/CuModMatrix/inverse/mod_arith.jl`
- `src/CuModMatrix/inverse/perm_vectors.jl`
- `src/CuModMatrix/inverse/basecase_pluq.jl`
- `src/CuModMatrix/inverse/trsm.jl`
- `src/CuModMatrix/inverse/schur_update.jl`
- `src/CuModMatrix/inverse/blocked_recursive_pluq.jl`
- `src/CuModMatrix/inverse/extract.jl`
- `src/CuModMatrix/inverse/api.jl`
- `src/CuModMatrix/inverse/validation.jl`

### Tests

- `test/CuModMatrix/inverse/runtests.jl`
- `test/CuModMatrix/inverse/test_mod_arith.jl`
- `test/CuModMatrix/inverse/test_perm_vectors.jl`
- `test/CuModMatrix/inverse/test_basecase_pluq.jl`
- `test/CuModMatrix/inverse/test_api_smoke.jl`

## Functions To Implement

### Types and options

- `PLUQOptions`
- `PLUQFactorization`

Needed to carry decomposition metadata, block tuning, rank, and permutation vectors through recursive stages.

### Modular arithmetic

- `pluq_mod_reduce(x, N)`
- `pluq_mod_inv(a, N)`
- `pluq_mod_mul(a, b, N)`
- `pluq_mod_add(a, b, N)`
- `pluq_mod_sub(a, b, N)`

Needed for finite-field correctness in all elimination and solve steps.

### Permutation vectors

- `pluq_init_perm(n)`
- `pluq_inverse_perm(p)`
- `pluq_compose_segment!(perm, offset, local)`
- `pluq_apply_rows!(A, swaps, row_lo, row_hi, col_lo, col_hi)`
- `pluq_apply_cols!(A, swaps, row_lo, row_hi, col_lo, col_hi)`

Needed to represent and apply `P`, `Q` efficiently and correctly during recursion and extraction.

### Base case factorization

- `pluq_basecase!(A, p, q, row0, col0, b, opts)`

Needed for tiny blocks where ICCS-style one-block-per-matrix kernels are strongest.

### Triangular solves

- `pluq_trsm_left_lower_unit!(...)`
- `pluq_trsm_right_upper!(...)`

Needed to compute off-diagonal factors from factored diagonal block.

### Schur update

- `pluq_schur_update!(A22, L21, U12, N)`

Needed to recurse on a reduced trailing system and keep cubic work in GEMM.

### Recursive driver

- `pluq_blocked!(A, opts)`
- `pluq_blocked_recursive!(A, p, q, r0, c0, n, opts)`

Needed to realize HPDC recursion and compose global permutations/rank.

### Extraction and validation

- `pluq_extract_L(F)`
- `pluq_extract_U(F)`
- `pluq_check_identity(F, A)`

Needed for correctness tests and downstream usage.

### Public API

- `pluq_new(A; kwargs...)`
- `pluq_new!(A; kwargs...)`
- `inverse_new(A; kwargs...)`
- `is_invertible_new(A; kwargs...)`

Needed for isolated rollout without breaking existing exported behavior.

## Coalesced Access Strategy

- Keep matrix tiles in fast memory in base case.
- Load contiguous column-major segments when possible.
- Prefer lazy `Q` for trailing updates and solves to avoid repeated strided global column swaps.
- Reuse existing GEMM/multiply pathways for Schur updates.
- Defer expensive materialization of explicit permutations/matrices to validation or output extraction.

## Testing Strategy

1. Modular arithmetic identities
2. Permutation composition and inversion
3. Base-case PLUQ identity check
4. End-to-end `P*A*Q == L*U (mod N)` on random and structured matrices
5. Rank-deficient behavior and inverse existence conditions
6. API smoke tests and deterministic behavior under fixed seeds

## Execution Order

1. Build type/options and arithmetic primitives
2. Build permutation utilities
3. Build basecase implementation and tests
4. Add blocked driver shell and API
5. Add recursive solve/update pieces and expand tests
6. Tune thresholds and optimize memory traffic
