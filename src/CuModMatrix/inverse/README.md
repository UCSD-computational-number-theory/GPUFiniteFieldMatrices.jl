# CuModMatrix Inverse/PLUQ (GPU)

This folder contains the new GPU-first PLUQ and inverse implementation for `CuModMatrix`.

## Quick Start

```julia
using GPUFiniteFieldMatrices

N = 101
A = CuModMatrix([3 5 7; 2 4 9; 1 8 6], N)

F = pluq_new(A)
ok = pluq_check_identity(F, A)

Ainv = inverse_new(A)
is_full_rank = is_invertible_new(A)

# rectangular one-sided inverses
B = CuModMatrix([1 2 3; 0 1 4], N)
Xr = right_inverse_new(B)  # B * Xr = I
```

## File-by-File Map

### `types.jl`

- `PLUQOptions(; blocksize, basecase, pivot_policy, lazy_q, check_prime)`
- `PLUQFactorization`

Example:

```julia
opts = PLUQOptions(blocksize=32, basecase=16)
F = pluq_new(A, options=opts)
```

Option meanings:

- `blocksize`: panel size used by recursive blocked PLUQ.
- `basecase`: recursion stop threshold for tiny-block basecase kernel.
- `pivot_policy`: pivoting policy selector (currently first-nonzero behavior).
- `lazy_q`: intended switch for lazy column-permutation strategy.
- `check_prime`: verifies modulus primality before factoring/inversion.

### `mod_arith.jl`

- `pluq_mod_reduce(x, N)`
- `pluq_mod_add(a, b, N)`
- `pluq_mod_sub(a, b, N)`
- `pluq_mod_mul(a, b, N)`
- `pluq_mod_inv(a, N)`
- `pluq_is_prime(n)`

Example:

```julia
x = pluq_mod_inv(7, 101)
y = pluq_mod_mul(7, x, 101)
```

### `perm_vectors.jl`

- `pluq_init_perm(n)`
- `pluq_inverse_perm(p)`
- `pluq_compose_segment!(perm, offset, locperm)`

Example:

```julia
p = pluq_init_perm(8)
pluq_compose_segment!(p, 3, [2, 1, 3])
pinv = pluq_inverse_perm(p)
```

### `basecase_pluq.jl`

- `_pluq_mod_t(x, N)`
- `_pluq_mod_mul_t(a, b, N)`
- `pluq_find_pivot_kernel!(...)`
- `pluq_swap_rows_kernel!(...)`
- `pluq_swap_cols_kernel!(...)`
- `pluq_scale_column_kernel!(...)`
- `pluq_rank1_update_kernel!(...)`
- `pluq_basecase_gpu!(Adata, N, p, q, k0, kend, n)`

Example:

```julia
Awork = copy(A.data)
p = pluq_init_perm(rows(A))
q = pluq_init_perm(rows(A))
r = pluq_basecase_gpu!(Awork, A.N, p, q, 1, rows(A), rows(A))
```

### `trsm.jl`

- `pluq_trsm_left_kernel!(...)`
- `pluq_trsm_right_kernel!(...)`
- `pluq_trsm_left_lower_unit_gpu!(Adata, N, k0, kend, n)`
- `pluq_trsm_right_upper_gpu!(Adata, N, k0, kend, n)`

Example:

```julia
pluq_trsm_left_lower_unit_gpu!(A.data, A.N, 1, 16, rows(A))
pluq_trsm_right_upper_gpu!(A.data, A.N, 1, 16, rows(A))
```

### `schur_update.jl`

- `pluq_copy_block_kernel!(...)`
- `pluq_write_block_kernel!(...)`
- `pluq_schur_update_gpu!(Adata, N, k0, kend, n)`

Example:

```julia
pluq_schur_update_gpu!(A.data, A.N, 1, 16, rows(A))
```

### `blocked_recursive_pluq.jl`

- `pluq_blocked_recursive_gpu!(Adata, N, opts, p, q, start, stop, n)`
- `pluq_blocked_gpu!(Adata, N, opts, n)`

Example:

```julia
opts = PLUQOptions(blocksize=32)
p, q, rank = pluq_blocked_gpu!(A.data, A.N, opts, rows(A))
```

### `api.jl`

- `pluq_new!(A; options=PLUQOptions())`
- `pluq_new(A; options=PLUQOptions())`
- `is_invertible_new(A; options=PLUQOptions())`
- `pluq_init_aug_kernel!(...)`
- `pluq_aug_find_pivot_kernel!(...)`
- `pluq_aug_scale_row_kernel!(...)`
- `pluq_aug_elim_kernel!(...)`
- `pluq_copy_block_kernel!(...)`
- `inverse_new(A; options=PLUQOptions())`
- `right_inverse_new(A; options=PLUQOptions())`
- `left_inverse_new(A; options=PLUQOptions())`

Example:

```julia
F = pluq_new(A)
Ainv = inverse_new(A)
Xr = right_inverse_new(CuModMatrix([1 2 3; 0 1 4], 101))
```

### `extract.jl`

- `pluq_extract_l_kernel!(...)`
- `pluq_extract_u_kernel!(...)`
- `pluq_apply_paq_kernel!(...)`
- `pluq_extract_L(F)`
- `pluq_extract_U(F)`

Example:

```julia
L = pluq_extract_L(F)
U = pluq_extract_U(F)
```

### `validation.jl`

- `pluq_nonzero_mod_kernel!(...)`
- `pluq_check_identity(F, Aorig)`

Example:

```julia
ok = pluq_check_identity(F, A)
```

`pluq_check_identity` computes whether the factorization satisfies
`P*A*Q == L*U (mod N)` using GPU kernels and `CuModMatrix` operations.

## Notes on GPU Behavior

- The decomposition and inversion paths are GPU-kernel based.
- Existing `CuModMatrix` arithmetic (`mul!`, `sub!`, etc.) is reused to avoid redundant kernel code.
- Validation and extraction are also implemented through GPU buffers and kernels in this folder.

## Internal Augmented-Matrix Kernels

`api.jl` includes kernels that initialize and update augmented systems (`[A|I]`).
These are internal implementation kernels used by:

- `inverse_new` (square Gauss-Jordan path)
- `right_inverse_new` (rectangular one-sided solve path)

They are not separately exposed public APIs; they exist to keep hot loops on GPU.

## Typical Kernel Call Flow

### `inverse_new(A)` for square `n x n`

1. `pluq_init_aug_kernel!` builds `[A | I_n]` in padded GPU memory.
2. For each `k=1:n`:
   - `pluq_aug_find_pivot_kernel!` finds a nonzero pivot row in column `k`.
   - `pluq_swap_rows_kernel!` swaps rows if needed.
   - `pluq_aug_scale_row_kernel!` normalizes pivot row.
   - `pluq_aug_elim_kernel!` eliminates column `k` from all other rows.
3. `pluq_copy_block_kernel!` copies the right block into the inverse output.

### `pluq_new(A)` for square matrices

1. `pluq_blocked_recursive_gpu!` enters recursive blocked factorization.
2. Per recursion node:
   - `pluq_basecase_gpu!` (panel/base block factorization)
   - `pluq_trsm_left_lower_unit_gpu!`
   - `pluq_trsm_right_upper_gpu!`
   - `pluq_schur_update_gpu!`
3. Recurse on trailing Schur complement until `basecase`.

### `pluq_new(A)` for rectangular matrices

1. `pluq_rectangular_rank_gpu!` runs rank-revealing elimination directly:
   - `pluq_find_pivot_rect_kernel!`
   - `pluq_swap_rows_kernel!` / `pluq_swap_cols_kernel!`
   - `pluq_scale_column_rect_kernel!`
   - `pluq_rank1_update_rect_kernel!`
2. Returns `(p, q, rank)` for the rectangular matrix.

## Current Improvement Backlog

The current implementation is correct and modular, but there are clear next
optimizations aligned with HPDC/ICCS guidance:

1. Merge square/rectangular pivot search into a single generic kernel path.
2. Keep `q` permutation lazy deeper into TRSM/Schur paths to reduce column swaps.
3. Replace host scalar pivot reads (`Array(@view ...)`) with tiny device reductions.
4. Batch small panel kernels with CUDA graphs for lower launch overhead.
5. Add device-side permutation composition for `p`/`q` to remove host updates.
6. Add specialized micro-kernels for tiny basecases (8/16/32) with shared-memory staging.
