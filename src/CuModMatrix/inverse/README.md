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
