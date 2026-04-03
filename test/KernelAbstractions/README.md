# KernelAbstractions Tests

This directory contains the KernelAbstractions (`*_ka`) test suite.

## Current API Functions Covered

- `PLUQOptionsKA`
- `pluq_new_ka`, `pluq_new_ka!`, `is_invertible_new_ka`
- `inverse_new_ka`, `inverse_pluq_new_ka`
- `right_inverse_new_ka`, `left_inverse_new_ka`
- `pluq_new_batch_ka`, `inverse_new_batch_ka`
- `pluq_batched_4x4_ka!`, `pluq_batched_8x8_ka!`, `pluq_batched_16x16_ka!`, `pluq_batched_32x32_ka!`
- `inverse_batched_4x4_ka!`, `inverse_batched_8x8_ka!`, `inverse_batched_16x16_ka!`, `inverse_batched_32x32_ka!`
- `pluq_extract_L_ka`, `pluq_extract_U_ka`, `pluq_check_identity_ka`
- `add_ka!`, `sub_ka!`, `mul_ka!`

## Test Functions

Defined in `runtests.jl`:

- `test_kernel_abstractions_suite()`

Included test groups:

- `test_mod_arith_ka()`
- `test_perm_vectors_ka()`
- `test_pluq_square_ka()`
- `test_inverse_square_ka()`
- `test_inverse_rectangular_ka()`
- `test_batch_ka()`
- `test_batched_tiny_ka()`
- `test_matops_ka()`
- `test_padding_and_edgecases_ka()`
- `test_crosscheck_cuda_vs_ka()`

## How To Run

From repository root:

Run the full package test suite:

```bash
julia --project=. test/runtests.jl
```

Run only KernelAbstractions tests:

```bash
julia --project=. -e 'using Test, GPUFiniteFieldMatrices; include("test/KernelAbstractions/runtests.jl"); test_kernel_abstractions_suite()'
```

Run a single group:

```bash
julia --project=. -e 'using Test, GPUFiniteFieldMatrices; include("test/KernelAbstractions/test_matops_ka.jl"); test_matops_ka()'
```
