# KernelAbstractions Parity Gates

## Gate Criteria

- `mod_arith_ka` passes for all configured primes.
- `perm_vectors_ka` passes for initialization/inversion/composition.
- `pluq_square_ka` passes on data-driven square matrix grid.
- `inverse_square_ka` passes on data-driven square matrix grid.
- `inverse_rectangular_ka` passes for full-row and full-column rank cases.
- `batch_ka` and `batched_tiny_ka` pass for inverse identity checks.
- `matops_ka` passes for add/sub/mul correctness and identity/distributive checks.
- `padding_and_edgecases_ka` passes for boundary sizes around padding buckets.
- `crosscheck_cuda_vs_ka` passes by validating product identities for KA and CUDA paths.

## Verification Command

```bash
julia --project=. -e 'using Test, GPUFiniteFieldMatrices; include("test/KernelAbstractions/runtests.jl"); fns = [test_mod_arith_ka,test_perm_vectors_ka,test_pluq_square_ka,test_inverse_square_ka,test_inverse_rectangular_ka,test_batch_ka,test_batched_tiny_ka,test_matops_ka,test_padding_and_edgecases_ka,test_crosscheck_cuda_vs_ka]; names=["mod","perm","pluq","invsq","invrect","batch","tiny","matops","padding","cross"]; for (n,f) in zip(names,fns); try f(); println("PASS:",n); catch e; println("FAIL:",n,":",typeof(e)); end; end'
```

## Latest Result

- PASS: mod
- PASS: perm
- PASS: pluq
- PASS: invsq
- PASS: invrect
- PASS: batch
- PASS: tiny
- PASS: matops
- PASS: padding
- PASS: cross
