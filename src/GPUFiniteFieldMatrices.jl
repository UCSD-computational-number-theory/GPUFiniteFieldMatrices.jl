module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using SparseArrays
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles
using Unroll

const DEBUG = false

include("CuModMatrix/CuModMatrix.jl")

include("CuModMatrix/kernel_mul/mat_mul_gpu_direct.jl")
include("CuModMatrix/kernel_mul/mat_mul_ops.jl")
include("CuModMatrix/kernel_mul/stripe_mul.jl")
include("KaratsubaMatrices.jl")

include("CuModMatrix/rref_lu_pluq/permutations.jl")
include("CuModMatrix/rref_lu_pluq/pluq_kernels.jl")

include("CuModMatrix/triangular/triangular_inverse_no_copy.jl")
include("CuModMatrix/triangular/substitution_inplace.jl")
include("CuModMatrix/inverse/types.jl")
include("CuModMatrix/inverse/mod_arith.jl")
include("CuModMatrix/inverse/perm_vectors.jl")
include("CuModMatrix/inverse/basecase_pluq.jl")
include("CuModMatrix/inverse/rectangular_pluq.jl")
include("CuModMatrix/inverse/trsm.jl")
include("CuModMatrix/inverse/schur_update.jl")
include("CuModMatrix/inverse/blocked_recursive_pluq.jl")
include("CuModMatrix/inverse/extract.jl")
include("CuModMatrix/inverse/validation.jl")
include("CuModMatrix/inverse/api.jl")
include("CuModMatrix/inverse/batched_tiny.jl")
include("KernelAbstractions/KernelAbstractions.jl")

# Export the main type and its operations
export CuModArray, CuModMatrix, CuModVector
export inverse

export KaratsubaArray, KaratsubaMatrix, KaratsubaVector

# Export utility functions
export eye 
export change_modulus, change_modulus_no_alloc!
export elementwise_multiply!, negate!
export scalar_add!, scalar_sub!, rmul!, lmul!
export mod_elements!, fill! 

# do not export: add!, sub!, zero!, is_invertible, is_invertible_with_inverse
#     (since they conflict with AbstractAlgebra

# Export GPU operations
export mat_mul_gpu_type, mat_mul_type_inplace!
export perm_array_to_matrix
export is_invertible, inverse, is_invertible_with_inverse
export perm_array_to_matrix
export apply_col_perm!, apply_row_perm!
export mod_inv
export pluq_gpu_kernel
export upper_triangular_inverse_no_copy, lower_triangular_inverse_no_copy
export forward_sub_gpu_type_32, backward_sub_gpu_type_32
export PLUQOptions, PLUQFactorization
export pluq_new, pluq_new!, inverse_new, inverse_pluq_new, is_invertible_new
export pluq_new_batch, inverse_new_batch
export pluq_batched_4x4!, pluq_batched_8x8!, pluq_batched_16x16!, pluq_batched_32x32!
export inverse_batched_4x4!, inverse_batched_8x8!, inverse_batched_16x16!, inverse_batched_32x32!
export right_inverse_new, left_inverse_new
export pluq_extract_L, pluq_extract_U, pluq_check_identity
export PLUQOptionsKA
export pluq_new_ka, pluq_new_ka!, inverse_new_ka, inverse_pluq_new_ka, is_invertible_new_ka
export right_inverse_new_ka, left_inverse_new_ka
export pluq_new_batch_ka, inverse_new_batch_ka
export pluq_batched_4x4_ka!, pluq_batched_8x8_ka!, pluq_batched_16x16_ka!, pluq_batched_32x32_ka!
export inverse_batched_4x4_ka!, inverse_batched_8x8_ka!, inverse_batched_16x16_ka!, inverse_batched_32x32_ka!
export pluq_extract_L_ka, pluq_extract_U_ka, pluq_check_identity_ka
export add_ka!, sub_ka!, mul_ka!

end
