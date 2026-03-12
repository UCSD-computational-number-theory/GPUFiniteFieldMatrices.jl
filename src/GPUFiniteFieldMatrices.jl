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
export pluq_new, pluq_new!, inverse_new, is_invertible_new
export right_inverse_new, left_inverse_new
export pluq_extract_L, pluq_extract_U, pluq_check_identity

end
