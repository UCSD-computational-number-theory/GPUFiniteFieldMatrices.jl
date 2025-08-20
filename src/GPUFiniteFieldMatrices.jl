module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles

const DEBUG = false

include("CuModMatrix/gpu_matrix_mod_N/gpu_mat.jl")
include("CuModMatrix/rref_lu_pluq/rref_new_kernels.jl")
include("CuModMatrix/rref_lu_pluq/rref_gpu_type.jl")
include("CuModMatrix/kernel_mul/mat_mul_gpu_direct.jl")
include("CuModMatrix/kernel_mul/mat_mul_ops.jl")
include("CuModMatrix/hensel_lifting/hensel.jl")
include("CuModMatrix/kernel_mul/stripe_mul.jl")
include("KaratsubaMatrices.jl")

# Export the main type and its operations
export CuModArray, CuModMatrix, CuModVector
export inverse, unsafe_CuModMatrix

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
export rref_gpu_type, lu_gpu_type, plup_gpu_type
export mat_mul_gpu_type, mat_mul_type_inplace!
export perm_array_to_matrix
export backward_sub_gpu_type, forward_sub_gpu_type
export is_invertible, inverse, is_invertible_with_inverse
export perm_array_to_matrix

end
