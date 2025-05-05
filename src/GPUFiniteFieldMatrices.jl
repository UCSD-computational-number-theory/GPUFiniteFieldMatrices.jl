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
#include("CuModMatrix/kernel_mul/mat_mul_hybrid.jl")
include("CuModMatrix/rref_lu_pluq/rref_gpu_direct.jl")
include("CuModMatrix/kernel_mul/mat_mul_gpu_direct.jl")
#include("CuModMatrix/kernel_mul/mat_mul_ops.jl")
#include("CuModMatrix/hensel_lifting/hensel.jl")
include("CuModMatrix/kernel_mul/stripe_mul.jl")

# Export the main type and its operations
export CuModArray, CuModMatrix, CuModVector
export is_invertible, inverse, unsafe_CuModMatrix

# Export utility functions
export eye 
export change_modulus, change_modulus_no_alloc!
export add!, sub!, elementwise_multiply!, negate!
export scalar_add!, scalar_subtract!, rmul!, lmul!
export mod_elements!, fill!, zero!

# Export GPU operations
export rref_gpu_type, lu_gpu_type, plup_gpu_type
export mat_mul_gpu_type, mat_mul_type_inplace!

end
