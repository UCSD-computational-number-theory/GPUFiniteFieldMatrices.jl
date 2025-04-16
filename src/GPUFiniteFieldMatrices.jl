module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles

include("matrix_mod_N/gpu_rref/rref_new_kernels.jl")
include("matrix_mod_N/gpu_mat_mul/mat_mul_hybrid.jl")
include("matrix_mod_N/gpu_mat_type/gpu_mat.jl")
include("matrix_mod_N/gpu_rref/rref_gpu_direct.jl")
include("matrix_mod_N/gpu_mat_mul/mat_mul_gpu_direct.jl")

# Export the main type and its operations
export GPUFiniteFieldMatrix, is_invertible, inverse, unsafe_GPUFiniteFieldMatrix

# Export utility functions
export identity, zeros, rand
export change_modulus, change_modulus!
export add!, subtract!, elementwise_multiply!, negate!
export scalar_add!, scalar_subtract!, scalar_multiply!
export multiply!, copy!, mod_elements!

# Export GPU operations
export rref_gpu_type, lu_gpu_type, plup_gpu_type
export mat_mul_gpu_type, mat_mul_type_inplace!

end
