module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles

include("gpu_rref/rref_new_kernels.jl")
include("gpu_mat_mul/mat_mul_hybrid.jl")
include("gpu_mat_type/gpu_mat.jl")
include("gpu_rref/rref_gpu_direct.jl")
include("gpu_mat_mul/mat_mul_gpu_direct.jl")

# Export the main type and its operations
export GPUFiniteFieldMatrix, is_invertible, inverse

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
