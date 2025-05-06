module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles

include("GpuMatrixModN/rref_lu_pluq/rref_new_kernels.jl")
include("GpuMatrixModN/kernel_mul/mat_mul_hybrid.jl")
include("GpuMatrixModN/gpu_matrix_mod_N/gpu_mat.jl")
include("GpuMatrixModN/rref_lu_pluq/rref_gpu_type.jl")
include("GpuMatrixModN/kernel_mul/mat_mul_gpu_direct.jl")
include("GpuMatrixModN/hensel_lifting/hensel.jl")

# Export the main type and its operations
export GpuMatrixModN, is_invertible, inverse, unsafe_GpuMatrixModN

# Export utility functions
export identity, zeros, rand
export change_modulus, change_modulus
export add!, sub!, elementwise_multiply!, negate!
export scalar_add!, scalar_sub!, scalar_multiply!
export multiply!, copy!, mod_elements!

# Export GPU operations
export rref_gpu_type, lu_gpu_type, plup_gpu_type
export mat_mul_gpu_type

end
