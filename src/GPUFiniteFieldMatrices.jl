module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles

const DEBUG = false

include("GpuMatrixModN/gpu_matrix_mod_N/gpu_mat.jl")
include("GpuMatrixModN/rref_lu_pluq/rref_new_kernels.jl")
#include("GpuMatrixModN/kernel_mul/mat_mul_hybrid.jl")
include("GpuMatrixModN/rref_lu_pluq/rref_gpu_direct.jl")
include("GpuMatrixModN/kernel_mul/mat_mul_gpu_direct.jl")
#include("GpuMatrixModN/kernel_mul/mat_mul_ops.jl")
#include("GpuMatrixModN/hensel_lifting/hensel.jl")

# Export the main type and its operations
export GpuMatrixModN, is_invertible, inverse, unsafe_GpuMatrixModN

# Export utility functions
export identity, zeros, rand
export change_modulus, change_modulus_no_alloc!
export add!, sub!, elementwise_multiply!, negate!
export scalar_add!, scalar_subtract!, scalar_multiply!
export multiply!, copy!, mod_elements!

# Export GPU operations
export rref_gpu_type, lu_gpu_type, plup_gpu_type
export mat_mul_gpu_type, mat_mul_type_inplace!

end
