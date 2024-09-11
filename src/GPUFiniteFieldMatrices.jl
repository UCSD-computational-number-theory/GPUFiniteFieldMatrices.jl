module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using IterTools

include("gpu_rref/rref_kernels.jl")
include("gpu_mat_mul/mat_mul_hybrid.jl")

end
