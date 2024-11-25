module GPUFiniteFieldMatrices

using CUDA
using LinearAlgebra
using IterTools
using BenchmarkTools
using CSV
using DelimitedFiles

include("gpu_rref/rref_new_kernels.jl")
include("gpu_mat_mul/mat_mul_hybrid.jl")

end
