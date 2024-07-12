using CUDA
using LinearAlgebra
using BenchmarkTools
CUDA.allowscalar(true)

function modP!(n) 
    return mod(n, 5)
end

function gpuMod()
    # m = 10
    # n = 1
    # A = rand(Int64, m)
    A = CuArray([1,2,3,4,5])
    # d_A = CuVector(A)
    println(A)
    modP!.(A)
    println(A)
end

@cuda gpuMod()

# function cpuMod()
#     m = 10
#     n = 10
#     A = rand(Int32, m, n)
#     println(A)
#     A = modP.(A)
#     println(A)
# end

