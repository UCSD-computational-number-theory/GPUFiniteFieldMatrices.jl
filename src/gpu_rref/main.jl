using CUDA, LinearAlgebra, BenchmarkTools
include("rref_kernels.jl")

function main()
    A = [
        [0 1 2 3]
        [2 3 4 4]
        [0 0 1 2]
        [0 0 0 1]
    ]
    P = 5

    DEFAULT_SIZE = 1000
    A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)

    # println("ORIGINAL:")
    # println(A)
    println(@benchmark begin
        A_rref = rref_gpu($A,$P)
    end)
    # println("REDUCED:")
    # println(A_rref)

    DEFAULT_SIZE = 1000
    A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)

    # println("ORIGINAL:")
    # println(A)
    println(@benchmark begin
        A_rref = rref_gpu($A,$P)
    end)
    # println("REDUCED:")
    # println(A_rref)

    return
end

main()