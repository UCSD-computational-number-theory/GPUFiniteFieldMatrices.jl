using CUDA, LinearAlgebra, BenchmarkTools
include("rref_new_kernels.jl")
include("test_swap.jl")

function main()
    # A = [
    #     [0 1 2 3]
    #     [2 3 4 4]
    #     [0 0 1 2]
    #     [0 0 0 1]
    # ]
    # P = 5

    # DEFAULT_SIZE = 1000
    # A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)

    # # println("ORIGINAL:")
    # # println(A)
    # println(@benchmark begin
    #     A_rref = rref_gpu($A,$P)
    # end)
    # # println("REDUCED:")
    # # println(A_rref)

    # DEFAULT_SIZE = 1000
    # A = rand(1:(P-1), DEFAULT_SIZE, DEFAULT_SIZE)

    # # println("ORIGINAL:")
    # # println(A)
    # println(@benchmark begin
    #     A_rref = rref_gpu($A,$P)
    # end)
    # # println("REDUCED:")
    # # println(A_rref)

    # return

    size = 100
    TILE_WIDTH = 32
    A = rand(1:11, (size,size))
    d_A = CUDA.CuArray(A)
    Perm = CUDA.CuArray(collect(1:size))
    d_L = CUDA.zeros(size,size)

    println(CUDA.@profile begin
        rref_gpu(A, 11)
    end)

    # println(CUDA.@profile begin
    #     CUDA.@sync @cuda threads=(100) blocks=(10) normalize(d_A,5000,1,7,11)
    # end)

    # println(CUDA.@profile begin
    #     CUDA.@sync @cuda threads=(100) blocks=(10) normalize_lu(d_A,5000,1,7,11,d_L)
    # end)

    # println(CUDA.@profile begin 
    #     CUDA.@sync normalize_broadcast(d_A,1,7,11)
    # end)

    # println(CUDA.@profile begin 
    #     CUDA.@sync normalize_lu_broadcast(d_A,1,7,11,d_L)
    # end)

end

main()