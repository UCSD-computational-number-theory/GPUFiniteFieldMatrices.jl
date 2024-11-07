using CUDA, LinearAlgebra, BenchmarkTools
using CSV, DelimitedFiles
include("rref_new_kernels.jl")
include("test_swap.jl")

function main()
    A_rows = 10
    A_cols = 10
    TILE_WIDTH = 2
    P = 7
    A = rand(1:P-1, (A_rows,A_cols))
    # A = [2 2 1 2 2; 1 2 1 2 2; 2 1 1 1 1; 2 1 2 2 1]
    # A = Matrix{Int}(I,A_rows,A_cols)

    res_U, res_L, res_Perm = 0,0,0
    println(CUDA.@profile begin
        (res_U, res_L, res_Perm) = lu_gpu(A, P)
    end)
    # println(CUDA.@profile begin
    #     (res_U, res_L, res_Perm) = lu_gpu(A, P)
    # end)
    writedlm("plu_A.csv", A, ',')
    writedlm("plu_U.csv", res_U, ',')
    writedlm("plu_L.csv", res_L, ',')
    writedlm("plu_Perm.csv", res_Perm, ',')
    writedlm("plu_res.csv", (res_L * res_U) .% P, ',')

    # println(A)
    # println(res_U)
    # println(res_L)

    Perm = zeros(Int, A_rows, A_rows)
    for i in 1:A_rows
        Perm[i, res_Perm[i]] = 1
    end

    # println(Perm)
    # println(res)

    @assert Perm * A == (res_L * res_U) .% P "Result is not equal"

end

main()