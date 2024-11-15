using CUDA, LinearAlgebra, BenchmarkTools
using CSV, DelimitedFiles
include("rref_new_kernels.jl")
include("test_swap.jl")

function main()
    A_rows = 10000
    A_cols = 10000
    TILE_WIDTH = 32
    P = 7
    A = rand(1:P-1, (A_rows,A_cols))
    # A = [2 0 1 2 2; 
    # 1 0 2 0 1; 
    # 2 0 1 1 1; 
    # 2 0 2 2 1]

    res_U, res_L, res_Perm_rows, res_Perm_cols = 0,0,0,0
    println(CUDA.@profile begin
        (res_U, res_L, res_Perm_rows, res_Perm_cols) = plup_gpu(A, P)
    end)

    writedlm("plu_A.csv", A, ',')
    writedlm("plu_U.csv", res_U, ',')
    writedlm("plu_L.csv", res_L, ',')

    Perm_rows = zeros(Int, A_rows, A_rows)
    for i in 1:A_rows
        Perm_rows[i, res_Perm_rows[i]] = 1
    end

    Perm_cols = zeros(Int, A_cols, A_cols)
    for i in 1:A_cols
        Perm_cols[i, res_Perm_cols[i]] = 1
    end

    writedlm("plu_Perm_rows.csv", Perm_rows, ',')
    writedlm("plu_Perm_cols.csv", Perm_cols, ',')

    @assert Perm_rows * A == (res_L * res_U) * Perm_cols .% P "Result is not equal"

end

main()