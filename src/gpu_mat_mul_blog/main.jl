include("naive_non_coalesced.jl")
# include("naive_coalesced.jl")
# include("2d_blocks.jl")
# include("2d_blocks_1d_threads.jl")
# include("2d_blocks_2d_threads.jl")
# include("2d_blocks_2d_threads_vectorized.jl")
# include("2d_blocks_2d_threads_warp.jl")
# include("2d_blocks_2d_threads_tensor.jl")

function main()
    P = 7
    A_rows = 100
    A_cols = 100
    B_rows = A_cols
    B_cols = 100

    A = rand(1:P, (A_rows,A_cols))
    B = rand(1:P, (B_rows,B_cols))

    C = naive_non_coalesced(A, B, P, false)
    @assert C = A*B "A*B ≠ C"
end

main()